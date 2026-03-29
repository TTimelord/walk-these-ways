# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import copy
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv


DEFAULT_SCENE = {
    "obstacles": [
        {
            "name": "obstacle_0",
            "position": [2.6, 0.0, 0.0],
            "radius": 0.18,
            "height": 0.50,
        },
    ],
    "goals": [
        {
            "name": "goal_0",
            "position": [1.3, 0.0, 0.0],
            "radius": 0.35,
            "height": 0.03,
        },
    ],
}


def make_default_scene():
    return copy.deepcopy(DEFAULT_SCENE)


class ReachAvoidEnv(VelocityTrackingEasyEnv):
    """Velocity-tracking env with optional obstacle and goal scene objects."""

    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        prone=False,
        deploy=False,
        cfg=None,
        eval_cfg=None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
        scene_cfg=None,
        collision_margin=0.12,
        goal_min_base_height=0.18,
        auto_reset=False,
    ):
        if scene_cfg is None:
            scene_cfg = DEFAULT_SCENE
        self.scene_cfg = copy.deepcopy(scene_cfg)
        self.scene_collision_margin = float(collision_margin)
        self.goal_min_base_height = float(goal_min_base_height)
        self.auto_reset = bool(auto_reset)
        self.scene_objects = None
        self.scene_status = {}
        self.scene_asset_cache = {}

        super().__init__(
            sim_device=sim_device,
            headless=headless,
            num_envs=num_envs,
            prone=prone,
            deploy=deploy,
            cfg=cfg,
            eval_cfg=eval_cfg,
            initial_dynamics_dict=initial_dynamics_dict,
            physics_engine=physics_engine,
        )

    def _scene_asset_path(self, kind, radius, height, visual_only):
        asset_dir = Path(tempfile.gettempdir()) / "walk_these_ways_scene_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        suffix = "goal" if visual_only else "obstacle"
        safe_radius = f"{radius:.4f}".replace(".", "p")
        safe_height = f"{height:.4f}".replace(".", "p")
        asset_path = asset_dir / f"{kind}_{suffix}_r{safe_radius}_h{safe_height}.urdf"
        if asset_path.exists():
            return asset_path

        material_color = "0.10 0.80 0.20 1.0" if visual_only else "0.95 0.55 0.10 1.0"
        link_name = suffix
        collision_xml = ""
        if not visual_only:
            collision_xml = f"""
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <cylinder length=\"{height:.6f}\" radius=\"{radius:.6f}\"/>
      </geometry>
    </collision>"""

        asset_path.write_text(
            f"""<?xml version=\"1.0\"?>
<robot name=\"{kind}_{suffix}\">
  <material name=\"{suffix}\">
    <color rgba=\"{material_color}\"/>
  </material>
  <link name=\"{link_name}\">
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <cylinder length=\"{height:.6f}\" radius=\"{radius:.6f}\"/>
      </geometry>
      <material name=\"{suffix}\"/>
    </visual>{collision_xml}
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"0.5\"/>
      <inertia ixx=\"0.01\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.01\" iyz=\"0.0\" izz=\"0.01\"/>
    </inertial>
  </link>
</robot>
"""
        )
        return asset_path

    def _load_scene_asset(self, kind, radius, height, visual_only=False):
        asset_path = self._scene_asset_path(kind, radius, height, visual_only)
        asset_root = str(asset_path.parent)
        asset_file = asset_path.name

        cache_key = (kind, float(radius), float(height), bool(visual_only))
        if cache_key in self.scene_asset_cache:
            return self.scene_asset_cache[cache_key]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.replace_cylinder_with_capsule = False
        if hasattr(asset_options, "use_mesh_materials"):
            asset_options.use_mesh_materials = True

        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.scene_asset_cache[cache_key] = asset
        return asset

    def _spawn_scene_actor(self, env_handle, env_id, asset, entry, kind):
        collection_key = "obstacles" if kind == "obstacle" else "goals"
        local_position = np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        height = float(entry.get("height", 0.0))

        world_position = self.env_origins[env_id].detach().cpu().numpy().astype(np.float32)
        world_position = world_position + local_position
        world_position[2] += 0.5 * height

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*world_position.tolist())

        actor_name = entry.get("name", f"{kind}_{len(self.scene_objects[collection_key][env_id])}")
        actor_handle = self.gym.create_actor(
            env_handle,
            asset,
            pose,
            f"{actor_name}_env{env_id}",
            env_id,
            0,
            0,
        )

        self.scene_objects[collection_key][env_id].append(
            {
                "name": actor_name,
                "kind": kind,
                "env_id": env_id,
                "position": local_position,
                "world_position": world_position,
                "radius": float(entry.get("radius", 0.0)),
                "height": height,
                "actor_handle": actor_handle,
            }
        )

    def _create_envs(self):
        """Create robots and optional reach/avoid scene objects."""
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        self.scene_objects = {
            "obstacles": [[] for _ in range(self.num_envs)],
            "goals": [[] for _ in range(self.num_envs)],
        }

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(
                -self.cfg.terrain.x_init_range,
                self.cfg.terrain.x_init_range,
                (1, 1),
                device=self.device,
            ).squeeze(1)
            pos[1:2] += torch_rand_float(
                -self.cfg.terrain.y_init_range,
                self.cfg.terrain.y_init_range,
                (1, 1),
                device=self.device,
            ).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                self.robot_asset,
                start_pose,
                "anymal",
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            for obstacle in self.scene_cfg.get("obstacles", []):
                obstacle_asset = self._load_scene_asset(
                    "cylinder",
                    obstacle.get("radius", 0.0),
                    obstacle.get("height", 0.0),
                    visual_only=False,
                )
                self._spawn_scene_actor(env_handle, i, obstacle_asset, obstacle, "obstacle")
            for goal in self.scene_cfg.get("goals", []):
                goal_asset = self._load_scene_asset(
                    "cylinder",
                    goal.get("radius", 0.0),
                    goal.get("height", 0.0),
                    visual_only=True,
                )
                self._spawn_scene_actor(env_handle, i, goal_asset, goal, "goal")

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(
                self.rendering_camera,
                self.envs[0],
                gymapi.Vec3(1.5, 1, 3.0),
                gymapi.Vec3(0, 0, 0),
            )
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(
                    self.envs[self.num_train_envs], self.camera_props
                )
                self.gym.set_camera_location(
                    self.rendering_camera_eval,
                    self.envs[self.num_train_envs],
                    gymapi.Vec3(1.5, 1, 3.0),
                    gymapi.Vec3(0, 0, 0),
                )
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def check_collision(self, env_id=0):
        if not self.scene_objects or len(self.scene_objects["obstacles"][env_id]) == 0:
            return False, None

        body_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[env_id, :, 0:3]
        for obstacle in self.scene_objects["obstacles"][env_id]:
            center = torch.as_tensor(obstacle["world_position"], device=body_positions.device, dtype=body_positions.dtype)
            radius = float(obstacle["radius"]) + self.scene_collision_margin
            height = float(obstacle["height"])
            radial_dist = torch.linalg.norm(body_positions[:, :2] - center[:2], dim=-1)
            within_xy = radial_dist <= radius
            bottom = center[2] - 0.5 * height
            top = center[2] + 0.5 * height
            within_z = (body_positions[:, 2] >= bottom) & (body_positions[:, 2] <= top)
            if torch.any(within_xy & within_z):
                return True, obstacle["name"]
        return False, None

    def check_goal_reached(self, env_id=0):
        if not self.scene_objects or len(self.scene_objects["goals"][env_id]) == 0:
            return False, None

        base_pos = self.base_pos[env_id]
        for goal in self.scene_objects["goals"][env_id]:
            center = torch.as_tensor(goal["world_position"], device=base_pos.device, dtype=base_pos.dtype)
            radius = float(goal["radius"])
            if torch.linalg.norm(base_pos[:2] - center[:2]) <= radius and base_pos[2] >= self.goal_min_base_height:
                return True, goal["name"]
        return False, None

    def get_scene_status(self, env_id=0):
        collision, collision_name = self.check_collision(env_id=env_id)
        goal_reached, goal_name = self.check_goal_reached(env_id=env_id)
        return {
            "collision": collision,
            "collision_name": collision_name,
            "goal_reached": goal_reached,
            "goal_name": goal_name,
        }

    def check_termination(self):
        """Task-specific termination: timeout, collision, or goal reached."""
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf = self.time_out_buf.clone()
        if self.scene_status.get("collision", False) or self.scene_status.get("goal_reached", False):
            self.reset_buf |= torch.ones_like(self.reset_buf, dtype=torch.bool)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        self._post_physics_step_callback()
        self.scene_status = self.get_scene_status(env_id=0)
        self.extras["scene"] = self.scene_status

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.auto_reset:
            self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()
