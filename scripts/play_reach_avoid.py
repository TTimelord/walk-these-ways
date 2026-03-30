import argparse
import json
from datetime import datetime
import isaacgym

assert isaacgym
import torch
import numpy as np
import imageio.v2 as imageio

import glob
import pickle as pkl
from pathlib import Path

from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.reach_avoid_env import ReachAvoidEnv, make_default_scene

from tqdm import tqdm


def quat_xyzw_to_yaw(quat_xyzw):
    x, y, z, w = quat_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def unwrap_base_env(env):
    return env.env if hasattr(env, "env") else env


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_target_velocities(target_velocity, target_velocities, num_envs):
    if target_velocities is None:
        target_velocities = np.asarray(target_velocity, dtype=np.float32)
        if target_velocities.shape != (3,):
            raise ValueError("target_velocity must be a length-3 vector.")
        target_velocities = np.repeat(target_velocities[None, :], num_envs, axis=0)
    else:
        target_velocities = np.asarray(target_velocities, dtype=np.float32)
        if target_velocities.ndim == 1:
            if target_velocities.shape[0] != 3:
                raise ValueError("target_velocities must be a list of [x, y, yaw] triples.")
            target_velocities = np.repeat(target_velocities[None, :], num_envs, axis=0)
        if target_velocities.shape != (num_envs, 3):
            raise ValueError(f"target_velocities must have shape ({num_envs}, 3).")
    return target_velocities


def make_output_dir(output_dir=None):
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(__file__).resolve().parents[1] / "outputs" / "reach_avoid" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def capture_frame(base_env):
    frame = np.asarray(base_env.render(mode="rgb_array"), dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.copy()


def write_video(frames, video_path, fps):
    if len(frames) == 0:
        return
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)


def load_env(label, headless=False, scene=None, num_envs=1):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = num_envs
    Cfg.env.record_video = True
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    # Parallel eval uses a simple grid on a plane so all envs stay visible.
    Cfg.terrain.center_robots = num_envs == 1
    Cfg.terrain.center_span = 1 if num_envs == 1 else 0
    if num_envs > 1:
        Cfg.terrain.mesh_type = "plane"
        Cfg.env.env_spacing = 5.0
        Cfg.terrain.teleport_robots = False
    else:
        Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = ReachAvoidEnv(sim_device='cuda:0', headless=headless, cfg=Cfg, scene_cfg=scene, auto_reset=True)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True, scene=None, target_velocity=(1.5, 0.0, 0.0), target_velocities=None,
             num_envs=1, num_eval_steps=250, stop_on_collision=False, stop_on_goal_reached=False,
             output_dir=None, record_video=None, save_plots=None):
    label = "gait-conditioned-agility/pretrain-v0/train"

    if scene is None:
        scene = make_default_scene()

    target_velocities = normalize_target_velocities(target_velocity, target_velocities, num_envs)

    if record_video is None:
        record_video = headless
    if save_plots is None:
        save_plots = headless
    if output_dir is not None:
        record_video = True
        save_plots = True

    artifacts_dir = None
    if record_video or save_plots:
        artifacts_dir = make_output_dir(output_dir)

    env, policy = load_env(label, headless=headless, scene=scene, num_envs=num_envs)
    base_env = unwrap_base_env(env)
    num_envs = base_env.num_envs

    if not headless and num_envs > 1:
        env_origins = base_env.env_origins.detach().cpu().numpy()
        center_xy = env_origins[:, :2].mean(axis=0)
        span_xy = np.ptp(env_origins[:, :2], axis=0)
        camera_dist = max(8.0, 1.5 * float(np.max(span_xy)) + 4.0)
        camera_height = max(8.0, camera_dist)
        base_env.set_camera(
            [float(center_xy[0]), float(center_xy[1] - camera_dist), float(camera_height)],
            [float(center_xy[0]), float(center_xy[1]), 0.6],
        )

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    target_vel_tensor = torch.tensor(target_velocities, device=base_env.device, dtype=torch.float)
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"], device=base_env.device, dtype=torch.float)
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros((num_eval_steps, num_envs), dtype=np.float32)
    target_x_vels = np.repeat(target_velocities[None, :, 0], num_eval_steps, axis=0)
    joint_positions = np.zeros((num_eval_steps, num_envs, 12), dtype=np.float32)
    base_positions = np.zeros((num_eval_steps, num_envs, 3), dtype=np.float32)
    base_yaws = np.zeros((num_eval_steps, num_envs), dtype=np.float32)
    collision_flags = np.zeros((num_eval_steps, num_envs), dtype=np.int32)
    goal_flags = np.zeros((num_eval_steps, num_envs), dtype=np.int32)
    collision_step = [None for _ in range(num_envs)]
    collision_name = [None for _ in range(num_envs)]
    goal_step = [None for _ in range(num_envs)]
    goal_name = [None for _ in range(num_envs)]

    obs = env.reset()

    frames = []
    if record_video:
        frames.append(capture_frame(base_env))

    actual_steps = 0
    plot_env = 0

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0:3] = target_vel_tensor
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)
        scene_info = info.get("scene", {})

        measured_x_vels[i] = base_env.base_lin_vel[:, 0].detach().cpu().numpy()
        joint_positions[i] = base_env.dof_pos[:, :].detach().cpu().numpy()
        base_positions[i] = base_env.base_pos.detach().cpu().numpy()
        base_quat = base_env.base_quat.detach().cpu().numpy()
        base_yaws[i] = np.array([quat_xyzw_to_yaw(q) for q in base_quat], dtype=np.float32)
        if record_video:
            frames.append(capture_frame(base_env))

        collision_now = as_numpy(scene_info.get("collision", np.zeros(num_envs, dtype=bool))).astype(bool)
        goal_now = as_numpy(scene_info.get("goal_reached", np.zeros(num_envs, dtype=bool))).astype(bool)
        if collision_now.ndim == 0:
            collision_now = np.full((num_envs,), bool(collision_now), dtype=bool)
        if goal_now.ndim == 0:
            goal_now = np.full((num_envs,), bool(goal_now), dtype=bool)
        if collision_now.shape[0] != num_envs:
            collision_now = np.reshape(collision_now, (num_envs,))
        if goal_now.shape[0] != num_envs:
            goal_now = np.reshape(goal_now, (num_envs,))

        collision_hit = scene_info.get("collision_name", [None] * num_envs)
        goal_hit = scene_info.get("goal_name", [None] * num_envs)
        if not isinstance(collision_hit, (list, tuple)):
            collision_hit = [collision_hit] * num_envs
        if not isinstance(goal_hit, (list, tuple)):
            goal_hit = [goal_hit] * num_envs

        collision_flags[i] = collision_now.astype(np.int32)
        goal_flags[i] = goal_now.astype(np.int32)
        for env_id in range(num_envs):
            if collision_now[env_id] and collision_step[env_id] is None:
                collision_step[env_id] = i
                collision_name[env_id] = collision_hit[env_id]
            if goal_now[env_id] and goal_step[env_id] is None:
                goal_step[env_id] = i
                goal_name[env_id] = goal_hit[env_id]

        actual_steps = i + 1
        if (stop_on_collision and np.any(collision_now)) or (stop_on_goal_reached and np.any(goal_now)):
            break

    if actual_steps == 0:
        actual_steps = num_eval_steps

    time = np.linspace(0, actual_steps * base_env.dt, actual_steps, endpoint=False)
    valid = slice(0, actual_steps)

    if record_video and artifacts_dir is not None:
        fps = max(1, int(round(1.0 / base_env.dt)))
        video_path = artifacts_dir / "rollout.mp4"
        write_video(frames[:actual_steps + 1], video_path, fps)
        print(f"Saved video to {video_path}")

    base_xy = base_positions[:actual_steps, :, :2]
    base_xy_rel = base_xy - base_xy[0:1, :, :]
    base_yaws_valid = base_yaws[:actual_steps]

    # plot target and measured forward velocity
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(5, 1, figsize=(12, 16))
    traj_fig, traj_ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_envs, 1)))
    axs[0].plot(time, measured_x_vels[valid, plot_env], color=colors[plot_env % len(colors)], linestyle="-",
                label=f"Measured env {plot_env}")
    axs[0].plot(time, target_x_vels[valid, plot_env], color=colors[plot_env % len(colors)], linestyle="--",
                label=f"Desired env {plot_env}")
    if goal_step[plot_env] is not None:
        axs[0].axvline(time[goal_step[plot_env]], color=colors[plot_env % len(colors)], linestyle=':', alpha=0.5)
    if collision_step[plot_env] is not None:
        axs[0].axvline(time[collision_step[plot_env]], color=colors[plot_env % len(colors)], linestyle='-.', alpha=0.5)
    axs[0].legend()
    axs[0].set_title(f"Forward Linear Velocity (env {plot_env})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(time, joint_positions[valid, plot_env], linestyle="-", label=f"Measured env {plot_env}")
    axs[1].set_title(f"Joint Positions (env {plot_env})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(time, base_positions[valid, plot_env, 0], linestyle="-", label="x")
    axs[2].plot(time, base_positions[valid, plot_env, 1], linestyle="-", label="y")
    axs[2].plot(time, base_positions[valid, plot_env, 2], linestyle="-", label="z")
    axs[2].legend()
    axs[2].set_title(f"Base Position (env {plot_env})")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Position (m)")

    arrow_step = max(1, int(round(1.0 / base_env.dt)))
    arrow_length = 0.25
    for env_id in range(num_envs):
        traj_ax.plot(
            base_xy_rel[:, env_id, 0],
            base_xy_rel[:, env_id, 1],
            color=colors[env_id % len(colors)],
            linewidth=1.5,
            label=f"Env {env_id}",
        )
        arrow_indices = np.arange(0, actual_steps, arrow_step, dtype=int)
        if len(arrow_indices) > 0:
            yaws = base_yaws_valid[arrow_indices, env_id]
            traj_ax.quiver(
                base_xy_rel[arrow_indices, env_id, 0],
                base_xy_rel[arrow_indices, env_id, 1],
                np.cos(yaws) * arrow_length,
                np.sin(yaws) * arrow_length,
                color=colors[env_id % len(colors)],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                alpha=0.7,
            )

    traj_ax.scatter([0.0], [0.0], color="black", s=35, marker="o", label="Start")
    traj_ax.set_aspect("equal", adjustable="box")
    traj_ax.set_title("Base Trajectories Relative to Initial Position")
    traj_ax.set_xlabel("Relative X (m)")
    traj_ax.set_ylabel("Relative Y (m)")
    traj_ax.legend(loc="best", ncol=2)

    axs[3].plot(time, base_yaws[valid, plot_env], linestyle="-", color='tab:orange')
    axs[3].set_title(f"Base Yaw Angle (env {plot_env})")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Yaw (rad)")

    axs[4].step(time, collision_flags[valid, plot_env], where="post", color="tab:red", label="Collision")
    axs[4].step(time, goal_flags[valid, plot_env], where="post", color="tab:green", label="Goal reached")
    axs[4].set_ylim(-0.1, 1.1)
    axs[4].set_title(f"Scene Checks (env {plot_env})")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Flag")
    axs[4].legend(loc="upper right")

    fig.tight_layout()
    traj_fig.tight_layout()
    if save_plots and artifacts_dir is not None:
        plot_path = artifacts_dir / "rollout_metrics.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plots to {plot_path}")
        traj_plot_path = artifacts_dir / "rollout_trajectory.png"
        traj_fig.savefig(traj_plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved trajectory plot to {traj_plot_path}")
    if headless:
        plt.close(fig)
        plt.close(traj_fig)
    else:
        plt.show()

    collision_summary = ", ".join(
        f"env {env_id}: step {collision_step[env_id]} ({collision_name[env_id]})"
        for env_id in range(num_envs)
        if collision_step[env_id] is not None
    )
    goal_summary = ", ".join(
        f"env {env_id}: step {goal_step[env_id]} ({goal_name[env_id]})"
        for env_id in range(num_envs)
        if goal_step[env_id] is not None
    )
    print(f"Collision detected: {bool(collision_summary)}" + (f" [{collision_summary}]" if collision_summary else ""))
    print(f"Goal reached: {bool(goal_summary)}" + (f" [{goal_summary}]" if goal_summary else ""))

    return {
        "collision_step": collision_step,
        "collision_name": collision_name,
        "goal_step": goal_step,
        "goal_name": goal_name,
        "collision_steps": collision_step,
        "collision_names": collision_name,
        "goal_steps": goal_step,
        "goal_names": goal_name,
        "collision_flags": collision_flags[:actual_steps],
        "goal_flags": goal_flags[:actual_steps],
        "base_positions": base_positions[:actual_steps],
        "base_yaws": base_yaws[:actual_steps],
        "measured_x_vels": measured_x_vels[:actual_steps],
        "target_x_vels": target_x_vels[:actual_steps],
        "target_velocities": target_velocities,
        "artifacts_dir": str(artifacts_dir) if artifacts_dir is not None else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and save artifacts to disk.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots and video.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout video even when not headless.")
    parser.add_argument("--save-plots", action="store_true", help="Save rollout plots even when not headless.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--target-velocity", nargs=3, type=float, default=[1.5, 0.0, 0.0],
                        help="Default [x y yaw] target velocity broadcast to all envs.")
    parser.add_argument("--target-velocities", type=str, default=None,
                        help="JSON list of per-env [x, y, yaw] target velocities.")
    parser.add_argument("--steps", type=int, default=250, help="Number of rollout steps.")
    parser.add_argument("--stop-on-collision", action="store_true", help="Stop early when a collision is detected.")
    parser.add_argument("--stop-on-goal-reached", action="store_true", help="Stop early when the goal is reached.")
    args = parser.parse_args()

    parsed_target_velocities = None
    if args.target_velocities is not None:
        parsed_target_velocities = json.loads(args.target_velocities)

    play_go1(
        headless=args.headless,
        target_velocity=tuple(args.target_velocity),
        target_velocities=parsed_target_velocities,
        num_envs=args.num_envs,
        num_eval_steps=args.steps,
        stop_on_collision=args.stop_on_collision,
        stop_on_goal_reached=args.stop_on_goal_reached,
        output_dir=args.output_dir,
        record_video=args.save_video,
        save_plots=args.save_plots,
    )
