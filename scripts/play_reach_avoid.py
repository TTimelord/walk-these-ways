import argparse
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


def load_env(label, headless=False, scene=None):
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
    Cfg.env.num_envs = 1
    Cfg.env.record_video = True
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = ReachAvoidEnv(sim_device='cuda:0', headless=headless, cfg=Cfg, scene_cfg=scene, auto_reset=True)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True, scene=None, target_velocity=(1.5, 0.0, 0.0), num_eval_steps=250,
             stop_on_collision=False, stop_on_goal_reached=False, output_dir=None,
             record_video=None, save_plots=None):
    label = "gait-conditioned-agility/pretrain-v0/train"

    if scene is None:
        scene = make_default_scene()

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

    env, policy = load_env(label, headless=headless, scene=scene)
    base_env = unwrap_base_env(env)

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = target_velocity
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"], device=base_env.device, dtype=torch.float)
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    base_positions = np.zeros((num_eval_steps, 3))
    base_yaws = np.zeros(num_eval_steps)
    collision_flags = np.zeros(num_eval_steps, dtype=np.int32)
    goal_flags = np.zeros(num_eval_steps, dtype=np.int32)

    obs = env.reset()

    frames = []
    if record_video:
        frames.append(capture_frame(base_env))

    collision_step = None
    collision_name = None
    goal_step = None
    goal_name = None
    actual_steps = 0

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
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

        measured_x_vels[i] = float(base_env.base_lin_vel[0, 0].detach().cpu())
        joint_positions[i] = base_env.dof_pos[0, :].detach().cpu().numpy()
        base_positions[i] = base_env.root_states[0, 0:3].detach().cpu().numpy()
        base_quat = base_env.root_states[0, 3:7].detach().cpu().numpy()
        base_yaws[i] = quat_xyzw_to_yaw(base_quat)
        if record_video:
            frames.append(capture_frame(base_env))

        collision_now = bool(scene_info.get("collision", False))
        collision_hit = scene_info.get("collision_name")
        goal_now = bool(scene_info.get("goal_reached", False))
        goal_hit = scene_info.get("goal_name")
        collision_flags[i] = int(collision_now)
        goal_flags[i] = int(goal_now)
        if collision_now and collision_step is None:
            collision_step = i
            collision_name = collision_hit
        if goal_now and goal_step is None:
            goal_step = i
            goal_name = goal_hit

        actual_steps = i + 1
        if (stop_on_collision and collision_now) or (stop_on_goal_reached and goal_now):
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

    # plot target and measured forward velocity
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    axs[0].plot(time, measured_x_vels[valid], color='black', linestyle="-", label="Measured")
    axs[0].plot(time, target_x_vels[valid], color='black', linestyle="--", label="Desired")
    if goal_step is not None:
        axs[0].axvline(time[goal_step], color='tab:green', linestyle=':', alpha=0.8, label="Goal reached")
    if collision_step is not None:
        axs[0].axvline(time[collision_step], color='tab:red', linestyle=':', alpha=0.8, label="Collision")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(time, joint_positions[valid], linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(time, base_positions[valid, 0], linestyle="-", label="x")
    axs[2].plot(time, base_positions[valid, 1], linestyle="-", label="y")
    axs[2].plot(time, base_positions[valid, 2], linestyle="-", label="z")
    axs[2].legend()
    axs[2].set_title("Base Position")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Position (m)")

    axs[3].plot(time, base_yaws[valid], linestyle="-", color='tab:orange')
    axs[3].set_title("Base Yaw Angle")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Yaw (rad)")

    axs[4].step(time, collision_flags[valid], where="post", color="tab:red", label="Collision")
    axs[4].step(time, goal_flags[valid], where="post", color="tab:green", label="Goal reached")
    axs[4].set_ylim(-0.1, 1.1)
    axs[4].set_title("Scene Checks")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Flag")
    axs[4].legend(loc="upper right")

    plt.tight_layout()
    if save_plots and artifacts_dir is not None:
        plot_path = artifacts_dir / "rollout_metrics.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved plots to {plot_path}")
    if headless:
        plt.close(fig)
    else:
        plt.show()

    print(f"Collision detected: {collision_step is not None}" +
          (f" at step {collision_step} ({collision_name})" if collision_step is not None else ""))
    print(f"Goal reached: {goal_step is not None}" +
          (f" at step {goal_step} ({goal_name})" if goal_step is not None else ""))

    return {
        "collision_step": collision_step,
        "collision_name": collision_name,
        "goal_step": goal_step,
        "goal_name": goal_name,
        "collision_flags": collision_flags[:actual_steps],
        "goal_flags": goal_flags[:actual_steps],
        "base_positions": base_positions[:actual_steps],
        "base_yaws": base_yaws[:actual_steps],
        "measured_x_vels": measured_x_vels[:actual_steps],
        "target_x_vels": target_x_vels[:actual_steps],
        "artifacts_dir": str(artifacts_dir) if artifacts_dir is not None else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and save artifacts to disk.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots and video.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout video even when not headless.")
    parser.add_argument("--save-plots", action="store_true", help="Save rollout plots even when not headless.")
    parser.add_argument("--steps", type=int, default=250, help="Number of rollout steps.")
    parser.add_argument("--stop-on-collision", action="store_true", help="Stop early when a collision is detected.")
    parser.add_argument("--stop-on-goal-reached", action="store_true", help="Stop early when the goal is reached.")
    args = parser.parse_args()

    play_go1(
        headless=args.headless,
        num_eval_steps=args.steps,
        stop_on_collision=args.stop_on_collision,
        stop_on_goal_reached=args.stop_on_goal_reached,
        output_dir=args.output_dir,
        record_video=args.save_video,
        save_plots=args.save_plots,
    )
