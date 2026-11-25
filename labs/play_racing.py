"""载入训练好的策略, 渲染一段视频 (门用 VisualizationMarkers 画出来)。

用法:
  python play_racing.py --checkpoint <path.pt> --out C:/work/drone_lab/out/videos --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--out", type=str, default="C:/work/drone_lab/out/videos")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def main() -> None:
    task = "Isaac-DroneRacing-Direct-v0"
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env_cfg.seed = 7
    env_cfg.debug_vis = True  # 画出所有门

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    u = env.unwrapped

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    print("[PLAY] loaded policy from", args_cli.checkpoint)

    # 把观察视角摆到赛道侧面
    origin = u._terrain.env_origins[0].cpu().numpy()
    u.sim.set_camera_view(eye=origin + np.array([1.0, -5.0, 2.0]), target=origin + np.array([3.5, 0.0, 0.9]))

    video_dir = args_cli.out
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda s: s == 0,
        video_length=args_cli.steps,
        name_prefix="drone_racing",
        disable_logger=True,
    )
    obs, _ = env.reset()

    from tensordict import TensorDict

    for i in range(args_cli.steps + 3):
        # rsl_rl 的 act_inference 期望 TensorDict (按 obs group 取), 不是裸张量
        if isinstance(obs, dict):
            o = TensorDict({k: v for k, v in obs.items()}, batch_size=[u.num_envs])
        else:
            o = obs
        with torch.inference_mode():
            actions = policy(o)
        obs, _, terminated, truncated, _ = env.step(actions)
        if i % 40 == 0:
            p = (u._robot.data.root_pos_w[0] - u._terrain.env_origins[0]).cpu().numpy()
            print(f"[PLAY] step {i:3d} gate={int(u._gate_idx[0])} pos={p.round(2)}")

    env.close()
    print("[PLAY] DONE ->", video_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()