"""Lab 1 -- 冒烟测试: 验证 Isaac Lab 全链路是否打通.

目的 (也是你以后每次接手一台新 GPU 服务器时该做的第一件事):
  1) Isaac Sim (Kit) 能启动, PhysX/CUDA 正常
  2) 机器人 USD 资产能从"离线资产根"加载 (这是最常见的失败点)
  3) DirectRLEnv + 4096 并行环境能 step, 并测出真实吞吐

运行:
  call C:\\IsaacLab\\_isaac_sim\\python.bat -u lab1_smoke.py --num_envs 64 --headless
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Lab1 smoke test")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 关键: 必须在导入 isaaclab / torch 之前先把 Kit 拉起来
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  -- 导入即注册所有 Isaac-* 任务
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    task = "Isaac-Quadcopter-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42

    env = gym.make(task, cfg=env_cfg)

    u = env.unwrapped
    print("[LAB1] task            :", task)
    print("[LAB1] num_envs        :", u.num_envs)
    print("[LAB1] action_space    :", env.action_space)
    print("[LAB1] observation_space:", env.observation_space)
    print("[LAB1] sim dt          :", u.sim.cfg.dt)
    print("[LAB1] decimation      :", u.cfg.decimation, "-> control dt", u.step_dt)
    print("[LAB1] max_episode_len :", u.max_episode_length, "steps")
    print("[LAB1] episode_length_s:", u.max_episode_length_s)

    obs, _ = env.reset()
    print("[LAB1] obs['policy'].shape:", obs["policy"].shape, obs["policy"].dtype)
    print("[LAB1] robot mass (kg) :", round(float(u._robot_mass), 5))
    print("[LAB1] robot weight (N):", round(float(u._robot_weight), 4))
    # 悬停需要的归一化油门: thrust = 1.9 * W * (a+1)/2  == W  =>  a = 2/1.9 - 1
    print("[LAB1] hover throttle  :", round(2.0 / u.cfg.thrust_to_weight - 1.0, 4))

    # ---- 随机动作跑 200 步, 测吞吐, 顺便看物理有没有"炸" ----
    n_steps = 200
    actions = torch.zeros((u.num_envs, u.single_action_space.shape[0]), device=u.device)
    t0 = time.perf_counter()
    for i in range(n_steps):
        actions.uniform_(-0.2, 0.2)
        actions[:, 0] = 2.0 / u.cfg.thrust_to_weight - 1.0  # 保持接近悬停, 只抖其余通道
        env.step(actions)
    t1 = time.perf_counter()

    wall = t1 - t0
    print(f"[LAB1] {n_steps} steps in {wall:.2f}s -> {n_steps / wall:.1f} env-steps/s "
          f"({n_steps * u.num_envs / wall:.0f} env-steps/s across {u.num_envs} envs)")
    print("[LAB1] mean root height:", round(float(u._robot.data.root_pos_w[:, 2].mean()), 3), "m")
    print("[LAB1] mean rew (last) :", round(float(u._robot.data.root_lin_vel_b.norm(dim=-1).mean()), 3), "m/s")

    env.close()
    print("[LAB1] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()