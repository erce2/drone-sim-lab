"""Lab 3 -- 自定义任务冒烟测试 + 手工控制器验证。

这一步是整个实操里最有价值的自检环节:
  在花 40 分钟训练一个神经网络之前, 先用手写的几何控制器把门穿过去。
  如果手写控制器都穿不过去, 那问题一定在环境/物理/坐标系, 而不是 RL。

手工控制器 (级联结构, 真实的 PX4/ArduPilot 也是这个结构) 见 scripted_controller.py。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Lab3: racing env smoke + scripted controller")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from scripted_controller import scripted_actions  # noqa: E402


def main() -> None:
    task = "Isaac-DroneRacing-Direct-v0"
    print("[LAB3] SCRIPT_VERSION=3")
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(task, cfg=env_cfg)
    u = env.unwrapped

    print("[LAB3] num_gates          :", u.cfg.num_gates)
    print("[LAB3] gate_x             :", u.cfg.gate_x)
    print("[LAB3] gate window (w x h):", 2 * u.cfg.gate_half_width, "x", 2 * u.cfg.gate_half_height)
    print("[LAB3] obs shape          :", env.observation_space.shape)
    print("[LAB3] action shape       :", env.action_space.shape)
    print("[LAB3] robot mass         :", round(float(u._robot_mass), 5), "kg")
    print("[LAB3] env_origins[:3]    :", u._terrain.env_origins[:3].tolist())

    obs, _ = env.reset()
    print("[LAB3] gate centres (env-local):")
    origins = u._terrain.env_origins
    for i in range(min(3, u.num_envs)):
        g = (u._gate_pos_w[i] - origins[i]).cpu()
        print(f"        env{i}: x={g[:, 0].tolist()} y={[round(v, 2) for v in g[:, 1].tolist()]} "
              f"z={[round(v, 2) for v in g[:, 2].tolist()]}")
    scripted_actions(u)  # 先跑一次, 让它从仿真里读出惯量并缓存
    print("[LAB3] inertia scale      :", f"{u._scripted_inertia:.3e}")

    # ------------------- 手工控制器跑一整回合 -------------------
    n_done = 0
    for step in range(args_cli.steps):
        actions = scripted_actions(u)
        obs, rew, terminated, truncated, extras = env.step(actions)
        done = terminated | truncated
        if done.any():
            n_done += int(done.sum())
            logs = extras.get("log", {})
            print(f"[LAB3] step {step:4d}: resets={int(done.sum())}, "
                  f"gates_passed(mean)={float(u._gates_passed.mean()):.2f}, "
                  f"success_rate={logs.get('Metrics/success_rate', float('nan')):.3f}")
        if step < 60 or step % 50 == 0:
            pos = u._robot.data.root_pos_w - origins
            # projected_gravity_b = R^T @ (0,0,-1) = -b3  -> 倾角 = arccos(-pg_z)
            tilt = float(torch.rad2deg(torch.arccos((-u._robot.data.projected_gravity_b[0, 2]).clamp(-1, 1))))
            speed = float(u._robot.data.root_lin_vel_w[0].norm())
            om = u._robot.data.root_ang_vel_b[0].cpu().numpy()
            a0 = float(actions[0, 0])
            m123 = actions[0, 1:].cpu().numpy()
            print(f"[LAB3] step {step:4d} | pos={[round(v, 2) for v in pos[0].tolist()]} "
                  f"| gate={int(u._gate_idx[0])} | tilt={tilt:5.1f}deg | |v|={speed:.2f} "
                  f"| w=[{om[0]:+.1f},{om[1]:+.1f},{om[2]:+.1f}] | a0={a0:+.2f} m=[{m123[0]:+.2f},{m123[1]:+.2f},{m123[2]:+.2f}]")

    print("[LAB3] ---- scripted controller result ----")
    print("[LAB3] gates passed per env:", u._gates_passed.cpu().numpy().astype(int).tolist())
    print("[LAB3] success flags       :", u._episode_success.cpu().numpy().astype(int).tolist())
    n_pass = int((u._gates_passed >= u.cfg.num_gates).sum())
    print(f"[LAB3] environments that finished the course: {n_pass}/{u.num_envs}")
    print(f"[LAB3] total resets during run: {n_done}")

    env.close()
    print("[LAB3] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()