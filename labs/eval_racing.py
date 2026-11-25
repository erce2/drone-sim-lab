"""评测: 载入训练好的策略 (或手写控制器), 测成功率 / 过门数, 并导出单回合轨迹。

统一走 RslRlVecEnvWrapper: 它的 reset/step 直接返回 obs 张量, 这是 Isaac Lab 官方的
play.py 也在用的接口, 避免自己猜 obs 的字典结构。

用法:
  python eval_racing.py --controller rl --checkpoint <path.pt> --num_envs 32 --steps 800
  python eval_racing.py --controller scripted --num_envs 32 --steps 800
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--controller", type=str, default="rl", choices=["rl", "scripted"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--out", type=str, default="C:/work/drone_lab/out")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from scripted_controller import scripted_actions  # noqa: E402


def unpack_step(out):
    """兼容 (obs, rew, dones, extras) 和 (obs, rew, term, trunc, extras) 两种接口。"""
    if len(out) == 5:
        obs, rew, terminated, truncated, extras = out
        return obs, rew, terminated | truncated, extras
    obs, rew, dones, extras = out
    return obs, rew, dones, extras


def main() -> None:
    task = "Isaac-DroneRacing-Direct-v0"
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 123
    raw_env = gym.make(task, cfg=env_cfg)
    u = raw_env.unwrapped
    env = RslRlVecEnvWrapper(raw_env)
    print("[EVAL] wrapper obs space:", env.observation_space)

    policy = None
    if args_cli.controller == "rl":
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args_cli.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        print("[EVAL] loaded policy:", args_cli.checkpoint)

    obs, _ = env.reset()
    print("[EVAL] obs type:", type(obs).__name__, "shape:", getattr(obs, "shape", None))

    traj = []
    recording = True
    n_reset = 0
    sum_success = 0.0
    sum_gates = 0.0
    sum_len = 0.0

    for step in range(args_cli.steps):
        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
        else:
            actions = scripted_actions(u)

        out = env.step(actions)
        obs, rew, done, extras = unpack_step(out)

        if recording:
            p = (u._robot.data.root_pos_w[0] - u._terrain.env_origins[0]).cpu().tolist()
            traj.append([p[0], p[1], p[2], int(u._gate_idx[0])])
            if bool(done[0]):
                recording = False

        if bool(done.any()):
            k = int(done.sum())
            logs = extras.get("log", {}) if isinstance(extras, dict) else {}
            n_reset += k
            sum_success += logs.get("Metrics/success_rate", 0.0) * k
            sum_gates += logs.get("Metrics/gates_passed", 0.0) * k
            sum_len += float(u.episode_length_buf[done].float().mean()) * k

    result = {
        "controller": args_cli.controller,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "episodes_finished": n_reset,
        "success_rate": round(sum_success / max(n_reset, 1), 4),
        "mean_gates_passed": round(sum_gates / max(n_reset, 1), 3),
        "mean_episode_length": round(sum_len / max(n_reset, 1), 2),
    }
    print("[EVAL] " + json.dumps(result))
    with open(os.path.join(args_cli.out, f"eval_{args_cli.controller}.json"), "w") as f:
        json.dump(result, f, indent=2)

    gates = (u._gate_pos_w[0] - u._terrain.env_origins[0]).cpu().tolist()
    with open(os.path.join(args_cli.out, f"traj_{args_cli.controller}.json"), "w") as f:
        json.dump(
            {
                "traj": traj,
                "gates": gates,
                "half_width": u.cfg.gate_half_width,
                "half_height": u.cfg.gate_half_height,
            },
            f,
        )

    raw_env.close()
    print("[EVAL] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()