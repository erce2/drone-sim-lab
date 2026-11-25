"""诊断: 打印 env 0 的终止判据分解, 定位为什么每步都 reset。"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402


def scripted_actions(u) -> torch.Tensor:
    pos = u._robot.data.root_pos_w
    vel = u._robot.data.root_lin_vel_w
    omega_b = u._robot.data.root_ang_vel_b
    R = matrix_from_quat(u._robot.data.root_quat_w)
    idx = u._gate_idx.clamp(max=u.cfg.num_gates - 1)
    gate_w = u._gate_pos_w[u._env_ids_all, idx]
    err = gate_w - pos
    v_des = torch.clamp(2.0 * err, -2.0, 2.0)
    a_des = 4.0 * (v_des - vel)
    a_des[:, 2] += u._gravity_magnitude
    z_des_w = a_des / a_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b3 = R[:, :, 2]
    thrust = (u._robot_mass * torch.sum(a_des * b3, dim=-1)).clamp(min=0.0)
    a0 = 2.0 * thrust / (u.cfg.thrust_to_weight * u._robot_weight) - 1.0
    z_des_b = torch.einsum("nij,nj->ni", R, z_des_w)
    e_z = torch.tensor([0.0, 0.0, 1.0], device=z_des_b.device).expand_as(z_des_b)
    e_tilt = torch.cross(z_des_b, e_z, dim=-1)
    moment = 0.0015 * e_tilt - 0.00008 * omega_b
    a123 = moment / u.cfg.moment_scale
    return torch.cat([a0.unsqueeze(-1), a123], dim=-1).clamp(-1.0, 1.0)


def main():
    task = "Isaac-DroneRacing-Direct-v0"
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = 0
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped

    print("[DBG] max_episode_length:", u.max_episode_length, "step_dt:", u.step_dt)
    print("[DBG] episode_length_s :", u.max_episode_length_s)
    obs, _ = env.reset()
    org = u._terrain.env_origins
    print("[DBG] env_origins      :", org[:2].tolist())
    print("[DBG] gate_pos_w       :", u._gate_pos_w[:2].tolist())
    print("[DBG] gate local       :", (u._gate_pos_w[:2] - org[:2].unsqueeze(1)).tolist())
    print("[DBG] root_pos_w       :", u._robot.data.root_pos_w[:2].tolist())
    print("[DBG] gate_idx         :", u._gate_idx.tolist(), "prev_rel_x:", u._prev_rel_x.tolist())
    print("[DBG] episode_length_buf:", u.episode_length_buf.tolist())
    print("[DBG] thrust/moment    :", u._thrust[:, 0, :].tolist(), u._moment[:, 0, :].tolist())

    for i in range(8):
        act = scripted_actions(u)
        obs, r, term, trunc, extras = env.step(act)
        pos = u._robot.data.root_pos_w
        loc = pos - org
        print(
            f"[DBG] step {i}: act0={act[0,0].item():+.3f} "
            f"z={pos[0,2].item():.3f} loc0={[round(v,3) for v in loc[0].tolist()]} "
            f"term={term.tolist()} trunc={trunc.tolist()} gates={u._gates_passed.tolist()} "
            f"idx={u._gate_idx.tolist()} buf={u.episode_length_buf.tolist()}"
        )
        print(f"[DBG]        thrust={[round(v,4) for v in u._thrust[0,0,:].tolist()]} "
              f"rew={r.tolist()}")

    env.close()
    print("[DBG] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()