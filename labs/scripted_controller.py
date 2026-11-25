"""手写几何控制器 —— 用作 RL 的对照组 (classical baseline)。

结构就是真实飞控的级联环 (PX4/ArduPilot 同构):

    位置误差 --P--> 期望速度 --P--> 期望加速度
                                    |
                                    +--> 推力 = m * (a_des . b3)
                                    +--> 期望机体 z 轴 z_des_w
                                              |
                                    姿态误差 (机体系) --PD--> 力矩

坐标/符号约定 (极易搞错, 务必按这个推):
    b3          = R @ e3           机体 z 轴在世界系的方向
    z_des_b     = R^T @ z_des_w    期望 z 轴在机体系
    要把 b3 转向 z_des_w, 需要的机体系旋转向量是  delta = e3 x z_des_b
        (小角度: 绕 delta 方向转 |delta|)
    所以力矩 tau = k_R * delta - k_omega * omega_b
    —— 注意是 e3 x z_des_b, 不是 z_des_b x e3, 反了就是正反馈, 飞机会瞬间翻掉。
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import matrix_from_quat


def get_inertia_scale(u) -> float:
    """从仿真里读出机体转动惯量量级, 用于把姿态增益换算成力矩。"""
    try:
        inertias = u._robot.root_physx_view.get_inertias()
        flat = inertias[0, u._body_id[0]]
        return float(max(flat[0], flat[4]))
    except Exception:  # noqa: BLE001
        return 1.4e-5  # Crazyflie 典型值


def scripted_actions(u, kp_pos: float = 1.0, kp_vel: float = 2.5, v_max: float = 1.0) -> torch.Tensor:
    """给定环境, 返回 (num_envs, 4) 的动作。"""
    pos = u._robot.data.root_pos_w
    vel = u._robot.data.root_lin_vel_w
    omega_b = u._robot.data.root_ang_vel_b
    R = matrix_from_quat(u._robot.data.root_quat_w)  # (N,3,3) 机体->世界

    if not hasattr(u, "_scripted_inertia"):
        u._scripted_inertia = get_inertia_scale(u)
    I_ref = u._scripted_inertia

    idx = u._gate_idx.clamp(max=u.cfg.num_gates - 1)
    gate_w = u._gate_pos_w[u._env_ids_all, idx]

    # ---------------- 外环: 位置 -> 期望加速度 ----------------
    err = gate_w - pos
    v_des = torch.clamp(kp_pos * err, -v_max, v_max)
    a_des = kp_vel * (v_des - vel)
    a_des[:, 2] += u._gravity_magnitude  # 抵消重力

    # ---------------- 推力: 期望加速度在机体 z 轴上的投影 ----------------
    b3 = R[:, :, 2]
    thrust = (u._robot_mass * torch.sum(a_des * b3, dim=-1)).clamp(min=0.0)
    a0 = 2.0 * thrust / (u.cfg.thrust_to_weight * u._robot_weight) - 1.0

    # ---------------- 内环: 姿态误差 -> 力矩 ----------------
    z_des_w = a_des / a_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # 注意: 是 R^T @ z_des_w (世界->机体), 所以指标要写 "nji" 而不是 "nij"
    z_des_b = torch.einsum("nji,nj->ni", R, z_des_w)
    e_z = torch.tensor([0.0, 0.0, 1.0], device=z_des_b.device).expand_as(z_des_b)
    delta = torch.cross(e_z, z_des_b, dim=-1).clamp(-0.6, 0.6)  # 机体系下的期望旋转向量, 限幅 ~34 度

    # 带宽要满足 w_n * step_dt << 1 (50 Hz 下 w_n 必须远小于 50 rad/s), 否则离散化直接发散
    w_n = 12.0  # 姿态环自然频率 [rad/s]
    zeta = 1.0  # 阻尼比
    moment = I_ref * (w_n * w_n * delta - 2.0 * zeta * w_n * omega_b)

    a123 = (moment / u.cfg.moment_scale).clamp(-1.0, 1.0)
    return torch.cat([a0.unsqueeze(-1), a123], dim=-1).clamp(-1.0, 1.0)