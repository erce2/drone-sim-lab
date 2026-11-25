# Copyright (c) 2025, Drone Racing Lab
# SPDX-License-Identifier: BSD-3-Clause

"""门穿越 (drone racing) 任务 —— Direct RL 工作流实现。

任务定义
--------
一架 Crazyflie 从原点出发, 依次穿过 3 个"虚拟门"。门是纯数学平面, 没有碰撞几何体:
    门 i 的中心 (x=gate_x[i], y~U, z~U), 法向 = +x, 有效区域 = 宽 2*half_width, 高 2*half_height。
判定穿门: 机体相对门平面的 x 坐标由负变正, 且此刻在门的矩形窗口内。

为什么用"虚拟门"?
    - 赛道不需要碰撞 -> 物理最轻、最快;
    - 只要改 cfg 里的 gate_x 就能换赛道, 不需要 USD 资产;
    - 真实视觉赛道可以通过 VisualizationMarkers 画出来做视频 (见 play 脚本)。

动作空间 (4 维, 与官方 Quadcopter 任务一致)
    a[0]      : 归一化总推力, thrust = thrust_to_weight * W * (a0+1)/2
    a[1:4]    : 归一化三轴力矩, moment = moment_scale * a[1:4]  (机体坐标系)

观测 (13 维)
    [0:3] 机体线速度 / [3:6] 机体角速度 / [6:9] 投影重力
    [9:12] 当前目标门相对机体的位置 / [12] 阶段 gate_idx/num_gates

奖励设计 (v2: 势能式进度奖励)
------------------------------
第一版用的是"稠密距离奖励" 15*(1-tanh(d/0.8))*dt, 300 轮时成功率到 59%,
但继续训练后策略发现了漏洞: **在门前悬停、不穿门**, 每步都能拿到满额距离奖励,
回报涨到 65 而成功率掉到 0 (reward hacking)。

v2 改成势能式整形 (potential-based shaping):
    r_progress = k * (d_{t-1} - d_t)
悬停时 d 不变 -> 进度奖励为 0, 漏洞被堵住;
数学上它是"势函数差分", 不改变最优策略, 是奖励整形里少数有理论保证的写法。
另外加 alive 惩罚(抑制磨蹭)、crash 惩罚(抑制自杀式提前结束)、事件奖励(穿门/完赛)。
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class DroneRacingEnvCfg(DirectRLEnvCfg):
    """环境配置。Isaac Lab 的所有超参数都走 configclass, 便于 CLI/Hydra 覆盖。"""

    # ------------------------------------------------------------------ 接口尺寸
    episode_length_s = 8.0
    decimation = 2  # 100 Hz 物理 / 2 = 50 Hz 决策
    action_space = 4
    observation_space = 13
    state_space = 0
    debug_vis = False

    # ------------------------------------------------------------------ 物理
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ------------------------------------------------------------------ 场景
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
    )

    # ------------------------------------------------------------------ 机器人
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # ------------------------------------------------------------------ 赛道
    num_gates = 3
    gate_x = [2.0, 4.0, 6.0]
    gate_half_width = 0.45
    gate_half_height = 0.35
    gate_y_range = 1.0
    gate_z_min = 0.5
    gate_z_max = 1.5

    # ------------------------------------------------------------------ 奖励 (v2)
    progress_reward_scale = 3.0  # 每前进 1 m
    gate_pass_bonus = 5.0
    finish_bonus = 10.0
    crash_penalty = -5.0
    alive_penalty = -0.05  # 每个控制步 (按步给, 不乘 dt, 有意为之)
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01

    # ------------------------------------------------------------------ 终止条件
    min_height = 0.1
    max_height = 2.5
    max_lateral = 3.5
    miss_margin = 1.5


class DroneRacingEnv(DirectRLEnv):
    cfg: DroneRacingEnvCfg

    def __init__(self, cfg: DroneRacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- 动作 -> 物理量 ----
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ---- 赛道状态 ----
        self._gate_x = torch.tensor(self.cfg.gate_x, device=self.device)
        self._gate_pos_w = torch.zeros(self.num_envs, self.cfg.num_gates, 3, device=self.device)
        self._gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_rel_x = torch.zeros(self.num_envs, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._progress_reward = torch.zeros(self.num_envs, device=self.device)
        self._event_bonus = torch.zeros(self.num_envs, device=self.device)
        self._crash_penalty = torch.zeros(self.num_envs, device=self.device)
        self._gates_passed = torch.zeros(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._finish_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._env_ids_all = torch.arange(self.num_envs, device=self.device)

        # ---- 日志 ----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["progress", "lin_vel", "ang_vel", "alive", "event", "crash"]
        }

        # ---- 机体参数 ----
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    # ================================================================= 场景
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ================================================================= 动作
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        """把推力/力矩施加到机体。这是"混控之后"的理想执行器模型。

        sim2real 的第一大坑就在这里: 真实电机有一阶惯性、饱和、电压跌落。
        进阶做法: 对 _thrust 加一阶低通 + 随机化时间常数。
        """
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ================================================================= 观测
    def _get_observations(self) -> dict:
        cur_gate_w = self._gate_pos_w[self._env_ids_all, self._gate_idx.clamp(max=self.cfg.num_gates - 1)]
        gate_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, cur_gate_w
        )
        stage = (self._gate_idx.float() / self.cfg.num_gates).unsqueeze(-1)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                gate_pos_b,
                stage,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ================================================================= 穿门判定 + 进度
    def _update_gate_progress(self):
        """检测穿门、推进门序号、计算进度奖励。

        为什么在 _get_dones 里调用?
            DirectRLEnv.step 顺序: _get_dones -> _get_rewards -> _reset_idx。
            事件必须在 _get_rewards 之前算好, 又必须在 _reset_idx 之前算 (否则位置已被重置)。
        """
        pos_w = self._robot.data.root_pos_w
        idx = self._gate_idx.clamp(max=self.cfg.num_gates - 1)
        gate_w = self._gate_pos_w[self._env_ids_all, idx]
        rel = pos_w - gate_w
        dist_now = torch.linalg.norm(rel, dim=-1)

        # 势能式进度奖励: 只有真的靠近才给分, 悬停不产生奖励
        self._progress_reward[:] = self.cfg.progress_reward_scale * (self._prev_dist - dist_now)

        crossed = (self._prev_rel_x < 0.0) & (rel[:, 0] >= 0.0)
        inside = (rel[:, 1].abs() <= self.cfg.gate_half_width) & (rel[:, 2].abs() <= self.cfg.gate_half_height)
        active = self._gate_idx < self.cfg.num_gates
        passed = crossed & inside & active

        self._event_bonus[:] = 0.0
        self._event_bonus[passed] = self.cfg.gate_pass_bonus
        if passed.any():
            self._gate_idx[passed] += 1
            self._gates_passed[passed] += 1

        finished = (self._gate_idx >= self.cfg.num_gates) & (~self._finish_awarded)
        self._event_bonus[finished] += self.cfg.finish_bonus
        self._finish_awarded |= finished
        self._episode_success |= self._gate_idx >= self.cfg.num_gates

        # 换了门要用新门重算 prev_rel_x / prev_dist, 否则下一步会误判穿越 / 进度爆表
        new_idx = self._gate_idx.clamp(max=self.cfg.num_gates - 1)
        new_gate_w = self._gate_pos_w[self._env_ids_all, new_idx]
        new_dist = torch.linalg.norm(pos_w - new_gate_w, dim=-1)
        new_rel_x = (pos_w - new_gate_w)[:, 0]
        self._prev_rel_x[:] = torch.where(passed, new_rel_x, rel[:, 0])
        self._prev_dist[:] = torch.where(passed, new_dist, dist_now)

    # ================================================================= 奖励
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        alive = torch.full_like(lin_vel, self.cfg.alive_penalty)

        rewards = {
            "progress": self._progress_reward,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "alive": alive,
            "event": self._event_bonus,
            "crash": self._crash_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # ================================================================= 终止
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_gate_progress()

        pos = self._robot.data.root_pos_w
        # 越界判据必须用"环境局部坐标": 每个环境有自己的原点 (env_origins),
        # 直接用世界坐标会把"别的环境的地盘"误判成越界 —— 本次踩过的第一个坑。
        pos_local = pos - self._terrain.env_origins
        crashed = (
            (pos[:, 2] < self.cfg.min_height)
            | (pos[:, 2] > self.cfg.max_height)
            | (pos_local[:, 1].abs() > self.cfg.max_lateral)
        )

        idx = self._gate_idx.clamp(max=self.cfg.num_gates - 1)
        gate_w = self._gate_pos_w[self._env_ids_all, idx]
        missed = ((pos[:, 0] - gate_w[:, 0]) > self.cfg.miss_margin) & (self._gate_idx < self.cfg.num_gates)

        finished = self._gate_idx >= self.cfg.num_gates
        terminated = crashed | missed | finished
        self._crash_penalty[:] = self.cfg.crash_penalty * crashed.float()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ================================================================= 复位
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ---- 先把本回合指标写进 extras (必须在清零前) ----
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/gates_passed"] = torch.mean(self._gates_passed[env_ids]).item()
        extras["Metrics/success_rate"] = torch.mean(self._episode_success[env_ids].float()).item()
        idx = self._gate_idx[env_ids].clamp(max=self.cfg.num_gates - 1)
        gate_w = self._gate_pos_w[env_ids, idx]
        extras["Metrics/final_distance_to_goal"] = torch.mean(
            torch.linalg.norm(gate_w - self._robot.data.root_pos_w[env_ids], dim=-1)
        ).item()
        self.extras["log"].update(extras)

        # ---- 复位机器人 ----
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # ---- 采样新赛道 ----
        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]
        gx = self._gate_x.unsqueeze(0).expand(n, self.cfg.num_gates)
        gy = torch.empty((n, self.cfg.num_gates), device=self.device).uniform_(
            -self.cfg.gate_y_range, self.cfg.gate_y_range
        )
        gz = torch.empty((n, self.cfg.num_gates), device=self.device).uniform_(
            self.cfg.gate_z_min, self.cfg.gate_z_max
        )
        local = torch.stack([gx, gy, gz], dim=-1)
        self._gate_pos_w[env_ids] = local + origins.unsqueeze(1)

        self._gate_idx[env_ids] = 0
        self._gates_passed[env_ids] = 0.0
        self._episode_success[env_ids] = False
        self._finish_awarded[env_ids] = False
        self._prev_rel_x[env_ids] = -self.cfg.gate_x[0]
        self._prev_dist[env_ids] = self.cfg.gate_x[0]
        self._progress_reward[env_ids] = 0.0
        self._event_bonus[env_ids] = 0.0
        self._crash_penalty[env_ids] = 0.0

        # ---- 复位机体位姿 ----
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += origins
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ================================================================= 可视化 (可选)
    def _set_debug_vis_impl(self, debug_vis: bool):
        from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers

        if debug_vis:
            if not hasattr(self, "gate_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.02, 2 * self.cfg.gate_half_width, 2 * self.cfg.gate_half_height)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.3)
                marker_cfg.prim_path = "/Visuals/Command/gate"
                self.gate_visualizer = VisualizationMarkers(marker_cfg)
            self.gate_visualizer.set_visibility(True)
        else:
            if hasattr(self, "gate_visualizer"):
                self.gate_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # 画出所有门 (通常是 1~4 个环境, 4096 并行训练时 debug_vis=False 不会走到这里)
        self.gate_visualizer.visualize(self._gate_pos_w.reshape(-1, 3))