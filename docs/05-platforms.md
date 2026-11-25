# 05 · 三个平台的实操路径对照（以及面试怎么讲）

> 本次实操只用了 Isaac Lab。但 JD 上三个平台是一起写的，你需要知道"另外两个怎么上手"，
> 以及同一件事在三个平台里分别怎么写。这一篇是"迁移指南 + 面试弹药"。

---

## 1. 同一架无人机，三个平台的建模差异

| 维度 | Isaac Lab (PhysX/USD) | MuJoCo (MJCF) | Gazebo (SDF + PX4) |
|---|---|---|---|
| 模型格式 | `.usd`（二进制/ASCII，可引用） | `.xml`（MJCF，纯文本） | `.sdf`（纯文本） |
| 四旋翼怎么建 | 单刚体 Articulation（含 4 个转子关节）+ `set_external_force_and_torque` | 单刚体 `freejoint` + `<motor>`/`xfrc_applied` | `x500` 等整机模型 + PX4 固件的 mixer |
| 执行器 | 直接用"推力 + 力矩"（理想执行器）或 `ImplicitActuator` | `motor`（力/力矩映射）或自定义 plugin | **真实飞控固件**（PX4 的混控、PID、状态机全在环） |
| 物理频率 | 100–200 Hz（GPU） | 500–2000 Hz（CPU，软约束更稳） | 250–1000 Hz |
| 并行 | 数千环境（GPU） | 一般 1–64（CPU 多线程） | 通常 1（多实例靠多进程） |
| 渲染 | RTX 光追，最强 | 基础（mjv/mjr） | OGRE，够用 |
| 生态 | PyTorch / RL 库 | Python / MJX（GPU 版） | ROS 2 / MAVLink / PX4 |

一句话选型：

- **要训练策略** → Isaac Lab（吞吐）或 MuJoCo+MJX（单机精细）；
- **要验证飞控栈** → Gazebo + PX4 SITL（因为它跑的是**真固件**）；
- **要做感知/合成数据** → Isaac Sim（渲染）；

---

## 2. MuJoCo 路径：最小可跑的四旋翼（约 80 行）

MuJoCo 对无人机最友好的地方：**MJCF 是纯文本，你能看见每一行物理**。

```xml
<mujoco model="quadrotor">
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="drone" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.15 0.15 0.04" mass="1.0"/>
      <!-- 四个转子用 site 标出位置, 力矩 = r × F 由引擎自动算 -->
      <site name="r1" pos=" 0.18  0.18 0.02"/>
      <site name="r2" pos=" 0.18 -0.18 0.02"/>
      <site name="r3" pos="-0.18 -0.18 0.02"/>
      <site name="r4" pos="-0.18  0.18 0.02"/>
    </body>
  </worldbody>
  <actuator>
    <!-- gear 的第 6 位是转子反扭矩(偏航), 符号按旋转方向 -->
    <motor site="r1" gear="0 0 1 0 0  0.02"/>
    <motor site="r2" gear="0 0 1 0 0 -0.02"/>
    <motor site="r3" gear="0 0 1 0 0  0.02"/>
    <motor site="r4" gear="0 0 1 0 0 -0.02"/>
  </actuator>
</mujoco>
```

要点：

- **`<motor site=...>` + 6 维 `gear`** 直接给出"在指定点施加力 + 反扭矩"，这是四旋翼最干净的建模方式；
- 转子位置离质心 → 引擎自动产生滚转/俯仰力矩，**不需要你手写混控矩阵**；
- `gear` 第 6 位就是桨叶反扭矩（偏航控制来源），符号按顺时针/逆时针配对。

验证顺序（和 Isaac 里完全一样的思路）：

1. 四个电机给相同推力 → 应悬停（推力 = 重力）；
2. 前两个加、后两个减 → 应低头/抬头；
3. 对角两个加 → 应偏航。

`pip install mujoco` 大约 50 MB，本地就能跑，是**没有 GPU 时最好的练兵场**。

---

## 3. Gazebo 路径：PX4 SITL（无人机行业的标准动作）

这条链路的重点不是"物理"，而是**真固件在环**：

```
PX4 固件 (SITL)  ←MAVLink→  Gazebo (gazebo-classic 或 gz-sim)
        ↑
     ROS 2 / MAVROS  ← 你的规划、感知、控制节点
```

典型命令（Ubuntu）：

```bash
# PX4 + Gazebo classic 的 x500 机型
cd PX4-Autopilot && make px4_sitl gazebo-classic

# 或新版 gz-sim
make px4_sitl gz_x500

# QGroundControl 连上后可以手动飞, 也能用 offboard 模式接管
```

你要学的东西：

- **SDF 模型**（`x500` 的 body/propeller/IMU 插件）；
- **PX4 的 EKF2、位置控制器、混控器**（这些在 Isaac 里都被"跳过"了）；
- **`offboard` 模式**：把外部算出的位置/速度/姿态 setpoint 通过 MAVLink 发给飞控。

为什么这条路径对求职重要：**大多数无人机公司的产品栈就是 PX4/ArduPilot + ROS 2**。
Isaac Lab 证明你会训练策略，Gazebo+PX4 证明你懂"策略怎么落到真机上"。

---

## 4. 这次实操里能直接迁移到另两个平台的东西

| 你在 Isaac Lab 学到 | 在 MuJoCo 里对应 | 在 Gazebo/PX4 里对应 |
|---|---|---|
| `sim.dt` + `decimation` | `<option timestep>` | PX4 的 EKF/控制环频率 |
| 动作 = 总推力 + 三轴力矩 | `<motor>` 力映射 / `xfrc_applied` | offboard setpoint（更高层）或 mixer 输入 |
| 投影重力作观测 | 同（自算 `R^T @ (0,0,-1)`） | 飞控内部已有 |
| 域随机化 | 每次 `mj_resetData` 后改 `model.body_mass` | 用真机参数辨识 + 风场插件 |
| 事件检测放哪个钩子 | 每次 `mj_step` 后自算 | ROS 节点里的状态机 |
| 成功率的定义 | 一样 | 一样（但要加通信延迟） |

---

## 5. sim2real 检查清单（面试必问）

答这类问题时，按"信号链"从后往前讲，比罗列名词有效得多：

1. **动作层**：策略输出的是不是真机能接收的量？我们用的是"总推力 + 力矩"，
   真机需要 mixer 把力矩分配到 4 个电机 → **仿真里必须重建 mixer + 电机一阶模型**。
2. **时序**：控制频率、传感器延迟、通信延迟。仿真里加 `delay_steps`，训练时随机化。
3. **感知**：真机 IMU 有噪声、零偏、温漂；仿真里给 obs 加高斯噪声和零偏随机化。
4. **动力学**：质量/惯量/推力系数有 10–30% 偏差；用 DR 覆盖这个区间。
5. **安全**：真机部署要有包络限制、失控保护、软着陆——这些仿真里通常不建模，但会决定你敢不敢飞。

一句话总结（可以直接背）：

> "我在仿真里把策略输出接口做成和真实飞控一致的层级，显式建模了电机延迟和饱和，
> 训练时对质量、推力系数和观测噪声做域随机化，部署前用同一套控制频率和混控做在环比对。"

---

## 6. 面试高频问法 + 回答骨架

| 问题 | 回答骨架 |
|---|---|
| "你用过 Isaac Lab 吗？说说怎么定义一个任务" | `SimulationApp` 启动 → `DirectRLEnv` 或 `ManagerBasedRLEnv` 继承 → 实现 `_setup_scene/_pre_physics_step/_apply_action/_get_*/_reset_idx` → `gym.register` 注册 → 用 rsl_rl 的 `train.py --task` 训练；并行靠 `replicate_physics + clone_in_fabric` |
| "为什么用 Direct 而不是 Manager-based" | 逻辑特殊（事件式奖励、赛道状态机）时 Direct 更直接；多任务复用、团队协作时 Manager-based 更规范 |
| "4096 个环境怎么不爆显存" | 场景只加载一次物理描述，克隆到 Fabric；粒子/网格不重复实例化 |
| "怎么保证能上真机" | 动作层级对齐真机 + 电机模型 + 域随机化 + 同样的控制频率 + 安全包络 |
| "MuJoCo 和 Isaac 的动力学差别" | MuJoCo 软约束、CPU 高频、接触更稳；PhysX 硬约束 + GPU 并行、吞吐高。四旋翼几乎不接触地面，所以差别主要在**执行器建模和时序**，不在接触 |
| "Gazebo 你怎么用" | PX4 SITL 在环验证飞控与通信链路，ROS 2 下跑规划/感知节点，MAVLink 走 offboard |
| "仿真里的成功率能代表真机吗" | 不能。要在随机化后的参数分布上统计，且明确"成功"的判据（窗口宽度、速度上限、扰动强度） |

---

## 7. 建议的下一步学习顺序

1. **把本次的域随机化补上**（2 小时）→ 得到一个"鲁棒性曲线"，这是最像论文/工程的一步；
2. **在 MuJoCo 里重建同一个赛道**（3 小时）→ 你会真正理解"物理引擎差异"；
3. **跑通 PX4 SITL + offboard**（1 天，需要 Ubuntu 或 WSL2）→ 补齐行业工程链路；
4. **加视觉**（1–2 天）→ Isaac Sim 的渲染 + TiledCamera，走向端到端感知。