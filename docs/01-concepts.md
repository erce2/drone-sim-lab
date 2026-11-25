# 01 · 机器人仿真平台：你真正需要理解的东西

> 目标读者：做无人机、但没系统用过 Isaac / MuJoCo / Gazebo 的人。
> 读完这一篇，你应该能在面试里把"仿真平台"从名词变成一套方法论。

---

## 1. 先破除一个误区：三个平台不是竞品

招聘 JD 里常把三者并列写成"robot simulation platforms"，但它们解决的是**不同层次**的问题。

| 平台 | 一句话定位 | 你什么时候用它 |
|---|---|---|
| **Gazebo** (+ PX4 SITL) | ROS 生态里的"系统级"仿真器 | 验证**整条栈**：飞控固件、MAVLink/ROS 2 通信、感知、规划。传感器模型和插件生态最贴近真实飞控开发流程 |
| **MuJoCo** | 单环境、高精度接触动力学 | 控制算法研究、接触丰富的任务、快速迭代。CPU 上单环境比 GPU 仿真器更快（小场景） |
| **Isaac Sim** | NVIDIA 的 RTX + PhysX + USD 仿真器 | 高保真渲染、合成数据、传感器仿真、USD 资产管线 |
| **Isaac Lab** | 建在 Isaac Sim 之上的 **RL 训练框架** | 大规模并行（数千环境）训练策略，内置 rsl_rl/skrl/rl_games，是现在具身智能论文的默认平台之一 |

关键区分：

- **Isaac Sim 是仿真器，Isaac Lab 是训练框架**。面试被问"Isaac Lab 里怎么定义一个任务"，答不上 `DirectRLEnv` / `ManagerBasedRLEnv`、`gym.register`、`rsl_rl` 这套流程，就等于没有用过。
- **Gazebo 和 Isaac 的差别不是"精度"，是"生态"**：Gazebo 连着 ROS 2 和 PX4，Isaac 连着 USD 和 PyTorch。
- **MuJoCo 不是"弱化版 Isaac"**：它是 CPU 上最快的单环境仿真器之一，DeepMind 的 RL 论文大量基于它；很多无人机控制论文（尤其涉及接触/抓取/气动）仍在用。

> 面试话术（无人机岗）：
> "我按验证目标分层选型——飞控和通信链路用 Gazebo+PX4 SITL 做 HIL 前的闭环验证；控制算法原型用 MuJoCo 单环境快速迭代；需要大规模并行训练或视觉合成数据时上 Isaac Lab。"

---

## 2. 仿真器到底在算什么

一个刚体仿真器每步做四件事：

```
读取状态 → 计算力(重力/气动/接触/执行器) → 积分(求解约束) → 写回状态
```

你需要关心的四个旋钮：

### 2.1 物理步长 vs 控制步长（decimation）

Isaac Lab 里这两个数分开配：

```python
sim.dt = 1/100      # 物理步长 100 Hz
decimation = 2      # 每 2 个物理步，策略做一次决策
# => 控制频率 50 Hz（step_dt = 0.02s）
```

**为什么不能一步到位？** 物理步长受稳定性限制（小质量刚体 + 强执行器 → 必须小 dt）；而控制/策略的决策频率受**感知延迟和动力学带宽**限制（真实飞控 50–500 Hz）。两者解耦后，你可以在不动策略的前提下改物理精度。

本次实操里 Crazyflie 只有 28 g，`dt=1/100` 是够用的下限——再粗就会出现数值不稳定。

### 2.2 求解器与接触

- PhysX（Isaac）用 TGS 求解器 + 子迭代，接触靠"位置/速度迭代次数"控制精度；
- MuJoCo 用凸优化 + 软约束，接触更平滑、更适合做接触丰富任务；
- **无人机大部分时间不接触地面**，所以求解器差异对你影响很小 —— 这也是为什么无人机 RL 在 Isaac 里能跑得极快。

### 2.3 执行器模型（sim2real 第一大坑）

Isaac Lab 的官方 Quadcopter 任务把动作直接映射成"总推力 + 三轴力矩"：

```python
thrust = thrust_to_weight * W * (a0 + 1) / 2     # 归一化动作 -> 牛顿
moment = moment_scale * a[1:4]                    # 归一化动作 -> N·m
```

这是一个**理想执行器**：指令瞬时生效、推力线性、无延迟。

真实的四旋翼是：油门 → 电机转速（一阶惯性，τ≈20–50 ms）、转速² → 推力、电池电压跌落、桨叶效率随来流变化。**这中间的差距，就是 sim2real 要做的事**（见第 4 节）。

### 2.4 观测的坐标系

官方环境用 `projected_gravity_b`（重力在机体系的投影）而不是欧拉角/四元数：

- 无万向锁，全局连续；
- 与真实 IMU 的加速度计读数同构（悬停时 ≈ (0,0,-1)）；
- 网络学起来最容易。

这是无人机 RL 的标准做法，值得记住：**姿态用"投影重力"，不用欧拉角**。

---

## 3. 动作空间怎么设计（这是最像"工程"的部分）

四旋翼 RL 的动作空间有三档，越往下越"底层"：

| 层级 | 动作 | 优点 | 缺点 |
|---|---|---|---|
| 高层 | 期望位置/速度 | 学得快，行为稳 | 学不到敏捷机动，且需要底层控制器 |
| 中层 | 总推力 + 三轴力矩（**本次采用**） | 兼顾可学性与机动性，是无人机 RL 主流 | 仍需自己做混控到电机 |
| 底层 | 4 个电机转速 | 可直接 sim2real | 学习难度大，探索空间大 |

有意思的一点：**中层动作其实是"混控之后"的量**。真实飞控里你要写 mixer，把 `(F, τx, τy, τz)` 分配成 4 个电机推力。若要做 sim2real，通常还要在仿真里重建 mixer + 电机模型，让策略输出直接是电机指令——这样策略才可能迁到真机。

---

## 4. 域随机化：为什么"仿真里成功"不等于"真机能飞"

三件事让仿真和现实分家：

1. **动力学参数偏差**：质量、惯量、推力系数、气动阻尼
2. **执行器偏差**：电机响应延迟、饱和度、死区
3. **感知偏差**：传感器噪声、量化、延迟、标定误差

对应三种手段：

| 手段 | 做法 | 在我们环境里的落点 |
|---|---|---|
| 域随机化 (DR) | 每个 episode 随机采样参数 | 在 `_reset_idx` 里随机化 `thrust_to_weight`、质量缩放、动作延迟 |
| 观测噪声 | 对 obs 加高斯/延迟 | 用 `DirectRLEnvCfg.observation_noise_model` |
| 系统辨识 | 用真机数据拟合参数 | 训练后对比真机响应（离线） |

**面试高频问题**："你训练的策略怎么保证能上真机？"
→ 答：动作空间选能与真实飞控对接的层级 + 执行器模型显式建模延迟/饱和 + 训练时域随机化 + 部署时用相同的控制频率和混控，而不是"仿真里 reward 高就上机"。

---

## 5. Isaac Lab 的架构（背下来）

```
SimulationApp (Kit)          ← 启动 Isaac Sim 运行时, 必须在 import 之前
  └─ SimulationCfg            ← 物理步长、GPU、重力、材质
  └─ InteractiveScene         ← USD 场景: 机器人 + 地形 + 传感器
       └─ clone_environments(copy_from_source=False)
       └─ replicate_physics=True / clone_in_fabric=True   ← 4096 并行的关键
  └─ DirectRLEnv / ManagerBasedRLEnv
       ├─ _setup_scene()       ← 搭场景
       ├─ _pre_physics_step()  ← 动作预处理（每个控制步一次）
       ├─ _apply_action()      ← 每个物理步施加动作
       ├─ _get_observations()  ← 观测
       ├─ _get_rewards()       ← 奖励
       ├─ _get_dones()         ← 终止
       └─ _reset_idx()         ← 复位（含阈值采样、随机化）
  └─ RslRlVecEnvWrapper  →  OnPolicyRunner (rsl_rl PPO)
```

两个必须分清的写法：

- **Direct 工作流**：所有 reward/obs/termination 逻辑都写在一个 `Env` 类里，适合单文件、逻辑特殊的任务（无人机、赛车、灵巧手）。本次实操用这个。
- **Manager-based 工作流**：用 `ObservationTermCfg / RewardTermCfg / TerminationTermCfg` 这种"配置声明式"写法，适合多人协作、大量复用 locomotion/manipulation 的场景。

`step()` 里的调用顺序（**这个顺序有坑，务必记住**）：

```
_pre_physics_step → [decimation 次: _apply_action + 物理步进] 
→ _get_dones → _get_rewards → _reset_idx → _get_observations
```

自己记牢的理由：**穿门/到点这类"事件检测"必须在 `_get_rewards` 之前算好、又必须在 `_reset_idx` 之前算**，所以本次把检测放在 `_get_dones` 里（详见 [03](03-task-design.md)）。

---

## 6. 术语速查（面试前扫一眼）

| 术语 | 含义 | 易错点 |
|---|---|---|
| Articulation | 带关节的刚体树（机器人） | 四旋翼是"单刚体 + free joint"，仍被包装成 Articulation |
| Floating base | 无固定基座的机构 | 根状态要自己管复位 |
| USD | Universal Scene Description，NVIDIA 的资产格式 | Isaac 资产几乎都是 `.usd`，加载失败是最常见的报错 |
| Fabric | PhysX 的 GPU 数据副本，绕过 USD/场景图 | `clone_in_fabric` 让 4096 环境几乎不花内存 |
| decimation | 每个控制步内的物理步数 | 忘了它就会把 dt 配错 |
| episode_length_buf | 每个环境的已走步数 | 集体清零会产生训练曲线尖峰，要错开 |
| rsl_rl | ETH 的 RL 库（PPO） | Isaac Lab 里通过 `rsl_rl_cfg_entry_point` 挂接 |