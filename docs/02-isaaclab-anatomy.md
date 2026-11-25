# 02 · 逐段解剖官方的 Quadcopter 环境

> 对象：[isaaclab_tasks/direct/quadcopter/quadcopter_env.py](../reference/quadcopter/quadcopter_env.py)（253 行）。
> 这是 Isaac Lab 2.x 自带的四旋翼任务，也是本次实操的起点。
> **读懂它 = 掌握 Direct 工作流的全部套路**，之后所有无人机任务都是在这个骨架上改。

---

## 0. 先看这 7 个问题（读任何 Isaac Lab 环境的通用顺序）

拿到一个陌生环境，不要从第 1 行读。按这个顺序问自己 7 个问题：

1. **时间尺度**：`sim.dt` × `decimation` = 控制周期是多少？`episode_length_s` 多长？
2. **动作是什么物理量**：`action_space` 几维？`_pre_physics_step` 里怎么映射成力和力矩？
3. **观测是什么**：`observation_space` 几维？每个分量在哪个坐标系？
4. **奖励怎么设计**：`_get_rewards` 里每一项的物理动机是什么？
5. **什么时候结束**：`_get_dones` 的终止条件是什么？
6. **怎么复位**：`_reset_idx` 采样了什么？为什么要错开 episode 长度？
7. **怎么并行**：`scene` 的克隆/复现配置。

下面按这 7 个问题拆。

---

## 1. 配置："超参数都在 cfg 里"

```python
episode_length_s = 10.0     # 一回合 10 秒
decimation = 2              # 每 2 个物理步决策 1 次
action_space = 4            # 动作维度
observation_space = 12      # 观测维度
sim.dt = 1/100              # 物理 100 Hz
# => 控制频率 50 Hz，一回合 500 个决策步
```

`state_space = 0` 的意思是"没有额外的 critic 特权观测"（asymmetric actor-critic 才用）。

**场景配置**（`InteractiveSceneCfg`）是 Isaac Lab 能跑 4096 并行的关键：

```python
num_envs=4096, env_spacing=2.5,
replicate_physics=True,     # 所有环境共用一份物理描述
clone_in_fabric=True,       # 数据克隆进 PhysX 的 Fabric，不经过 USD 场景图
```

`env_spacing` 决定环境网格的间距；每个环境有自己的**原点** `env_origins`，所有"世界坐标"都在各自原点的偏移下——**忘了这一点就会写出所有环境共享坐标的 bug**（本次实操就踩了这个坑，见 [03](03-task-design.md)）。

**机器人配置**：

```python
robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
thrust_to_weight = 1.9      # 最大推力 ≈ 1.9 倍重力
moment_scale = 0.01         # 每单位动作对应 0.01 N·m
```

`CRAZYFLIE_CFG` 来自 `isaaclab_assets`，指向 USD 资产：

```
{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd
```

> **坑点**：`ISAAC_NUCLEUS_DIR` 默认指向 AWS S3。国内网络不可达时，`Could not open asset` 就是这个原因。
> 解决办法是把资产离线镜像到本地，再把 `persistent.isaac.asset_root.default` 指向本地目录（本次的 `C:\isaac_assets`）。

---

## 2. `__init__`：把该缓存的都缓存好

```python
self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
self._body_id = self._robot.find_bodies("body")[0]
self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
```

三个要点：

- **所有张量都在 GPU 上预分配**。RL 训练每步要执行几万次，任何 `torch.zeros(...)` 出现在 step 里都是性能事故。
- `find_bodies("body")` 用**正则**查刚体索引，因为施加外力需要 body id。
- 质量是从 USD 里读出来的（不是硬编码的），所以换机型不用改代码。本次实测 Crazyflie = **28.2 g**，和真实 cf2x（约 27 g）一致——这就是"仿真模型对不对"的第一个检查点。

---

## 3. `_setup_scene`：五步搭场景

```python
self._robot = Articulation(self.cfg.robot)        # 1. 创建机器人
self.scene.articulations["robot"] = self._robot   # 2. 注册进 scene

self._terrain = self.cfg.terrain.class_type(...)  # 3. 地形
self.scene.clone_environments(copy_from_source=False)  # 4. 克隆成 N 份
if self.device == "cpu":                          # 5. CPU 下要手动过滤碰撞
    self.scene.filter_collisions(...)
```

注意 `filter_collisions` 只在 **CPU** 后端需要——说明这套代码同时支持 CPU/GPU 物理，GPU 下 fabric 会自动处理。

---

## 4. 动作：从 4 个归一化数到 6 维力旋量

这是整个环境最"物理"的 6 行：

```python
def _pre_physics_step(self, actions):
    self._actions = actions.clone().clamp(-1.0, 1.0)
    # a0 ∈ [-1,1] -> 推力 ∈ [0, 1.9*W]
    self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
    # a1..a3 ∈ [-1,1] -> 力矩 ∈ [-0.01, 0.01] N·m
    self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

def _apply_action(self):
    self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
```

**为什么推力要这么映射？** 因为要让动作 0 对应"略低于悬停"：

| 动作 a0 | 推力 | 效果 |
|---|---|---|
| -1 | 0 N | 自由落体 |
| **2/1.9 − 1 ≈ 0.0526** | **= W = 0.277 N** | **精确悬停** |
| +1 | 1.9W = 0.526 N | 最大爬升 |

这个 `0.0526` 就是本次 Lab 1 打印出来的 `hover throttle`。**这是验证"你的执行器模型有没有配对"的最快方法**：给定重量和推重比，能算出悬停油门；如果仿真里给这个油门飞机悬停不住，说明力施加错了（坐标系、方向、body id 等）。

另有两点要注意：

- `set_external_force_and_torque` 的力/力矩在**世界系**给出（这里 z 向上，所以 `[:, 0, 2]` 是竖直推力）；力矩在**机体系**。混用会导致莫名的翻滚。
- 每个物理步都会重新 `set`（不是累加），所以不会出现"力越积越大"的问题。

---

## 5. 观测：12 维，全部机体坐标系

```python
obs = cat([
    root_lin_vel_b,        # 3  机体系线速度
    root_ang_vel_b,        # 3  机体系角速度
    projected_gravity_b,   # 3  重力在机体系的投影（≈ 姿态）
    desired_pos_b,         # 3  目标点相对机体的位置
])                         # 共 12 维
```

三个设计决策值得记住：

1. **全部用机体坐标系**（`_b` 后缀）。这样策略对"世界朝向"不敏感，换个初始朝向不用重训。
2. **用投影重力代替欧拉角/四元数**：无万向锁、连续、和 IMU 加速度计同构。四元数也行，但网络对 `(w,x,y,z)` 双覆盖不友好。
3. **目标用相对位置**，不是绝对坐标。绝对坐标会让策略学"地图"而不是"导航"。

`subtract_frame_transforms(root_pos_w, root_quat_w, target_w)` 就是做"世界 → 机体"的坐标变换，是 Isaac Lab 里最常用的工具函数之一。

---

## 6. 奖励：三项，都是"整形"而非"稀疏"

```python
lin_vel  = ||v_b||²        × (-0.05)   # 惩罚速度（鼓励悬停而不是乱飞）
ang_vel  = ||ω_b||²        × (-0.01)   # 惩罚角速度（鼓励平稳）
dist     = ||p_goal - p||  
dist_mapped = 1 - tanh(dist / 0.8)     # 距离 → (0,1)
dist_mapped × 15.0                     # 越近奖励越高

总奖励 = Σ(各项 × step_dt)
```

关键设计：

- **`tanh` 距离整形**：远距离时梯度平缓（不会被"距离"一项完全支配），近距离时敏感（能精细收敛）。这是无人机/移动机器人 RL 里最常见的距离奖励。
- **每项都乘 `step_dt`**：让奖励积分与物理时间一致，改控制频率时不用重新调 reward scale。这是个容易被忽略的工程细节。
- **代价**：`-0.05||v||²` 和 `+tanh(d)` 在目标点附近会打架（想停在目标点就得不移动）。这类权衡是 reward 设计的主要工作量。

---

## 7. 终止与复位

```python
died     = (z < 0.1) or (z > 2.0)     # 撞地 或 飞太高
time_out = episode_length_buf >= max_episode_length - 1
```

复位时做的三件事：

```python
# 1) 随机采样目标点（这才是任务随机的来源）
self._desired_pos_w[env_ids, :2] = uniform(-2, 2) + env_origins[env_ids, :2]
self._desired_pos_w[env_ids, 2]  = uniform(0.5, 1.5)

# 2) 复位机器人状态
default_root_state[:, :3] += env_origins[env_ids]     # 世界坐标 = 局部 + 原点
self._robot.write_root_pose_to_sim(...)
self._robot.write_root_velocity_to_sim(...)

# 3) 错开 episode 长度
if len(env_ids) == self.num_envs:
    self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=max_episode_length)
```

第 3 条是**训练稳定性的关键技巧**：如果 4096 个环境同时到达 episode 末尾，就会同时复位、同时产生大量低价值样本，训练曲线上出现周期性尖峰。随机错开后，每步只有少量环境复位，采样分布平稳。

---

## 8. 读完之后你应该能回答

- 如果我把 `decimation` 从 2 改成 4，会发生什么？（控制频率降到 25 Hz，同样的 `max_episode_length` 对应的时间翻倍，reward 积分也变）
- 如果我想让飞机学会穿越障碍，最小改动是什么？（加障碍物 + 改目标采样 + 改终止条件）
- 如果 sim2real 时真机比仿真"迟钝"，应该在仿真里加什么？（动作延迟 / 一阶惯性 / DR）
- 4096 个环境为什么不会把显存吃光？（`replicate_physics` + `clone_in_fabric`）

---

## 附：本次实操的自查清单

| 检查项 | 期望 | 本次实测 |
|---|---|---|
| 机器质量 | 与真实 cf2x 一致（~27 g） | 28.2 g ✓ |
| 悬停油门 | `2/thrust_to_weight − 1` | 0.0526 ✓ |
| 观测维度 | 与 cfg 一致 | (N, 13)（我加了阶段位）✓ |
| 环境原点 | 每个 env 独立 | 需用 `env_origins` 修正，踩过坑 ✗→✓ |
| 资产加载 | 无 `Could not open asset` | 离线镜像 `cf2x.usd` ✓ |