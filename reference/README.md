# reference/ —— 官方源码副本（仅供对照阅读）

这里的文件是 **Isaac Lab 2.7.0 / Isaac Sim 5.1** 中与本次实操直接相关的原始文件，
为方便离线对照阅读而原样拷贝（未做任何修改），**不参与本项目构建**。

| 文件 | 上游路径 | 为什么读它 |
|---|---|---|
| `quadcopter/` | `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/` | 本次任务的起点；[docs/02](../docs/02-isaaclab-anatomy.md) 逐段解剖它 |
| `direct_rl_env.py` | `source/isaaclab/isaaclab/envs/direct_rl_env.py` | `step()` 的调用顺序决定事件检测该放哪个钩子 |
| `articulation.py` | `source/isaaclab/isaaclab/assets/articulation/articulation.py` | `set_external_force_and_torque` 的坐标系约定（默认**机体坐标系**） |
| `train.py` | `scripts/reinforcement_learning/rsl_rl/train.py` | `--task/--num_envs/--max_iterations/--resume` 的真实行为 |
| `cli_args.py` | `scripts/reinforcement_learning/rsl_rl/cli_args.py` | rsl_rl 相关 CLI 参数 |

## 许可

上述文件版权归 **The Isaac Lab Project Developers**（NVIDIA），以 **BSD-3-Clause** 许可发布，
各文件头部保留了原始版权声明与 SPDX 标识。使用、修改、再分发请遵循 BSD-3-Clause。

本项目自身代码（`labs/`、`remote/`、`scripts/`）采用 [MIT License](../LICENSE)。