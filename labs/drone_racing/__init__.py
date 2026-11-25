# Copyright (c) 2025, Drone Racing Lab
# SPDX-License-Identifier: BSD-3-Clause

"""注册门穿越任务, 让 `--task Isaac-DroneRacing-Direct-v0` 可用。"""

import gymnasium as gym

from . import agents

##
# 注册 Gym 环境
##

gym.register(
    id="Isaac-DroneRacing-Direct-v0",
    entry_point=f"{__name__}.drone_racing_env:DroneRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racing_env:DroneRacingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DroneRacingPPORunnerCfg",
    },
)