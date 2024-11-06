# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents
from .humanoid_env import HumanoidEnv, HumanoidEnvCfg
from .h1_direct_env import H1DirectEnv, H1EnvCfg
from .g1_direct_env import G1DirectEnv, G1EnvCfg
from .go2_direct_env import Go2DirectEnv, Go2EnvCfg
from .berkeley_humanoid_direct_env import BerkeleyHumanoidDirectEnv, BerkeleyHumanoidEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Go2-Direct-v0",
    entry_point="berkeley_humanoid.tasks.direct.humanoid:Go2DirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Direct-v0",
    entry_point="berkeley_humanoid.tasks.direct.humanoid:G1DirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="H1-Direct-v0",
    entry_point="berkeley_humanoid.tasks.direct.humanoid:H1DirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Berkeley-Direct-v0",
    entry_point="berkeley_humanoid.tasks.direct.humanoid:BerkeleyHumanoidDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BerkeleyHumanoidEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
