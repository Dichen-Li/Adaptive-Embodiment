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
from .gen_dog_direct_env import *
from .gen_humanoid_direct_env import *
from .berkeley_humanoid_direct_env import BerkeleyHumanoidDirectEnv, BerkeleyHumanoidEnvCfg

##
# Register Gym environments.
##

# register standard robot envs
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

"""
Register customized robot configs.
Due to the large number of configs, ideally we want to automate the process
"""

id_entry_pair = {
    "GenDog": GenDogCfg,
    "GenDog1": GenDog1Cfg,
    "GenDog2": GenDog2Cfg,
    "GenDog3": GenDog3Cfg,
    "GenDog4": GenDog4Cfg,
    "GenDog5": GenDog5Cfg,
    "GenHumanoid": GenHumanoidCfg
}

for id, env_cfg_entry_point in id_entry_pair.items():
    rsl_rl_cfg_entry_point = f"{agents.__name__}.rsl_rl_ppo_cfg:{id}PPORunnerCfg"
    gym.register(
        id=id,
        entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point
        },
    )

# gym.register(
#     id="GenDog",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDogCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="GenDog1",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDog1Cfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="GenDog2",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDog2Cfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="GenDog3",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDog3Cfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="GenDog4",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDog4Cfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
#
# gym.register(
#     id="GenDog5",
#     entry_point="berkeley_humanoid.tasks.direct.humanoid:GenDirectEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": GenDog5Cfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
#     },
# )
