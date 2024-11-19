# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 200
    # experiment_name = "g1_direct_v3"
    experiment_name = "standard"
    # experiment_name = "berkeley_reproduce"
    # experiment_name = "humanoid_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

"""
The name of the following Cfg must be {id}PPORunnerCfg
where id is the gym registry id used for trigger the task
This is for consistency with __init__.py 
"""

@configclass
class GenDog0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog0-v6"
    # experiment_name = "GenDog0"

@configclass
class GenDog1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog1"

@configclass
class GenDog2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog2"

@configclass
class GenDog3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog3"

@configclass
class GenDog4PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog4"

@configclass
class GenDog5PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDog5"

@configclass
class GenHumanoid1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid1"

@configclass
class GenHumanoid2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid2"

@configclass
class GenHumanoid3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid3"

@configclass
class GenHumanoid4PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid4"

@configclass
class GenHumanoid5PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid5"

@configclass
class GenHumanoid6PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenHumanoid6"

@configclass
class GenDogF0R0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF0R0"

@configclass
class GenDogF0R1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF0R1"

@configclass
class GenDogF1R0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF1R0"

@configclass
class GenDogF2R2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF2R2"

@configclass
class GenDogF2R3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF2R3"

@configclass
class GenDogF3R2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogF3R2"

@configclass
class GenDogOriginal0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal0"

@configclass
class GenDogOriginal1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal1"

@configclass
class GenDogOriginal2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal2"

@configclass
class GenDogOriginal3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal3"

@configclass
class GenDogOriginal4PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal4"

@configclass
class GenDogOriginal5PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal5"

@configclass
class GenDogOriginal6PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal6"

@configclass
class GenDogOriginal7PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal7"

@configclass
class GenDogOriginal8PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "GenDogOriginal8"
