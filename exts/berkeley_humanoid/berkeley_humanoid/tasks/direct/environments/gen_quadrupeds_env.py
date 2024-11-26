# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg

from berkeley_humanoid.tasks.direct.locomotion.locomotion_env import LocomotionEnv
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from berkeley_humanoid.assets.gen_quadrupeds import *
from omni.isaac.lab.sensors import RayCasterCfg, ContactSensorCfg, patterns


@configclass
class GenDogEnvCfg(DirectRLEnvCfg):
    """
    A parent config class that will be inherited by robot-specific config classes
    """
    seed = 42

    # env
    episode_length_s = 20.0
    decimation = 4
    dt = 0.005
    action_space = 12
    observation_space = 69

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5,
                                                     replicate_physics=True)

    # robot
    robot: ArticulationCfg = MISSING

    # sensor for reward calculation
    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
                                      track_air_time=True, track_pose=True)

    asset_name = "robot"

    # Velocity command ranges
    x_vel_range = (-1.0, 1.0)
    y_vel_range = (-1.0, 1.0)
    yaw_vel_range = (-1.0, 1.0)
    resampling_interval = 10 / (dt * decimation)

    # controller
    controller_use_offset = True
    action_scale = 0.5
    controlled_joints = ".*"

    # reward configurations
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*foot"),
        'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*foot"),
        'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*", ".*thigh.*", ".*trunk.*"]),
        'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint"]),
        'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[".*knee.*joint"]),
        'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*trunk.*", ".*hip.*",
                                                                            ".*thigh.*", ".*calf.*"])
    }

    def __init__(self, robot_cfg, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot_cfg  # Set the specific robot configuration


# @configclass
# class GenDog0Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG0_CFG
#
# @configclass
# class GenDog1Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG1_CFG
#
# @configclass
# class GenDog2Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG2_CFG
#
# @configclass
# class GenDog3Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG3_CFG
#
# @configclass
# class GenDog4Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG4_CFG
#
# @configclass
# class GenDog5Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG5_CFG

@configclass
class GenDogF0R0KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_F0R0_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*foot"),
        'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*foot"),
        'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*"]),
        'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint"]),
        'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*trunk.*", ".*hip.*",
                                                                            ".*thigh.*", ".*calf.*"])
    }

@configclass
class GenDogF0R1KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 10
    robot: ArticulationCfg = GEN_DOG_F0R1_CFG

@configclass
class GenDogF1R0KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 10
    robot: ArticulationCfg = GEN_DOG_F1R0_CFG

@configclass
class GenDogF2R2KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_F2R2_CFG

@configclass
class GenDogF2R3KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_DOG_F2R3_CFG

@configclass
class GenDogF3R2KneeJoint0Cfg(GenDogEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_DOG_F3R2_CFG

@configclass
class GenDogOriginalJoint0Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_0_CFG

@configclass
class GenDogOriginalJoint1Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_1_CFG

@configclass
class GenDogOriginalJoint2Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_2_CFG

@configclass
class GenDogOriginalJoint3Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_3_CFG

@configclass
class GenDogOriginalJoint4Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_4_CFG

@configclass
class GenDogOriginalJoint5Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_5_CFG

@configclass
class GenDogOriginalJoint6Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_6_CFG

@configclass
class GenDogOriginalJoint7Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_7_CFG

@configclass
class GenDogOriginalJoint8Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG_ORIGINAL_8_CFG


# # Add CFG classes dynamically
#
# # Mapping robot file names to canonical class names
# robot_name_map = {
#     "gen_dog_f0r0_knee_joint_0": "GenDogF0R0KneeJoint0Cfg",
#     "gen_dog_f0r1_knee_joint_0": "GenDogF0R1KneeJoint0Cfg",
#     "gen_dog_f1r0_knee_joint_0": "GenDogF1R0KneeJoint0Cfg",
#     "gen_dog_f2r2_knee_joint_0": "GenDogF2R2KneeJoint0Cfg",
#     "gen_dog_f2r3_knee_joint_0": "GenDogF2R3KneeJoint0Cfg",
#     "gen_dog_f3r2_knee_joint_0": "GenDogF3R2KneeJoint0Cfg",
#     "gen_dog_original_joint_0": "GenDogOriginalJoint0Cfg",
#     "gen_dog_original_joint_1": "GenDogOriginalJoint1Cfg",
#     "gen_dog_original_joint_2": "GenDogOriginalJoint2Cfg",
#     "gen_dog_original_joint_3": "GenDogOriginalJoint3Cfg",
#     "gen_dog_original_joint_4": "GenDogOriginalJoint4Cfg",
#     "gen_dog_original_joint_5": "GenDogOriginalJoint5Cfg",
#     "gen_dog_original_joint_6": "GenDogOriginalJoint6Cfg",
#     "gen_dog_original_joint_7": "GenDogOriginalJoint7Cfg",
#     "gen_dog_original_joint_8": "GenDogOriginalJoint8Cfg",
# }
#
# # Dynamically generate and register configuration classes globally
# for robot_file_name, canonical_name in robot_name_map.items():
#     # Dynamically define the class at the global level
#     @configclass
#     class TempClass(GenDogEnvCfg):
#         robot: ArticulationCfg = globals()[robot_file_name.upper() + "_CFG"]
#
#     # Assign the class a proper name
#     TempClass.__name__ = canonical_name
#
#     # Register the class in the global namespace
#     globals()[canonical_name] = TempClass


"""
GenBot-1K quadrupeds
"""

@configclass
class Gendog10Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_10_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog9Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_9_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog8Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_8_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog7Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_7_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog6Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_6_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog5Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_5_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog4Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_4_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog3Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_3_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog2Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_2_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog0Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_0_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog1Cfg(GenDogEnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_1_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog21Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_21_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog20Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_20_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog19Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_19_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog18Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_18_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog17Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_17_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog16Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_16_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog15Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_15_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog14Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_14_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog13Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_13_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog11Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_11_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog12Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_12_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog87Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_87_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog86Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_86_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog85Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_85_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog84Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_84_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog83Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_83_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog82Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_82_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog81Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_81_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog80Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_80_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog79Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_79_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog77Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_77_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog78Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_78_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog76Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_76_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog75Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_75_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog74Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_74_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog73Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_73_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog72Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_72_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog71Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_71_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog70Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_70_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog69Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_69_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog68Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_68_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog66Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_66_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog67Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_67_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog43Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_43_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog42Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_42_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog41Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_41_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog40Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_40_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog39Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_39_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog38Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_38_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog37Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_37_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog36Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_36_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog35Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_35_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog33Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_33_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog34Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_34_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog32Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_32_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog31Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_31_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog30Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_30_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog29Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_29_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog28Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_28_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog27Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_27_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog26Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_26_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog25Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_25_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog24Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_24_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog22Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_22_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog23Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_23_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog109Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_109_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog108Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_108_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog107Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_107_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog106Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_106_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog105Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_105_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog104Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_104_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog103Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_103_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog102Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_102_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog101Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_101_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog99Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_99_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog100Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_100_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog98Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_98_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog97Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_97_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog96Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_96_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog95Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_95_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog94Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_94_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog93Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_93_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog92Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_92_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog91Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_91_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog90Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_90_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog88Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_88_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog89Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_89_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog65Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_65_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog64Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_64_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog63Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_63_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog62Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_62_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog61Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_61_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog60Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_60_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog59Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_59_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog58Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_58_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog57Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_57_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog55Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_55_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog56Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_56_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog54Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_54_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog53Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_53_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog52Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_52_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog51Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_51_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog50Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_50_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog49Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_49_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog48Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_48_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog47Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_47_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog46Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_46_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog44Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_44_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog45Cfg(GenDogEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_45_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog120Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_120_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog119Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_119_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog118Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_118_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog117Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_117_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog116Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_116_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog115Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_115_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog114Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_114_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog113Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_113_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog112Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_112_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog110Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_110_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog111Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_111_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog186Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_186_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog185Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_185_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog184Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_184_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog183Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_183_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog182Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_182_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog181Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_181_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog180Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_180_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog179Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_179_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog178Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_178_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog176Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_176_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog177Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_177_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog175Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_175_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog174Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_174_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog173Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_173_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog172Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_172_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog171Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_171_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog170Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_170_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog169Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_169_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog168Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_168_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog167Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_167_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog165Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_165_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog166Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_166_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog142Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_142_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog141Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_141_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog140Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_140_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog139Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_139_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog138Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_138_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog137Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_137_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog136Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_136_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog135Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_135_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog134Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_134_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog132Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_132_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog133Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_133_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog131Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_131_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog130Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_130_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog129Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_129_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog128Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_128_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog127Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_127_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog126Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_126_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog125Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_125_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog124Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_124_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog123Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_123_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog121Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_121_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog122Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_122_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog208Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_208_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog207Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_207_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog206Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_206_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog205Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_205_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog204Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_204_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog203Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_203_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog202Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_202_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog201Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_201_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog200Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_200_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog198Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_198_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog199Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_199_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog197Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_197_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog196Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_196_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog195Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_195_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog194Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_194_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog193Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_193_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog192Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_192_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog191Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_191_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog190Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_190_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog189Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_189_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog187Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_187_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog188Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_188_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog164Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_164_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog163Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_163_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog162Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_162_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog161Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_161_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog160Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_160_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog159Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_159_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog158Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_158_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog157Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_157_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog156Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_156_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog154Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_154_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog155Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_155_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog153Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_153_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog152Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_152_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog151Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_151_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog150Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_150_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog149Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_149_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog148Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_148_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog147Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_147_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog146Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_146_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog145Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_145_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog143Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_143_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog144Cfg(GenDogEnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_144_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog219Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_219_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog218Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_218_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog217Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_217_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog216Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_216_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog215Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_215_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog214Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_214_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog213Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_213_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog212Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_212_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog211Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_211_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog209Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_209_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog210Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_210_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog285Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_285_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog284Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_284_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog283Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_283_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog282Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_282_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog281Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_281_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog280Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_280_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog279Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_279_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog278Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_278_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog277Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_277_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog275Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_275_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog276Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_276_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog274Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_274_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog273Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_273_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog272Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_272_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog271Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_271_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog270Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_270_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog269Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_269_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog268Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_268_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog267Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_267_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog266Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_266_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog264Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_264_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog265Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_265_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog241Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_241_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog240Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_240_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog239Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_239_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog238Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_238_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog237Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_237_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog236Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_236_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog235Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_235_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog234Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_234_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog233Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_233_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog231Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_231_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog232Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_232_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog230Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_230_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog229Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_229_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog228Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_228_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog227Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_227_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog226Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_226_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog225Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_225_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog224Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_224_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog223Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_223_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog222Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_222_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog220Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_220_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog221Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_221_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog307Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_307_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog306Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_306_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog305Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_305_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog304Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_304_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog303Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_303_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog302Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_302_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog301Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_301_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog300Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_300_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog299Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_299_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog297Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_297_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog298Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_298_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog296Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_296_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog295Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_295_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog294Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_294_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog293Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_293_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog292Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_292_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog291Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_291_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog290Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_290_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog289Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_289_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog288Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_288_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog286Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_286_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog287Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_287_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog263Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_263_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog262Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_262_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog261Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_261_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog260Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_260_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog259Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_259_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog258Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_258_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog257Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_257_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog256Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_256_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog255Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_255_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog253Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_253_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog254Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_254_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog252Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_252_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog251Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_251_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog250Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_250_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog249Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_249_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog248Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_248_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog247Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_247_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog246Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_246_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog245Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_245_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog244Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_244_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog242Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_242_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }

@configclass
class Gendog243Cfg(GenDogEnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_243_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*', '.*thigh.*', '.*trunk.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*trunk.*', '.*hip.*', '.*thigh.*', '.*calf.*'])
    }


class GenDirectEnv(LocomotionEnv):
    cfg: GenDogEnvCfg

    def __init__(self, cfg: GenDogEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
