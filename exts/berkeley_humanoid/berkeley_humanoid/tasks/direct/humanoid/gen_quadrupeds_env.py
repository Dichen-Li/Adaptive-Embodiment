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


@configclass
class GenDog0Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG0_CFG

@configclass
class GenDog1Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG1_CFG

@configclass
class GenDog2Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG2_CFG

@configclass
class GenDog3Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG3_CFG

@configclass
class GenDog4Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG4_CFG

@configclass
class GenDog5Cfg(GenDogEnvCfg):
    robot: ArticulationCfg = GEN_DOG5_CFG

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


class GenDirectEnv(LocomotionEnv):
    cfg: GenDogEnvCfg

    def __init__(self, cfg: GenDogEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
