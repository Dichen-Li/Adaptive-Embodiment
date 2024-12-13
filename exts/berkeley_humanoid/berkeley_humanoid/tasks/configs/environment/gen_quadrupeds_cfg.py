from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from berkeley_humanoid.assets.gen_quadrupeds import *
from .go2_cfg import Go2EnvCfg


# @configclass
# class GenDogEnvCfg(DirectRLEnvCfg):
#     """
#     A parent config class that will be inherited by robot-specific config classes
#     """
#     seed = 42
#
#     # env
#     episode_length_s = 20.0
#     decimation = 4
#     dt = 0.005
#     action_space = 12
#     observation_space = 69
#
#     # simulation
#     sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)
#     terrain = TerrainImporterCfg(
#         prim_path="/World/ground",
#         terrain_type="plane",
#         collision_group=-1,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="average",
#             restitution_combine_mode="average",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#             restitution=0.0,
#         ),
#         debug_vis=False,
#     )
#
#     # scene
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5,
#                                                      replicate_physics=True)
#
#     # robot
#     robot: ArticulationCfg = MISSING
#
#     # sensor for reward calculation
#     contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
#                                       track_air_time=True, track_pose=True)
#
#     asset_name = "robot"
#
#     # Velocity command ranges
#     x_vel_range = (-1.0, 1.0)
#     y_vel_range = (-1.0, 1.0)
#     yaw_vel_range = (-1.0, 1.0)
#     resampling_interval = 10 / (dt * decimation)
#
#     # controller
#     controller_use_offset = True
#     action_scale = 0.5
#     controlled_joints = ".*"
#
#     # reward configurations
#     reward_cfgs = {
#         'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*foot"),
#         'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*foot"),
#         'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*", ".*thigh.*", ".*trunk.*"]),
#         'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint"]),
#         'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[".*knee.*joint"]),
#         'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*trunk.*", ".*hip.*",
#                                                                             ".*thigh.*", ".*calf.*"])
#     }
#
#     def __init__(self, robot_cfg, **kwargs):
#         super().__init__(**kwargs)
#         self.robot = robot_cfg  # Set the specific robot configuration


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


# @configclass
# class GenDogF0R0KneeJoint0Cfg(Go2EnvCfg):
#     action_space = 8
#     robot: ArticulationCfg = GEN_DOG_F0R0_CFG
#     reward_cfgs = {
#         'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*foot"),
#         'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*foot"),
#         'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*"]),
#         'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint"]),
#         'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[]),
#         'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*trunk.*", ".*hip.*",
#                                                                             ".*thigh.*", ".*calf.*"])
#     }

# @configclass
# class GenDogF0R1KneeJoint0Cfg(GenDogEnvCfg):
#     action_space = 10
#     robot: ArticulationCfg = GEN_DOG_F0R1_CFG
#
# @configclass
# class GenDogF1R0KneeJoint0Cfg(GenDogEnvCfg):
#     action_space = 10
#     robot: ArticulationCfg = GEN_DOG_F1R0_CFG
#
# @configclass
# class GenDogF2R2KneeJoint0Cfg(GenDogEnvCfg):
#     action_space = 16
#     robot: ArticulationCfg = GEN_DOG_F2R2_CFG
#
# @configclass
# class GenDogF2R3KneeJoint0Cfg(GenDogEnvCfg):
#     action_space = 18
#     robot: ArticulationCfg = GEN_DOG_F2R3_CFG
#
# @configclass
# class GenDogF3R2KneeJoint0Cfg(GenDogEnvCfg):
#     action_space = 18
#     robot: ArticulationCfg = GEN_DOG_F3R2_CFG
#
# @configclass
# class GenDogOriginalJoint0Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_0_CFG
#
# @configclass
# class GenDogOriginalJoint1Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_1_CFG
#
# @configclass
# class GenDogOriginalJoint2Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_2_CFG
#
# @configclass
# class GenDogOriginalJoint3Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_3_CFG
#
# @configclass
# class GenDogOriginalJoint4Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_4_CFG
#
# @configclass
# class GenDogOriginalJoint5Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_5_CFG
#
# @configclass
# class GenDogOriginalJoint6Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_6_CFG
#
# @configclass
# class GenDogOriginalJoint7Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_7_CFG
#
# @configclass
# class GenDogOriginalJoint8Cfg(GenDogEnvCfg):
#     robot: ArticulationCfg = GEN_DOG_ORIGINAL_8_CFG


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
class Gendog10Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_10_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog9Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_9_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog8Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_8_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog7Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_7_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog6Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_6_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog5Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_5_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog4Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_4_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog3Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_3_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog2Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_2_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog0Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_0_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog1Cfg(Go2EnvCfg):
    action_space = 8
    robot: ArticulationCfg = GEN_DOG_1_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog21Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_21_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog20Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_20_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog19Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_19_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog18Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_18_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog17Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_17_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog16Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_16_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog15Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_15_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog14Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_14_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog13Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_13_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog11Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_11_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog12Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_12_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog87Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_87_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog86Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_86_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog85Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_85_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog84Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_84_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog83Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_83_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog82Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_82_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog81Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_81_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog80Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_80_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog79Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_79_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog77Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_77_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog78Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_78_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog76Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_76_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog75Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_75_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog74Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_74_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog73Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_73_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog72Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_72_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog71Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_71_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog70Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_70_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog69Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_69_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog68Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_68_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog66Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_66_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog67Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_67_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog43Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_43_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog42Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_42_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog41Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_41_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog40Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_40_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog39Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_39_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog38Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_38_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog37Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_37_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog36Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_36_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog35Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_35_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog33Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_33_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog34Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_34_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog32Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_32_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog31Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_31_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog30Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_30_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog29Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_29_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog28Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_28_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog27Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_27_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog26Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_26_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog25Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_25_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog24Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_24_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog22Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_22_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog23Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_23_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog109Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_109_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog108Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_108_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog107Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_107_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog106Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_106_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog105Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_105_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog104Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_104_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog103Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_103_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog102Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_102_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog101Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_101_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog99Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_99_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog100Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_100_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog98Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_98_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog97Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_97_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog96Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_96_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog95Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_95_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog94Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_94_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog93Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_93_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog92Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_92_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog91Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_91_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog90Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_90_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog88Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_88_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog89Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_89_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog65Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_65_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog64Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_64_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog63Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_63_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog62Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_62_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog61Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_61_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog60Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_60_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog59Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_59_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog58Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_58_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog57Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_57_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog55Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_55_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog56Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_56_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog54Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_54_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog53Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_53_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog52Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_52_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog51Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_51_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog50Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_50_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog49Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_49_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog48Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_48_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog47Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_47_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog46Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_46_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog44Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_44_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog45Cfg(Go2EnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_DOG_45_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog120Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_120_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog119Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_119_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog118Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_118_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog117Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_117_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog116Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_116_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog115Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_115_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog114Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_114_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog113Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_113_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog112Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_112_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog110Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_110_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog111Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_111_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog186Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_186_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog185Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_185_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog184Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_184_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog183Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_183_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog182Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_182_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog181Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_181_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog180Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_180_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog179Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_179_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog178Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_178_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog176Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_176_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog177Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_177_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog175Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_175_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog174Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_174_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog173Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_173_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog172Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_172_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog171Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_171_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog170Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_170_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog169Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_169_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog168Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_168_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog167Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_167_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog165Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_165_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog166Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_166_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog142Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_142_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog141Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_141_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog140Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_140_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog139Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_139_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog138Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_138_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog137Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_137_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog136Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_136_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog135Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_135_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog134Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_134_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog132Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_132_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog133Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_133_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog131Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_131_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog130Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_130_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog129Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_129_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog128Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_128_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog127Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_127_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog126Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_126_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog125Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_125_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog124Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_124_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog123Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_123_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog121Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_121_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog122Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_122_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog208Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_208_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog207Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_207_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog206Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_206_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog205Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_205_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog204Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_204_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog203Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_203_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog202Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_202_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog201Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_201_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog200Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_200_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog198Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_198_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog199Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_199_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog197Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_197_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog196Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_196_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog195Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_195_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog194Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_194_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog193Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_193_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog192Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_192_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog191Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_191_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog190Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_190_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog189Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_189_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog187Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_187_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog188Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_188_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog164Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_164_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog163Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_163_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog162Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_162_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog161Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_161_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog160Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_160_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog159Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_159_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog158Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_158_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog157Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_157_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog156Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_156_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog154Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_154_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog155Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_155_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog153Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_153_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog152Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_152_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog151Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_151_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog150Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_150_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog149Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_149_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog148Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_148_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog147Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_147_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog146Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_146_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog145Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_145_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog143Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_143_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog144Cfg(Go2EnvCfg):
    action_space = 16
    robot: ArticulationCfg = GEN_DOG_144_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog219Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_219_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog218Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_218_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog217Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_217_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog216Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_216_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog215Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_215_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog214Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_214_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog213Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_213_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog212Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_212_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog211Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_211_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog209Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_209_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog210Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_210_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog285Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_285_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog284Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_284_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog283Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_283_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog282Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_282_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog281Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_281_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog280Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_280_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog279Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_279_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog278Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_278_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog277Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_277_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog275Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_275_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog276Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_276_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog274Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_274_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog273Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_273_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog272Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_272_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog271Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_271_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog270Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_270_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog269Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_269_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog268Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_268_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog267Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_267_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog266Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_266_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog264Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_264_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog265Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_265_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog241Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_241_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog240Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_240_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog239Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_239_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog238Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_238_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog237Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_237_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog236Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_236_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog235Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_235_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog234Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_234_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog233Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_233_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog231Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_231_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog232Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_232_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog230Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_230_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog229Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_229_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog228Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_228_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog227Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_227_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog226Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_226_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog225Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_225_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog224Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_224_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog223Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_223_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog222Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_222_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog220Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_220_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog221Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_221_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog307Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_307_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog306Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_306_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog305Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_305_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog304Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_304_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog303Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_303_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog302Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_302_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog301Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_301_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog300Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_300_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog299Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_299_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog297Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_297_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog298Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_298_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog296Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_296_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog295Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_295_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog294Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_294_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog293Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_293_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog292Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_292_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog291Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_291_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog290Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_290_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog289Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_289_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog288Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_288_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog286Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_286_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog287Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_287_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog263Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_263_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog262Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_262_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog261Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_261_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog260Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_260_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog259Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_259_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog258Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_258_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog257Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_257_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog256Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_256_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog255Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_255_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog253Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_253_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog254Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_254_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog252Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_252_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog251Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_251_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog250Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_250_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog249Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_249_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog248Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_248_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog247Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_247_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog246Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_246_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog245Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_245_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog244Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_244_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog242Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_242_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Gendog243Cfg(Go2EnvCfg):
    action_space = 20
    robot: ArticulationCfg = GEN_DOG_243_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*trunk.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

