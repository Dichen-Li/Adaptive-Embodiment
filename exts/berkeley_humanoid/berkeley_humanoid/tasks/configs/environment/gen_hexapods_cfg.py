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

from berkeley_humanoid.assets.gen_hexapods import *


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
# class GenDogF0R0KneeJoint0Cfg(GenHexapodEnvCfg):
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
# TODO: Need to tune the specified parameters for hexapod. 
# Currently copy from GenHexapodEnvCfg and only change the number of joints from 4 to 6 to enable running.
class GenHexapodEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    episode_length_s = 20.0
    dt = 0.005
    decimation = 4
    nr_feet = 6
    action_space = 12
    observation_space = 69

    action_dt = dt * decimation

    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        disable_contact_processing=True,
    )
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

    asset_name = "robot"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = MISSING
    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True)
    all_bodies_cfg = SceneEntityCfg("robot", body_names=".*")
    all_joints_cfg = SceneEntityCfg("robot", joint_names=".*")

    # robot-specific config
    trunk_cfg = SceneEntityCfg("robot", body_names="base")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*base.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

    step_sampling_probability = 0.002

    action_scaling_factor = 0.3

    # Reward
    reward_curriculum_steps = 20e6
    tracking_xy_velocity_command_coeff = 2.0    * action_dt
    tracking_yaw_velocity_command_coeff = 1.0   * action_dt
    z_velocity_coeff = 2.0                      * action_dt
    pitch_roll_vel_coeff = 0.05                 * action_dt
    pitch_roll_pos_coeff = 0.2                  * action_dt
    actuator_joint_nominal_diff_coeff = 0.0     * action_dt
    actuator_joint_nominal_diff_joints = []
    joint_position_limit_coeff = 10.0           * action_dt
    joint_acceleration_coeff = 2.5e-7           * action_dt
    joint_torque_coeff = 2e-4                   * action_dt
    action_rate_coeff = 0.01                    * action_dt
    base_height_coeff = 50.0                    * action_dt
    air_time_coeff = 0.1                        * action_dt
    symmetry_air_coeff = 0.5                    * action_dt
    feet_symmetry_pairs = [(0, 5), (1, 4), (2, 3)]
    feet_y_distance_coeff = 2.0                 * action_dt

    # Action delay
    max_nr_action_delay_steps = 1
    mixed_action_delay_chance = 0.05

    # Control
    motor_strength_min = 0.5
    motor_strength_max = 1.5
    p_gain_factor_min = 0.5
    p_gain_factor_max = 1.5
    d_gain_factor_min = 0.5
    d_gain_factor_max = 1.5
    asymmetric_control_factor_min = 0.95
    asymmetric_control_factor_max = 1.05
    p_law_position_offset_min = -0.05
    p_law_position_offset_max = 0.05

    # Initial state
    initial_state_roll_angle_factor = 0.0625
    initial_state_pitch_angle_factor = 0.0625
    initial_state_yaw_angle_factor = 1.0
    initial_state_joint_nominal_position_factor = 0.5
    initial_state_joint_velocity_factor = 0.5
    initial_state_joint_velocity_clip = 20
    initial_state_max_linear_velocity = 0.5
    initial_state_max_angular_velocity = 0.5

    # Observation noise
    joint_position_noise = 0.01
    joint_velocity_noise = 1.5
    trunk_angular_velocity_noise = 0.2
    ground_contact_noise_chance = 0.05
    contact_time_noise_chance = 0.05
    contact_time_noise_factor = 1.0
    gravity_vector_noise = 0.05

    # Observation dropout
    joint_and_feet_dropout_chance = 0.05

    # Model
    static_friction_min = 0.05
    static_friction_max = 2.0
    dynamic_friction_min = 0.05
    dynamic_friction_max = 1.5
    restitution_min = 0.0
    restitution_max = 1.0
    added_trunk_mass_min = -2.0
    added_trunk_mass_max = 2.0
    added_gravity_min = -1.0
    added_gravity_max = 1.0
    joint_friction_min = 0.0
    joint_friction_max = 0.01
    joint_armature_min = 0.0
    joint_armature_max = 0.01

    # Perturbations
    perturb_velocity_x_min = -1.0
    perturb_velocity_x_max = 1.0
    perturb_velocity_y_min = -1.0
    perturb_velocity_y_max = 1.0
    perturb_velocity_z_min = -1.0
    perturb_velocity_z_max = 1.0
    perturb_add_chance = 0.5
    perturb_additive_multiplier = 1.5

@configclass
class Genhexapod10Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_10_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod9Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_9_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod8Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_8_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod7Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_7_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod6Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_6_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod5Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_5_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod4Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_4_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod3Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_3_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod2Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_2_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod0Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_0_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod1Cfg(GenHexapodEnvCfg):
    action_space = 12
    robot: ArticulationCfg = GEN_HEXAPOD_1_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod21Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_21_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod20Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_20_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod19Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_19_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod18Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_18_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod17Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_17_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod16Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_16_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod15Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_15_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod14Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_14_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod13Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_13_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod11Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_11_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod12Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_12_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod43Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_43_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod42Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_42_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod41Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_41_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod40Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_40_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod39Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_39_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod38Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_38_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod37Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_37_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod36Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_36_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod35Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_35_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod33Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_33_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod34Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_34_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod32Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_32_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod31Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_31_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod30Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_30_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod29Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_29_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod28Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_28_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod27Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_27_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod26Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_26_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod25Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_25_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod24Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_24_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod22Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_22_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod23Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_23_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod65Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_65_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod64Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_64_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod63Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_63_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod62Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_62_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod61Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_61_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod60Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_60_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod59Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_59_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod58Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_58_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod57Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_57_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod55Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_55_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod56Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_56_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod54Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_54_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod53Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_53_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod52Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_52_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod51Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_51_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod50Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_50_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod49Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_49_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod48Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_48_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod47Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_47_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod46Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_46_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod44Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_44_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod45Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_45_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod87Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_87_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod86Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_86_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod85Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_85_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod84Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_84_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod83Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_83_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod82Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_82_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod81Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_81_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod80Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_80_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod79Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_79_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod77Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_77_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod78Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_78_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod76Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_76_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod75Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_75_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod74Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_74_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod73Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_73_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod72Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_72_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod71Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_71_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod70Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_70_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod69Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_69_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod68Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_68_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod66Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_66_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod67Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_67_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod109Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_109_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod108Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_108_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod107Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_107_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod106Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_106_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod105Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_105_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod104Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_104_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod103Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_103_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod102Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_102_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod101Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_101_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod99Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_99_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod100Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_100_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod98Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_98_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod97Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_97_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod96Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_96_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod95Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_95_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod94Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_94_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod93Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_93_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod92Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_92_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod91Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_91_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod90Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_90_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod88Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_88_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod89Cfg(GenHexapodEnvCfg):
    action_space = 18
    robot: ArticulationCfg = GEN_HEXAPOD_89_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod120Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_120_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod119Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_119_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod118Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_118_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod117Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_117_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod116Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_116_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod115Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_115_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod114Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_114_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod113Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_113_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod112Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_112_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod110Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_110_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod111Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_111_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod142Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_142_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod141Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_141_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod140Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_140_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod139Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_139_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod138Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_138_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod137Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_137_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod136Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_136_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod135Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_135_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod134Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_134_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod132Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_132_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod133Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_133_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod131Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_131_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod130Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_130_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod129Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_129_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod128Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_128_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod127Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_127_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod126Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_126_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod125Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_125_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod124Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_124_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod123Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_123_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod121Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_121_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod122Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_122_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod164Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_164_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod163Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_163_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod162Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_162_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod161Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_161_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod160Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_160_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod159Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_159_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod158Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_158_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod157Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_157_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod156Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_156_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod154Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_154_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod155Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_155_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod153Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_153_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod152Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_152_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod151Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_151_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod150Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_150_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod149Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_149_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod148Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_148_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod147Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_147_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod146Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_146_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod145Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_145_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod143Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_143_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod144Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_144_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod186Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_186_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod185Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_185_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod184Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_184_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod183Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_183_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod182Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_182_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod181Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_181_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod180Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_180_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod179Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_179_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod178Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_178_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod176Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_176_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod177Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_177_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod175Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_175_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod174Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_174_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod173Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_173_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod172Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_172_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod171Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_171_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod170Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_170_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod169Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_169_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod168Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_168_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod167Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_167_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod165Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_165_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod166Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_166_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod208Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_208_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod207Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_207_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod206Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_206_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod205Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_205_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod204Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_204_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod203Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_203_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod202Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_202_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod201Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_201_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod200Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_200_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod198Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_198_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod199Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_199_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod197Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_197_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod196Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_196_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod195Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_195_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod194Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_194_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod193Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_193_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod192Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_192_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod191Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_191_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod190Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_190_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod189Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_189_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod187Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_187_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod188Cfg(GenHexapodEnvCfg):
    action_space = 24
    robot: ArticulationCfg = GEN_HEXAPOD_188_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod219Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_219_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod218Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_218_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod217Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_217_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod216Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_216_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod215Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_215_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod214Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_214_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod213Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_213_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod212Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_212_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod211Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_211_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod209Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_209_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod210Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_210_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod241Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_241_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod240Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_240_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod239Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_239_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod238Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_238_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod237Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_237_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod236Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_236_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod235Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_235_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod234Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_234_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod233Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_233_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod231Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_231_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod232Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_232_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod230Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_230_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod229Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_229_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod228Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_228_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod227Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_227_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod226Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_226_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod225Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_225_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod224Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_224_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod223Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_223_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod222Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_222_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod220Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_220_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod221Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_221_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod263Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_263_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod262Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_262_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod261Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_261_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod260Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_260_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod259Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_259_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod258Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_258_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod257Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_257_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod256Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_256_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod255Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_255_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod253Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_253_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod254Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_254_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod252Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_252_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod251Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_251_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod250Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_250_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod249Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_249_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod248Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_248_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod247Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_247_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod246Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_246_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod245Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_245_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod244Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_244_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod242Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_242_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod243Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_243_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod285Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_285_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod284Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_284_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod283Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_283_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod282Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_282_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod281Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_281_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod280Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_280_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod279Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_279_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod278Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_278_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod277Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_277_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod275Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_275_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod276Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_276_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod274Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_274_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod273Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_273_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod272Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_272_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod271Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_271_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod270Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_270_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod269Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_269_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod268Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_268_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod267Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_267_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod266Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_266_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod264Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_264_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod265Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_265_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod307Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_307_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod306Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_306_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod305Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_305_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod304Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_304_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod303Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_303_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod302Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_302_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod301Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_301_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod300Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_300_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod299Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_299_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod297Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_297_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod298Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_298_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod296Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_296_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod295Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_295_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod294Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_294_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod293Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_293_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod292Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_292_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod291Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_291_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod290Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_290_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod289Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_289_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod288Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_288_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod286Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_286_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhexapod287Cfg(GenHexapodEnvCfg):
    action_space = 30
    robot: ArticulationCfg = GEN_HEXAPOD_287_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*trunk.*', '.*hip.*', '.*thigh.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

