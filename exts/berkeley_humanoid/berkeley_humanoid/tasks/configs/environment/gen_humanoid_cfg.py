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

from berkeley_humanoid.assets.gen_humanoids import *


class GenHumanoidEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    episode_length_s = 20.0
    dt = 0.005
    decimation = 4
    nr_feet = 2
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
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*']) # TODO: maybe we need to tune
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

    step_sampling_probability = 0.002

    action_scaling_factor = 0.3

    # Reward
    reward_curriculum_steps = 300e6
    tracking_xy_velocity_command_coeff = 3.0    * action_dt
    tracking_yaw_velocity_command_coeff = 1.5   * action_dt
    z_velocity_coeff = 2.0                      * action_dt
    pitch_roll_vel_coeff = 0.05                 * action_dt
    pitch_roll_pos_coeff = 0.2                  * action_dt
    actuator_joint_nominal_diff_coeff = 0.0     * action_dt
    actuator_joint_nominal_diff_joints = []
    joint_position_limit_coeff = 120.0          * action_dt
    joint_acceleration_coeff = 3e-6             * action_dt
    joint_torque_coeff = 2.4e-3                 * action_dt
    action_rate_coeff = 0.12                    * action_dt
    base_height_coeff = 30.0                    * action_dt
    air_time_coeff = 0.1                        * action_dt
    symmetry_air_coeff = 0.5                    * action_dt
    feet_symmetry_pairs = [(0, 1)]
    feet_y_distance_coeff = 2.0                 * action_dt

    # Domain randomization
    domain_randomization_curriculum_steps = 300e6
    ## Action delay
    max_nr_action_delay_steps = 1
    mixed_action_delay_chance = 0.05
    ## Control
    motor_strength_min = 0.5
    motor_strength_max = 1.5
    p_gain_factor_min = 0.5
    p_gain_factor_max = 1.5
    d_gain_factor_min = 0.5
    d_gain_factor_max = 1.5
    p_law_position_offset_min = -0.05
    p_law_position_offset_max = 0.05
    ## Initial state
    initial_state_roll_angle_factor = 0.0625
    initial_state_pitch_angle_factor = 0.0625
    initial_state_yaw_angle_factor = 1.0
    initial_state_joint_nominal_position_factor = 0.5
    initial_state_joint_velocity_factor = 0.5
    initial_state_joint_velocity_clip = 1
    initial_state_max_linear_velocity = 0.5
    initial_state_max_angular_velocity = 0.5
    ## Observation noise
    joint_position_noise = 0.01
    joint_velocity_noise = 1.5
    trunk_angular_velocity_noise = 0.2
    ground_contact_noise_chance = 0.05
    contact_time_noise_chance = 0.05
    contact_time_noise_factor = 1.0
    gravity_vector_noise = 0.05
    ## Observation dropout
    joint_and_feet_dropout_chance = 0.05
    ## Model
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
    ## Perturbations
    perturb_velocity_x_min = -1.0
    perturb_velocity_x_max = 1.0
    perturb_velocity_y_min = -1.0
    perturb_velocity_y_max = 1.0
    perturb_velocity_z_min = -1.0
    perturb_velocity_z_max = 1.0
    perturb_add_chance = 0.5
    perturb_additive_multiplier = 1.5

@configclass
class Genhumanoid10Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_10_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid9Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_9_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid8Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_8_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid7Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_7_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid6Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_6_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid5Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_5_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid4Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_4_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid3Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_3_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid2Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_2_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid0Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_0_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid1Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_1_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid15Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_15_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid14Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_14_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid13Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_13_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid12Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_12_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid11Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_11_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid26Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_26_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid25Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_25_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid24Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_24_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid23Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_23_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid22Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_22_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid21Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_21_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid20Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_20_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid19Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_19_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid18Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_18_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid16Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_16_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid17Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_17_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid31Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_31_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid30Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_30_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid29Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_29_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid28Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_28_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid27Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_27_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid122Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_122_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid121Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_121_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid120Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_120_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid119Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_119_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid118Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_118_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid117Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_117_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid116Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_116_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid115Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_115_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid114Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_114_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid112Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_112_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid113Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_113_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid127Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_127_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid126Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_126_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid125Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_125_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid124Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_124_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid123Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_123_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid106Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_106_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid105Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_105_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid104Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_104_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid103Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_103_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid102Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_102_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid101Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_101_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid100Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_100_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid99Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_99_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid98Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_98_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid96Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_96_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid97Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_97_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid111Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_111_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid110Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_110_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid109Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_109_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid108Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_108_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid107Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_107_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid58Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_58_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid57Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_57_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid56Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_56_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid55Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_55_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid54Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_54_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid53Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_53_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid52Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_52_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid51Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_51_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid50Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_50_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid48Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_48_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid49Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_49_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid63Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_63_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid62Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_62_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid61Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_61_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid60Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_60_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid59Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_59_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid42Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_42_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid41Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_41_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid40Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_40_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid39Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_39_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid38Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_38_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid37Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_37_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid36Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_36_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid35Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_35_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid34Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_34_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid32Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_32_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid33Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_33_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid47Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_47_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid46Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_46_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid45Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_45_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid44Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_44_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid43Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_43_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid90Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_90_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid89Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_89_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid88Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_88_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid87Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_87_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid86Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_86_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid85Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_85_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid84Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_84_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid83Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_83_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid82Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_82_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid80Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_80_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid81Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_81_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid95Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_95_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid94Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_94_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid93Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_93_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid92Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_92_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid91Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_91_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid74Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_74_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid73Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_73_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid72Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_72_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid71Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_71_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid70Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_70_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid69Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_69_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid68Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_68_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid67Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_67_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid66Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_66_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid64Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_64_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid65Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_65_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid79Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_79_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid78Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_78_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid77Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_77_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid76Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_76_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid75Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_75_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid138Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_138_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid137Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_137_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid136Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_136_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid135Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_135_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid134Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_134_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid133Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_133_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid132Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_132_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid131Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_131_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid130Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_130_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid128Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_128_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid129Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_129_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid143Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_143_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid142Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_142_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid141Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_141_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid140Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_140_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid139Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_139_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid234Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_234_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid233Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_233_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid232Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_232_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid231Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_231_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid230Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_230_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid229Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_229_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid228Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_228_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid227Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_227_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid226Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_226_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid224Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_224_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid225Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_225_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid239Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_239_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid238Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_238_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid237Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_237_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid236Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_236_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid235Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_235_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid218Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_218_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid217Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_217_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid216Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_216_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid215Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_215_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid214Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_214_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid213Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_213_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid212Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_212_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid211Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_211_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid210Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_210_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid208Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_208_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid209Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_209_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid223Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_223_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid222Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_222_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid221Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_221_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid220Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_220_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid219Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_219_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid170Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_170_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid169Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_169_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid168Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_168_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid167Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_167_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid166Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_166_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid165Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_165_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid164Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_164_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid163Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_163_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid162Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_162_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid160Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_160_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid161Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_161_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid175Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_175_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid174Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_174_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid173Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_173_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid172Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_172_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid171Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_171_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid154Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_154_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid153Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_153_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid152Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_152_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid151Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_151_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid150Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_150_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid149Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_149_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid148Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_148_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid147Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_147_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid146Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_146_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid144Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_144_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid145Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_145_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid159Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_159_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid158Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_158_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid157Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_157_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid156Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_156_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid155Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_155_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid202Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_202_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid201Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_201_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid200Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_200_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid199Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_199_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid198Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_198_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid197Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_197_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid196Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_196_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid195Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_195_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid194Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_194_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid192Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_192_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid193Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_193_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid207Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_207_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid206Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_206_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid205Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_205_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid204Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_204_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid203Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_203_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid186Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_186_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid185Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_185_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid184Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_184_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid183Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_183_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid182Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_182_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid181Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_181_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid180Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_180_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid179Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_179_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid178Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_178_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid176Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_176_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid177Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_177_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid191Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_191_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid190Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_190_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid189Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_189_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid188Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_188_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid187Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_187_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid250Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_250_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid249Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_249_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid248Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_248_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid247Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_247_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid246Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_246_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid245Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_245_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid244Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_244_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid243Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_243_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid242Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_242_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid240Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_240_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid241Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_241_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid255Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_255_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid254Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_254_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid253Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_253_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid252Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_252_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid251Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_251_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid346Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_346_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid345Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_345_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid344Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_344_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid343Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_343_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid342Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_342_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid341Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_341_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid340Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_340_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid339Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_339_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid338Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_338_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid336Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_336_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid337Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_337_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid351Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_351_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid350Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_350_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid349Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_349_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid348Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_348_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid347Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_347_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid330Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_330_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid329Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_329_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid328Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_328_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid327Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_327_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid326Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_326_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid325Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_325_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid324Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_324_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid323Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_323_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid322Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_322_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid320Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_320_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid321Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_321_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid335Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_335_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid334Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_334_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid333Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_333_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid332Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_332_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid331Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_331_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid282Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_282_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid281Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_281_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid280Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_280_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid279Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_279_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid278Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_278_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid277Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_277_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid276Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_276_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid275Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_275_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid274Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_274_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid272Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_272_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid273Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_273_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid287Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_287_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid286Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_286_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid285Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_285_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid284Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_284_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid283Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_283_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid266Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_266_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid265Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_265_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid264Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_264_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid263Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_263_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid262Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_262_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid261Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_261_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid260Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_260_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid259Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_259_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid258Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_258_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid256Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_256_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid257Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_257_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid271Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_271_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid270Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_270_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid269Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_269_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid268Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_268_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid267Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_267_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid314Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_314_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid313Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_313_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid312Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_312_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid311Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_311_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid310Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_310_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid309Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_309_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid308Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_308_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid307Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_307_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid306Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_306_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid304Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_304_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid305Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_305_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid319Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_319_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid318Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_318_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid317Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_317_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid316Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_316_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid315Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_315_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid298Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_298_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid297Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_297_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid296Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_296_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid295Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_295_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid294Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_294_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid293Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_293_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid292Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_292_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid291Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_291_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid290Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_290_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid288Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_288_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid289Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_289_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid303Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_303_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid302Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_302_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid301Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_301_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid300Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_300_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

@configclass
class Genhumanoid299Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_299_CFG
    trunk_cfg = SceneEntityCfg("robot", body_names="pelvis")
    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=['.*pelvis.*', '.*hip.*', '.*thigh.*', '.*torso.*', '.*arm.*', '.*head.*'])
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")

