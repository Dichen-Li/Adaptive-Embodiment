# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from berkeley_humanoid.assets.unitree import GO2_CFG


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    episode_length_s = 20.0
    dt = 0.005
    decimation = 4
    nr_feet = 4
    action_space = 12
    observation_space = 69

    action_dt = dt * decimation

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

    asset_name = "robot"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, track_pose=True)

    step_sampling_probability = 0.002

    action_scaling_factor = 0.3

    reward_curriculum_steps = 12e6
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
    base_height_coeff = 30.0                    * action_dt
    air_time_coeff = 0.1                        * action_dt
    symmetry_air_coeff = 0.5                    * action_dt

    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*base.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")
