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

from berkeley_humanoid.assets.berkeley_humanoid import BERKELEY_HUMANOID_CFG
from berkeley_humanoid.tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class BerkeleyHumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    dt = 0.005
    action_space = 12
    observation_space = 69
    # state_space = 0

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
    # scene = MySceneCfg(num_envs=4096, env_spacing=4.0)

    # robot
    robot: ArticulationCfg = BERKELEY_HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # joint_gears: list = [
    #     50.0
    # ] * 12  # self.robot.find_joints(".*") gives 37 but why?

    # sensor for reward calculation
    contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
                                      track_air_time=True, track_pose=True)

    # # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

    # heading_weight: float = 0.5
    # up_weight: float = 0.1
    #
    # energy_cost_scale: float = 0.05
    # actions_cost_scale: float = 0.01
    # alive_reward_scale: float = 2.0
    # dof_vel_scale: float = 0.1

    # death_cost: float = -1.0
    # termination_height: float = 0.3     # extremely important

    # angular_velocity_scale: float = 0.25
    # contact_force_scale: float = 0.01

    # Velocity command configuration (formerly CommandsCfg)
    asset_name = "robot"
    # resampling_time_range = (10.0, 10.0)
    # rel_standing_envs = 0.02
    # rel_heading_envs = 1.0
    # heading_command = True
    # heading_control_stiffness = 0.5
    # debug_vis = True

    # Velocity command ranges
    x_vel_range = (-1.0, 1.0)
    y_vel_range = (-1.0, 1.0)
    yaw_vel_range = (-1.0, 1.0)
    resampling_interval = 10 / (dt * decimation)   # time before the command is changed in sec
    # dt * decimation is actually the time duration that corresponds to one env step

    # controller
    controller_use_offset = True
    action_scale = 0.5
    controlled_joints = ".*"

    # reward configs
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*faa"),
        'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*faa"),
        'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*hfe", ".*haa"]),
        'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"]),
        'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[".*KFE"]),
        'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names='torso')
    }
