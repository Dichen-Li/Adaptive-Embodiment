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
from berkeley_humanoid.assets.generated import *
from omni.isaac.lab.sensors import RayCasterCfg, ContactSensorCfg, patterns


@configclass
class GenHumanoidEnvCfg(DirectRLEnvCfg):
    """
    A parent config class that will be inherited by robot-specific config classes
    """
    seed = 42

    # env
    episode_length_s = 20.0
    decimation = 4
    dt = 0.005
    action_space = 15
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
        'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*"]),
        'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint", ".*elbow.*joint", ".*shoulder.*joint",
                                                              ".*torso.*joint"]),
        'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[".*knee.*joint"]),
        'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*head.*", ".*torso.*",
                                                                            ".*arm.*", ".*calf.*"])
    }

    def __init__(self, robot_cfg, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot_cfg  # Set the specific robot configuration


@configclass
class GenHumanoid1Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID1_CFG

@configclass
class GenHumanoid2Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID2_CFG

@configclass
class GenHumanoid3Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID3_CFG

@configclass
class GenHumanoid4Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID4_CFG

@configclass
class GenHumanoid5Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID5_CFG

@configclass
class GenHumanoid6Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID6_CFG


class GenHumanoidDirectEnv(LocomotionEnv):
    cfg: GenHumanoidEnvCfg

    def __init__(self, cfg: GenHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
