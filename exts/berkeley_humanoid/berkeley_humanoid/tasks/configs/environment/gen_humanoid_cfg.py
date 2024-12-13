# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

from berkeley_humanoid.tasks.environments.locomotion_env import LocomotionEnv


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
    x_vel_range = (-1.0, 1.0)   # (0.7, 0.7)
    y_vel_range = (-1.0, 1.0)   # (-0.7, -0.7)
    yaw_vel_range = (-1.0, 1.0)   # (0, 0)
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

@configclass
class GenHumanoidOriginalJoint0Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_0_CFG

@configclass
class GenHumanoidOriginalJoint1Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_1_CFG

@configclass
class GenHumanoidOriginalJoint2Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_2_CFG

@configclass
class GenHumanoidOriginalJoint3Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_3_CFG

@configclass
class GenHumanoidOriginalJoint4Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_4_CFG

@configclass
class GenHumanoidOriginalJoint5Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_5_CFG

@configclass
class GenHumanoidOriginalJoint6Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_ORIGINAL_JOINT_6_CFG

@configclass
class GenHumanoidL0R0Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_L0R0_CFG
    action_space = 13
    # reward configurations
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg("contact_sensor", body_names=".*foot"),
        'feet_ground_asset_cfg': SceneEntityCfg("robot", body_names=".*foot"),
        'undesired_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*calf.*"]),
        'joint_hip_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*joint", ".*elbow.*joint", ".*shoulder.*joint",
                                                              ".*torso.*joint"]),
        'joint_knee_cfg': SceneEntityCfg("robot", joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg("contact_sensor", body_names=[".*head.*", ".*torso.*",
                                                                            ".*arm.*", ".*calf.*"])
    }

@configclass
class GenHumanoidL2R2Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_L2R2_CFG
    action_space = 17

@configclass
class GenHumanoidL3R3Cfg(GenHumanoidEnvCfg):
    robot: ArticulationCfg = GEN_HUMANOID_L3R3_CFG
    action_space = 19

@configclass
class Genhumanoid10Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_10_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid9Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_9_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid8Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_8_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid7Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_7_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid6Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_6_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid5Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_5_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid4Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_4_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid3Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_3_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid2Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_2_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid0Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_0_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid1Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_1_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid14Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_14_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid13Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_13_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid12Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_12_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid11Cfg(GenHumanoidEnvCfg):
    action_space = 13
    robot: ArticulationCfg = GEN_HUMANOID_11_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=[]),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid25Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_25_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid24Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_24_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid23Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_23_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid22Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_22_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid21Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_21_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid20Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_20_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid19Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_19_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid18Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_18_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid17Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_17_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid15Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_15_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid16Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_16_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid29Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_29_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid28Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_28_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid27Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_27_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid26Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_26_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid115Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_115_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid114Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_114_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid113Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_113_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid112Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_112_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid111Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_111_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid110Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_110_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid109Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_109_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid108Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_108_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid107Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_107_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid105Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_105_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid106Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_106_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid119Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_119_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid118Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_118_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid117Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_117_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid116Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_116_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid100Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_100_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid99Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_99_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid98Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_98_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid97Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_97_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid96Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_96_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid95Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_95_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid94Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_94_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid93Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_93_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid92Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_92_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid90Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_90_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid91Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_91_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid104Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_104_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid103Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_103_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid102Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_102_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid101Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_101_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid55Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_55_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid54Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_54_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid53Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_53_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid52Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_52_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid51Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_51_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid50Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_50_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid49Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_49_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid48Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_48_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid47Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_47_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid45Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_45_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid46Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_46_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid59Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_59_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid58Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_58_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid57Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_57_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid56Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_56_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid40Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_40_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid39Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_39_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid38Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_38_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid37Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_37_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid36Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_36_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid35Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_35_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid34Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_34_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid33Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_33_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid32Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_32_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid30Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_30_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid31Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_31_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid44Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_44_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid43Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_43_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid42Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_42_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid41Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_41_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid85Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_85_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid84Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_84_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid83Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_83_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid82Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_82_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid81Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_81_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid80Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_80_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid79Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_79_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid78Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_78_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid77Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_77_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid75Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_75_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid76Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_76_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid89Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_89_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid88Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_88_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid87Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_87_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid86Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_86_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid70Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_70_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid69Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_69_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid68Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_68_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid67Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_67_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid66Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_66_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid65Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_65_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid64Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_64_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid63Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_63_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid62Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_62_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid60Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_60_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid61Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_61_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid74Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_74_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid73Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_73_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid72Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_72_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid71Cfg(GenHumanoidEnvCfg):
    action_space = 15
    robot: ArticulationCfg = GEN_HUMANOID_71_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid130Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_130_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid129Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_129_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid128Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_128_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid127Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_127_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid126Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_126_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid125Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_125_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid124Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_124_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid123Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_123_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid122Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_122_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid120Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_120_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid121Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_121_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid134Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_134_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid133Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_133_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid132Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_132_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid131Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_131_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid220Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_220_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid219Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_219_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid218Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_218_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid217Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_217_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid216Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_216_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid215Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_215_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid214Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_214_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid213Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_213_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid212Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_212_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid210Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_210_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid211Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_211_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid224Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_224_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid223Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_223_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid222Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_222_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid221Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_221_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid205Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_205_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid204Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_204_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid203Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_203_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid202Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_202_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid201Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_201_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid200Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_200_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid199Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_199_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid198Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_198_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid197Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_197_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid195Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_195_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid196Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_196_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid209Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_209_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid208Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_208_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid207Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_207_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid206Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_206_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid160Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_160_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid159Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_159_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid158Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_158_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid157Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_157_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid156Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_156_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid155Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_155_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid154Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_154_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid153Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_153_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid152Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_152_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid150Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_150_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid151Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_151_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid164Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_164_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid163Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_163_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid162Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_162_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid161Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_161_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid145Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_145_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid144Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_144_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid143Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_143_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid142Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_142_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid141Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_141_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid140Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_140_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid139Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_139_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid138Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_138_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid137Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_137_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid135Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_135_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid136Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_136_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid149Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_149_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid148Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_148_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid147Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_147_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid146Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_146_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid190Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_190_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid189Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_189_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid188Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_188_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid187Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_187_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid186Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_186_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid185Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_185_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid184Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_184_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid183Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_183_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid182Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_182_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid180Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_180_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid181Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_181_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid194Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_194_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid193Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_193_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid192Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_192_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid191Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_191_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid175Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_175_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid174Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_174_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid173Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_173_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid172Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_172_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid171Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_171_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid170Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_170_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid169Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_169_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid168Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_168_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid167Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_167_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid165Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_165_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid166Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_166_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid179Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_179_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid178Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_178_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid177Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_177_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid176Cfg(GenHumanoidEnvCfg):
    action_space = 17
    robot: ArticulationCfg = GEN_HUMANOID_176_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid235Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_235_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid234Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_234_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid233Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_233_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid232Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_232_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid231Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_231_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid230Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_230_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid229Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_229_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid228Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_228_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid227Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_227_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid225Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_225_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid226Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_226_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid239Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_239_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid238Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_238_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid237Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_237_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid236Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_236_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid325Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_325_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid324Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_324_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid323Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_323_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid322Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_322_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid321Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_321_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid320Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_320_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid319Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_319_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid318Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_318_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid317Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_317_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid315Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_315_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid316Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_316_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid329Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_329_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid328Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_328_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid327Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_327_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid326Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_326_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid310Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_310_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid309Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_309_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid308Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_308_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid307Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_307_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid306Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_306_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid305Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_305_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid304Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_304_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid303Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_303_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid302Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_302_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid300Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_300_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid301Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_301_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid314Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_314_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid313Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_313_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid312Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_312_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid311Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_311_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid265Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_265_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid264Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_264_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid263Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_263_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid262Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_262_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid261Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_261_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid260Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_260_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid259Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_259_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid258Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_258_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid257Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_257_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid255Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_255_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid256Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_256_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid269Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_269_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid268Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_268_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid267Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_267_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid266Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_266_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid250Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_250_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid249Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_249_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid248Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_248_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid247Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_247_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid246Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_246_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid245Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_245_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid244Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_244_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid243Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_243_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid242Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_242_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid240Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_240_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid241Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_241_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid254Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_254_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid253Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_253_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid252Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_252_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid251Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_251_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid295Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_295_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid294Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_294_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid293Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_293_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid292Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_292_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid291Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_291_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid290Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_290_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid289Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_289_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid288Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_288_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid287Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_287_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid285Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_285_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid286Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_286_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid299Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_299_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid298Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_298_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid297Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_297_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid296Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_296_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid280Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_280_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid279Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_279_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid278Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_278_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid277Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_277_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid276Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_276_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid275Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_275_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid274Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_274_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid273Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_273_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid272Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_272_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid270Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_270_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid271Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_271_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid284Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_284_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid283Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_283_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid282Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_282_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }

@configclass
class Genhumanoid281Cfg(GenHumanoidEnvCfg):
    action_space = 19
    robot: ArticulationCfg = GEN_HUMANOID_281_CFG
    reward_cfgs = {
        'feet_ground_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*foot']),
        'feet_ground_asset_cfg': SceneEntityCfg('robot', body_names=['.*foot']),
        'undesired_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*calf.*']),
        'joint_hip_cfg': SceneEntityCfg('robot', joint_names=['.*hip.*joint', '.*elbow.*joint', '.*shoulder.*joint', '.*torso.*joint']),
        'joint_knee_cfg': SceneEntityCfg('robot', joint_names=['.*knee.*joint']),
        'illegal_contact_cfg': SceneEntityCfg('contact_sensor', body_names=['.*head.*', '.*torso.*', '.*arm.*', '.*calf.*'])
    }
