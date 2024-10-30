# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from berkeley_humanoid.tasks.direct.locomotion.locomotion_env import LocomotionEnv
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from berkeley_humanoid.assets.unitree import H1_CFG
import berkeley_humanoid.tasks.locomotion.velocity.mdp as mdp

# import math
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
# from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
# from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
# from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
# from omni.isaac.lab.managers import EventTermCfg as EventTerm
# from omni.isaac.lab.managers import RewardTermCfg as RewTerm
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
# from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, ContactSensorCfg, patterns
# from omni.isaac.lab.terrains import TerrainImporterCfg
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensor


touchdown_feet_names = ".*faa"
undesired_contact_names = [".*hfe", ".*haa"]


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""
#     # -- task
#     track_lin_vel_xy_exp = RewTerm(
#         func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )
#     track_ang_vel_z_exp = RewTerm(
#         func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )
#
#     # -- penalties
#     lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
#     ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
#     joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
#     action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
#     feet_air_time = RewTerm(
#         func=mdp.feet_air_time,
#         weight=2.0,
#         params={
#             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=touchdown_feet_names),
#             "command_name": "base_velocity",
#             "threshold_min": 0.2,
#             "threshold_max": 0.5,
#         },
#     )
#     feet_slide = RewTerm(
#         func=mdp.feet_slide,
#         weight=-0.25,
#         params={
#             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=touchdown_feet_names),
#             "asset_cfg": SceneEntityCfg("robot", body_names=touchdown_feet_names),
#         },
#     )
#     undesired_contacts = RewTerm(
#         func=mdp.undesired_contacts,
#         weight=-1.0,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=undesired_contact_names), "threshold": 1.0},
#     )
#
#     # joint_deviation_hip = RewTerm(
#     #     func=mdp.joint_deviation_l1,
#     #     weight=-0.1,
#     #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])},
#     # )
#     # joint_deviation_knee = RewTerm(
#     #     func=mdp.joint_deviation_l1,
#     #     weight=-0.01,
#     #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*KFE"])},
#     # )
#     # -- optional penalties
#     flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
#     dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


# @configclass
# class MySceneCfg(InteractiveSceneCfg):
#     """Configuration for the terrain scene with a legged robot."""
#
#     # ground terrain
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
#     # robot
#     robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
#     joint_gears: list = [
#         50.0,  # left_hip_yaw
#         50.0,  # right_hip_yaw
#         50.0,  # torso
#         50.0,  # left_hip_roll
#         50.0,  # right_hip_roll
#         50.0,  # left_shoulder_pitch
#         50.0,  # right_shoulder_pitch
#         50.0,  # left_hip_pitch
#         50.0,  # right_hip_pitch
#         50.0,  # left_shoulder_roll
#         50.0,  # right_shoulder_roll
#         50.0,  # left_knee
#         50.0,  # right_knee
#         50.0,  # left_shoulder_yaw
#         50.0,  # right_shoulder_yaw
#         50.0,  # left_ankle
#         50.0,  # right_ankle
#         50.0,  # left_elbow
#         50.0,  # right_elbow
#     ]
#
#     # sensors
#     # height_scanner = RayCasterCfg(
#     #     prim_path="{ENV_REGEX_NS}/Robot/torso",
#     #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
#     #     attach_yaw_only=True,
#     #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
#     #     debug_vis=False,
#     #     mesh_prim_paths=["/World/ground"],
#     # )
#     contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
#                                                 track_air_time=True, track_pose=True)
#
#     # # lights
#     # sky_light = AssetBaseCfg(
#     #     prim_path="/World/skyLight",
#     #     spawn=sim_utils.DomeLightCfg(
#     #         intensity=750.0,
#     #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
#     #     ),
#     # )


@configclass
class H1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 19
    observation_space = 69
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0,
                                                     replicate_physics=True)
    # scene = MySceneCfg(num_envs=4096, env_spacing=4.0)

    # robot
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        50.0,  # left_hip_yaw
        50.0,  # right_hip_yaw
        50.0,  # torso
        50.0,  # left_hip_roll
        50.0,  # right_hip_roll
        50.0,  # left_shoulder_pitch
        50.0,  # right_shoulder_pitch
        50.0,  # left_hip_pitch
        50.0,  # right_hip_pitch
        50.0,  # left_shoulder_roll
        50.0,  # right_shoulder_roll
        50.0,  # left_knee
        50.0,  # right_knee
        50.0,  # left_shoulder_yaw
        50.0,  # right_shoulder_yaw
        50.0,  # left_ankle
        50.0,  # right_ankle
        50.0,  # left_elbow
        50.0,  # right_elbow
    ]

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

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class H1DirectEnv(LocomotionEnv):
    cfg: H1EnvCfg

    def __init__(self, cfg: H1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
