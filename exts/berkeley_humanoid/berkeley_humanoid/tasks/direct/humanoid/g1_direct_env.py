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
from berkeley_humanoid.assets.unitree import G1_CFG
import berkeley_humanoid.tasks.locomotion.velocity.mdp as mdp


import math
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

@configclass
class G1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 37
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
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        50.0
    ] * 37  # self.robot.find_joints(".*") gives 37 but why?

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
    termination_height: float = 0.4     # extremely important

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    # Velocity command configuration (formerly CommandsCfg)
    asset_name = "robot"
    resampling_time_range = (10.0, 10.0)
    rel_standing_envs = 0.02
    rel_heading_envs = 1.0
    heading_command = True
    heading_control_stiffness = 0.5
    debug_vis = True

    # Velocity command ranges
    x_vel_range = (-1.0, 1.0)
    y_vel_range = (-1.0, 1.0)
    yaw_vel_range = (-1.0, 1.0)




class G1DirectEnv(LocomotionEnv):
    cfg: H1EnvCfg

    def __init__(self, cfg: G1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
