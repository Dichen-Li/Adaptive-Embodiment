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
    decimation = 4
    dt = 0.005
    nr_feet = 4
    action_space = 12
    observation_space = 69

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

    action_scaling_factor = 0.3
    step_sampling_probability = 0.002

    trunk_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*base.*")
    feet_contact_cfg = SceneEntityCfg("contact_sensor", body_names=".*foot")
