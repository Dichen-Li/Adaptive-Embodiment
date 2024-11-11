# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for automatically generated robots.

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from berkeley_humanoid.assets import ISAAC_ASSET_DIR

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
activate_contact_sensors = True
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
init_state_dog = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.6),
    joint_pos={".*": 0.0},
    # joint_pos={
    #     ".*hip.*joint": 0.0,
    #     ".*knee.*joint": 1.0,
    #     ".*thigh.*joint": -0.3
    # },
    joint_vel={".*": 0.0},
)
soft_joint_pos_limit_factor = 0.9
actuators = {
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*joint"],
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
        friction=0.0,
    ),
}
prim_path = "/World/envs/env_.*/Robot"

"""
Robot CFG files
"""

GEN_DOG1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_1/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_2/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_3/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_4/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_5/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

init_state_humanoid = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.6),
    joint_pos={".*": 0.0},
    # joint_pos={
    #     ".*hip.*joint": 0.0,
    #     ".*knee.*joint": 1.0,
    #     ".*thigh.*joint": -0.3
    # },
    joint_vel={".*": 0.0},
)
actuators_humanoid = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "torso_joint",
        ],
        effort_limit=300,
        velocity_limit=100.0,
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            "torso_joint": 200.0,
        },
        damping={
            ".*_hip_yaw_joint": 5.0,
            ".*_hip_roll_joint": 5.0,
            ".*_hip_pitch_joint": 5.0,
            ".*_knee_joint": 5.0,
            "torso_joint": 5.0,
        },
        armature={
            ".*_hip_yaw_joint": 0.01,
            ".*_hip_roll_joint": 0.01,
            ".*_hip_pitch_joint": 0.01,
            ".*_knee_joint": 0.01,
            "torso_joint": 0.01,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_joint"
        ],
        effort_limit=20,
        stiffness=20.0,
        damping=2.0,
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_joint",
            ".*_elbow_joint",
        ],
        effort_limit=300,
        velocity_limit=100.0,
        stiffness={
            ".*_shoulder_joint": 40.0,
            ".*_elbow_joint": 40.0,
        },
        damping={
            ".*_shoulder_joint": 10.0,
            ".*_elbow_joint": 10.0,
        },
        armature={
            ".*_shoulder_joint": 0.01,
            ".*_elbow_joint": 0.01,
        },
    ),
}

GEN_HUMANOID1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_1/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_2/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_3/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_4/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_5/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_6/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_humanoid,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)
