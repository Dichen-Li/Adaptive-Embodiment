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
    pos=(0.0, 0.0, 0.5),
    # joint_pos={".*": 0.0},
    joint_pos={
        ".*_left_hip_pitch_joint": 0.1,
        ".*_right_hip_pitch_joint": -0.1,
        "front_left_thigh_joint": 0.8,
        "front_right_thigh_joint": 0.8,
        "rear_left_thigh_joint": 1.0,
        "rear_right_thigh_joint": 1.0,
        ".*_knee_joint": -1.5,
    },
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

GEN_DOG0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_0_v6/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=init_state_dog,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

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


# Add robot configurations dynamically using code
def create_robot_config(name):
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/{name}/robot.usd",
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        ),
        init_state=init_state_humanoid if "humanoid" in name else init_state_dog,
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators=actuators_humanoid if "humanoid" in name else actuators,
        prim_path=prim_path,
    )

# Mapping robot file names to canonical CFG names
robot_name_map = {
    # "gen_dog_f0r0_knee_joint_0": "GEN_DOG_F0R0_CFG",
    "gen_dog_f0r1_knee_joint_0": "GEN_DOG_F0R1_CFG",
    "gen_dog_f1r0_knee_joint_0": "GEN_DOG_F1R0_CFG",
    "gen_dog_f2r2_knee_joint_0": "GEN_DOG_F2R2_CFG",
    "gen_dog_f2r3_knee_joint_0": "GEN_DOG_F2R3_CFG",
    "gen_dog_f3r2_knee_joint_0": "GEN_DOG_F3R2_CFG",
    "gen_dog_original_joint_0": "GEN_DOG_ORIGINAL_0_CFG",
    # "gen_dog_original_joint_1": "GEN_DOG_ORIGINAL_1_CFG",
    # "gen_dog_original_joint_2": "GEN_DOG_ORIGINAL_2_CFG",
    "gen_dog_original_joint_3": "GEN_DOG_ORIGINAL_3_CFG",
    "gen_dog_original_joint_4": "GEN_DOG_ORIGINAL_4_CFG",
    # "gen_dog_original_joint_5": "GEN_DOG_ORIGINAL_5_CFG",
    "gen_dog_original_joint_6": "GEN_DOG_ORIGINAL_6_CFG",
    "gen_dog_original_joint_7": "GEN_DOG_ORIGINAL_7_CFG",
    "gen_dog_original_joint_8": "GEN_DOG_ORIGINAL_8_CFG",
}

# Dynamically generate the CFG objects
for robot_file_name, cfg_name in robot_name_map.items():
    globals()[cfg_name] = create_robot_config(robot_file_name)

# F0R0 is kind of different, as it has no knee joints
GEN_DOG_F0R0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_f0r0_knee_joint_0/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # joint_pos={".*": 0.0},
        joint_pos={
            ".*_left_hip_pitch_joint": 0.1,
            ".*_right_hip_pitch_joint": -0.1,
            "front_left_thigh_joint": 0.8,
            "front_right_thigh_joint": 0.8,
            "rear_left_thigh_joint": 1.0,
            "rear_right_thigh_joint": 1.0,
            # ".*_knee_joint": -1.5,
        },
        # joint_pos={
        #     ".*hip.*joint": 0.0,
        #     ".*knee.*joint": 1.0,
        #     ".*thigh.*joint": -0.3
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_ORIGINAL_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_original_joint_1/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        # joint_pos={".*": 0.0},
        joint_pos={
            ".*_left_hip_pitch_joint": 0.1,
            ".*_right_hip_pitch_joint": -0.1,
            "front_left_thigh_joint": 0.8,
            "front_right_thigh_joint": 0.8,
            "rear_left_thigh_joint": 1.0,
            "rear_right_thigh_joint": 1.0,
            ".*_knee_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_ORIGINAL_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_original_joint_2/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        # joint_pos={".*": 0.0},
        joint_pos={
            ".*_left_hip_pitch_joint": 0.1,
            ".*_right_hip_pitch_joint": -0.1,
            "front_left_thigh_joint": 0.8,
            "front_right_thigh_joint": 0.8,
            "rear_left_thigh_joint": 1.0,
            "rear_right_thigh_joint": 1.0,
            ".*_knee_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_ORIGINAL_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_original_joint_5/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        # joint_pos={".*": 0.0},
        joint_pos={
            ".*_left_hip_pitch_joint": 0.1,
            ".*_right_hip_pitch_joint": -0.1,
            "front_left_thigh_joint": 0.8,
            "front_right_thigh_joint": 0.8,
            "rear_left_thigh_joint": 1.0,
            "rear_right_thigh_joint": 1.0,
            ".*_knee_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

"""
Generated humanoids
"""

init_state_humanoid = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.5),
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

GEN_HUMANOID_ORIGINAL_JOINT_0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_0/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.4),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_1/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.1),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_2/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.12),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_3/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.4),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_4/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.4),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_5/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.8),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_ORIGINAL_JOINT_6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_original_joint_6/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_L0R0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_l0r0_knee_joint_0/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9 + 0.4),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                # ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                # ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                # ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.01,
                ".*_hip_roll_joint": 0.01,
                ".*_hip_pitch_joint": 0.01,
                # ".*_knee_joint": 0.01,
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
    },
    prim_path=prim_path
)

GEN_HUMANOID_L2R2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_l2r2_knee_joint_0_new/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.6),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)

GEN_HUMANOID_L3R3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_humanoid_l3r3_knee_joint_0_new/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.8),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_humanoid,
    prim_path=prim_path
)
