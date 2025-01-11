import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from berkeley_humanoid.assets import ISAAC_ASSET_DIR


activate_contact_sensors = True
rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
soft_joint_pos_limit_factor = 0.9
prim_path = "/World/envs/env_.*/Robot"

"""
===================================breakline=======================================
The above do not quite differ for different categories of robots, but the below do
"""

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
actuators = {
    # "base_legs": DCMotorCfg(
    #     joint_names_expr=[".*joint"],
    #     effort_limit=23.5,
    #     saturation_effort=23.5,
    #     velocity_limit=30.0,
    #     stiffness=20.0,
    #     damping=0.5,
    #     friction=0.0,
    # ),
    "all": DCMotorCfg(
        joint_names_expr=[".*joint"],
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=20.0,
        damping=0.5,
        friction=0.0,
    ),
}


"""
Robot CFG files
"""

# GEN_DOG0_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_0_v6/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )
#
# GEN_DOG1_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_1/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )
#
# GEN_DOG2_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_2/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )
#
# GEN_DOG3_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_3/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )
#
# GEN_DOG4_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_4/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )
#
# GEN_DOG5_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Generated/gen_dog_5/robot.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=init_state_dog,
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )


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
Sanity check: import Go2 here to see if the config makes sense 
"""

# GO2_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_ASSET_DIR}/Robots/Unitree/Go2/go2.usd",
#         activate_contact_sensors=activate_contact_sensors,
#         rigid_props=rigid_props,
#         articulation_props=articulation_props,
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.325),
#         joint_pos={
#             ".*L_hip_joint": 0.1,
#             ".*R_hip_joint": -0.1,
#             "F[L,R]_thigh_joint": 0.8,
#             "R[L,R]_thigh_joint": 1.0,
#             ".*_calf_joint": -1.5,
#         },
#         # joint_pos={".*": 0.0},
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
#     actuators=actuators,
#     prim_path=prim_path
# )


GEN_DOG_10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_9_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_8_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_11_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.00,
            "rear_.*_thigh_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_22_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_21_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_19_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_18_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_17_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_15_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_13_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_23_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_94_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_93_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_92_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_91_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_90_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_89_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_88_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_87_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_86_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_84_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_95_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_82_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_81_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_80_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_79_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_78_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_77_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_76_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_75_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_74_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_72_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_83_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_46_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_45_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_44_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_43_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_42_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_41_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_40_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_39_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_38_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_36_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_37_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_47_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_34_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_33_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_32_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_31_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_30_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_29_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_28_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_27_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_26_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_24_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_25_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_35_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_118_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_117_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_116_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_115_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_114_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_113_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_112_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_111_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_110_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_108_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_109_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_119_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_106_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_105_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_104_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_103_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_102_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_99_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_98_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_96_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_97_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_107_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_70_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_69_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_68_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_67_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_66_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_65_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_64_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_63_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_62_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_60_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_61_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_71_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_58_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_57_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_56_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_55_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_54_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_53_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_52_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_51_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_50_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_48_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_49_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_59_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_130_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_129_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_128_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_127_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_126_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_125_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_124_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_123_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_122_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_120_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_121_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_131_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_202_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_201_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_200_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_199_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_198_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_197_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_196_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_195_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_194_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_192_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_193_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_203_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_190_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_189_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_188_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_187_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_186_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_185_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_184_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_183_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_182_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_180_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_181_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_191_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_154_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_153_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_152_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_151_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_150_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_149_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_148_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_147_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_146_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_144_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_145_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_155_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_142_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_141_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_139_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_138_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_137_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_136_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_135_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_134_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_132_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_133_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_143_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_226_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_225_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_224_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_223_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_222_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_221_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_220_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_219_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_218_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_216_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_217_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_227_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_214_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_213_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_212_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_211_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_209_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_208_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_207_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_206_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_204_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_205_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_215_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_178_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_177_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_176_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_175_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_174_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_173_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_172_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_171_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_170_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_168_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.47),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_169_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_179_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.49),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_166_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_165_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_164_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_163_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_162_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_161_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_160_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_159_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_158_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_156_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_157_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_167_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_238_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_237_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_236_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_235_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_234_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_233_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_232_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_231_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_230_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_228_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_229_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_239_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_310_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_309_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_308_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_307_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_306_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_305_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_304_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_303_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.64),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_302_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_300_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_301_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.67),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_311_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_298_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_297_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_296_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_295_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_294_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_293_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_292_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_291_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_290_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_288_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_289_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_299_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_262_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_261_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_260_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_259_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_258_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_257_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_256_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_255_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.64),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_254_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_252_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_253_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.67),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_263_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_250_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_249_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_248_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_247_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_246_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_245_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_244_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_243_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_242_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_240_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_241_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_251_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_334_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_333_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_332_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_331_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_330_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_329_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_328_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_327_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.64),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_326_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_324_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_325_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.67),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_335_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_322_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_321_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_320_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_319_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_318_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_317_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_316_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_315_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_314_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_312_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_313_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_323_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_286_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_285_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.51),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_284_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_283_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_282_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_281_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_280_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_279_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.64),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_278_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_276_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_277_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.67),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_287_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.59,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_274_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.41),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_273_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_272_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_271_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_270_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_269_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_268_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_267_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_266_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_264_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.59),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_265_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_DOG_275_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v1/gen_dogs/gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            ".*_hip_pitch_joint": 0.00,
            "front_.*_thigh_joint": 0.80,
            "rear_.*_thigh_joint": 1.00,
            ".*_knee_joint": -1.50,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)
