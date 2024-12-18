
GEN_HEXAPOD_10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_9_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_8_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_21_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_19_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_18_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_17_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_15_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_13_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_11_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_43_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_42_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_41_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_40_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_39_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_38_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_37_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_36_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_35_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_33_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_34_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_32_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_31_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_30_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_29_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_28_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_27_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_26_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_25_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_24_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_22_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_23_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_65_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_64_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_63_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_62_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_61_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_60_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_59_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_58_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_57_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_55_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_56_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_54_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_53_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_52_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_51_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_50_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_49_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_48_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_47_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_46_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_44_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_45_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_87_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_86_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_84_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_83_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_82_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_81_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_80_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_79_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_77_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_78_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_76_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_75_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_74_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_72_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_71_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_70_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_69_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_68_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_66_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_67_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_109_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_108_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_107_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_106_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_105_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_104_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_103_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_102_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_99_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_98_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_97_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_96_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_95_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_94_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_93_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_92_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_91_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_90_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_88_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_89_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_120_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_119_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_118_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_117_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_116_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_115_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_114_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_113_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_112_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_110_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_111_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_142_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_141_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_139_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_138_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_137_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_136_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_135_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_134_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_132_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_133_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_131_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_130_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_129_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_128_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_127_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_126_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_125_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_124_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_123_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_121_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_122_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_164_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_163_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_162_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_161_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_160_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_159_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_158_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_157_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_156_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_154_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_155_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_153_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_152_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_151_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_150_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_149_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_148_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_147_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_146_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_145_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_143_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_144_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_186_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_185_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_184_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_183_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_182_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_181_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_180_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_179_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_178_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_176_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_177_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_175_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_174_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_173_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_172_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_171_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_170_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_169_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_168_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_167_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_165_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_166_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_208_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_207_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_206_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_205_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_204_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_203_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_202_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_201_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_200_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_198_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_199_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_197_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_196_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_195_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_194_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_193_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_192_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_191_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_190_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.19),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_189_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.81),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_187_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_188_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-2_l2-2_l3-2_l4-2_l5-2_l6-2__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_219_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_218_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_217_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_216_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_215_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_214_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_213_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_212_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_211_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_209_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_241_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_240_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_239_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_238_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_237_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_236_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_235_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_234_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_233_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_231_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_232_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_230_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_229_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_228_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_227_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_226_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_225_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_224_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_223_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_222_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_220_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_221_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-0_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_263_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_262_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_261_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_260_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_259_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_258_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_257_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_256_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_255_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_253_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_254_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_252_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_251_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_250_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_249_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_248_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_247_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_246_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_245_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_244_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_242_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_243_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-0_l4-1_l5-0_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_285_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_284_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_283_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_282_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_281_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_280_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_279_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_278_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_277_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_275_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_276_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_274_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_273_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_272_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_271_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_270_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_269_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_268_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_267_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_266_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_264_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_265_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-0_l3-1_l4-0_l5-1_l6-0_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_307_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_306_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_305_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_304_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_303_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_302_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_301_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_300_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_299_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_297_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_298_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_0_8__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_296_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_295_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_294_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_293_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.85),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_292_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_291_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_290_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_289_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_288_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_286_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.31),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)

GEN_HEXAPOD_287_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v0/gen_hexapods/genhexapod__KneeNum_l1-3_l2-3_l3-3_l4-3_l5-3_l6-3__ScaleJointLimit_l1-1_l2-1_l3-1_l4-1_l5-1_l6-1_1_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.57),
        joint_pos={
            ".*_hip_joint": 0.00,
            ".*_thigh_joint": 0.79,
            ".*_knee_joint": 0.79,
            ".*_knee_.*_joint": 0.00
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
    prim_path=prim_path
)
