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

actuators_without_knee = {

    # "legs": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', 'torso_joint'],
    #     effort_limit=300.00,
    #     velocity_limit=100.00,
    #     stiffness={
    #         ".*_hip_yaw_joint": 150.00,
    #         ".*_hip_roll_joint": 150.00,
    #         ".*_hip_pitch_joint": 200.00,
    #         "torso_joint": 200.00,
    #     },
    #     damping={
    #         ".*_hip_yaw_joint": 5.00,
    #         ".*_hip_roll_joint": 5.00,
    #         ".*_hip_pitch_joint": 5.00,
    #         "torso_joint": 5.00
    #     },
    #     armature={
    #         ".*_hip_yaw_joint": 0.01,
    #         ".*_hip_roll_joint": 0.01,
    #         ".*_hip_pitch_joint": 0.01,
    #         "torso_joint": 0.01
    #     },
    # ),

    # "feet": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_ankle_joint'],
    #     effort_limit=20.00,
    #     velocity_limit=None,
    #     stiffness=20.00,
    #     damping=2.00,
    #     armature=0.01,
    # ),

    # "arms": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_shoulder_joint', '.*_elbow_joint'],
    #     effort_limit=300.00,
    #     velocity_limit=100.00,
    #     stiffness={
    #         ".*_shoulder_joint": 40.00,
    #         ".*_elbow_joint": 40.00
    #     },
    #     damping={
    #         ".*_shoulder_joint": 10.00,
    #         ".*_elbow_joint": 10.00
    #     },
    #     armature={
    #         ".*_shoulder_joint": 0.01,
    #         ".*_elbow_joint": 0.01
    #     },
    # ),

    "all": DCMotorCfg(
        joint_names_expr=[
            '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', 'torso_joint',
            '.*_ankle_joint',
            '.*_shoulder_joint', '.*_elbow_joint',
        ],
        effort_limit={
            '.*_hip_yaw_joint': 200,
            '.*_hip_roll_joint': 200,
            '.*_hip_pitch_joint': 200,
            'torso_joint': 200,
            '.*_ankle_joint': 40,
            '.*_shoulder_joint': 40,
            '.*_elbow_joint': 18,
        },
        saturation_effort={
            '.*_hip_yaw_joint': 200,
            '.*_hip_roll_joint': 200,
            '.*_hip_pitch_joint': 200,
            'torso_joint': 200,
            '.*_ankle_joint': 40,
            '.*_shoulder_joint': 40,
            '.*_elbow_joint': 18,
        },
        velocity_limit={
            '.*_hip_yaw_joint': 23,
            '.*_hip_roll_joint': 23,
            '.*_hip_pitch_joint': 23,
            'torso_joint': 23,
            '.*_ankle_joint': 9,
            '.*_shoulder_joint': 9,
            '.*_elbow_joint': 20,
        },
        stiffness=60.0,
        damping=2.0,
        armature=0.01,
    ),
}

actuators_with_knee = {

    # "legs": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee.*joint', 'torso_joint'],
    #     effort_limit=300.00,
    #     velocity_limit=100.00,
    #     stiffness={
    #         ".*_hip_yaw_joint": 150.00,
    #         ".*_hip_roll_joint": 150.00,
    #         ".*_hip_pitch_joint": 200.00,
    #         ".*_knee.*joint": 200.00,
    #         "torso_joint": 200.00,
    #     },
    #     damping={
    #         ".*_hip_yaw_joint": 5.00,
    #         ".*_hip_roll_joint": 5.00,
    #         ".*_hip_pitch_joint": 5.00,
    #         ".*_knee.*joint": 5.00,
    #         "torso_joint": 5.00
    #     },
    #     armature={
    #         ".*_hip_yaw_joint": 0.01,
    #         ".*_hip_roll_joint": 0.01,
    #         ".*_hip_pitch_joint": 0.01,
    #         ".*_knee.*joint": 0.01,
    #         "torso_joint": 0.01
    #     },
    # ),

    # "feet": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_ankle_joint'],
    #     effort_limit=20.00,
    #     velocity_limit=None,
    #     stiffness=20.00,
    #     damping=2.00,
    #     armature=0.01,
    # ),

    # "arms": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_shoulder_joint', '.*_elbow_joint'],
    #     effort_limit=300.00,
    #     velocity_limit=100.00,
    #     stiffness={
    #         ".*_shoulder_joint": 40.00,
    #         ".*_elbow_joint": 40.00
    #     },
    #     damping={
    #         ".*_shoulder_joint": 10.00,
    #         ".*_elbow_joint": 10.00
    #     },
    #     armature={
    #         ".*_shoulder_joint": 0.01,
    #         ".*_elbow_joint": 0.01
    #     },
    # ),

    "all": DCMotorCfg(
        joint_names_expr=[
            '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee.*joint', 'torso_joint',
            '.*_ankle_joint',
            '.*_shoulder_joint', '.*_elbow_joint',
        ],
        effort_limit={
            '.*_hip_yaw_joint': 200, 
            '.*_hip_roll_joint': 200, 
            '.*_hip_pitch_joint': 200, 
            '.*_knee.*joint': 300,
            'torso_joint': 200,
            '.*_ankle_joint': 40,
            '.*_shoulder_joint': 40,
            '.*_elbow_joint': 18,
        },
        saturation_effort={
            '.*_hip_yaw_joint': 200, 
            '.*_hip_roll_joint': 200, 
            '.*_hip_pitch_joint': 200, 
            '.*_knee.*joint': 300,
            'torso_joint': 200,
            '.*_ankle_joint': 40,
            '.*_shoulder_joint': 40,
            '.*_elbow_joint': 18,
        },
        velocity_limit={
            '.*_hip_yaw_joint': 23, 
            '.*_hip_roll_joint': 23, 
            '.*_hip_pitch_joint': 23, 
            '.*_knee.*joint': 14,
            'torso_joint': 23,
            '.*_ankle_joint': 9,
            '.*_shoulder_joint': 9,
            '.*_elbow_joint': 20,
        },
        stiffness=60.0,
        damping=2.0,
        armature=0.01,
    ),
}

GEN_HUMANOID_10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_9_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_8_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.13000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.21000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.00200),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_0_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.49800),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_15_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.39000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_14_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_13_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_11_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.25000),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": 0.00000,
            ".*ankle.*": 0.00000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_without_knee,
    prim_path=prim_path
)

GEN_HUMANOID_26_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_25_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_24_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_23_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_22_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_21_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_19_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_18_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_17_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_31_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_30_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_29_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_28_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_27_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_122_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_121_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_120_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_119_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_118_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_117_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_116_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_115_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_114_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_112_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_113_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_127_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_126_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_125_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_124_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_123_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_106_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_105_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_104_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_103_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_102_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_99_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_98_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_96_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_97_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_111_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_110_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_109_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_108_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_107_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_58_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_57_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_56_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_55_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_54_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_53_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_52_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_51_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_50_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_48_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_49_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_63_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_62_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_61_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_60_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_59_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_42_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_41_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_40_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_39_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_38_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_37_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_36_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_35_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_34_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_32_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_33_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_47_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_46_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_45_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_44_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_43_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_90_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_89_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_88_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_87_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_86_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_84_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_83_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_82_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_80_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_81_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_95_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_94_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_93_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_92_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_91_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_74_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_72_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_71_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_70_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07632),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_69_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15001),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_68_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.22369),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_67_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.29738),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_66_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95148),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_64_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_65_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.42222),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_79_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.32685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_78_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_77_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_76_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_75_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.18685),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*knee.*": 0.80000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_138_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_137_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_136_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_135_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_134_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_133_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_132_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_131_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_130_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_128_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_129_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_143_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_142_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_141_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_139_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_234_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_233_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_232_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_231_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_230_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_229_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_228_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_227_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_226_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_224_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_225_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_239_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_238_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_237_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_236_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_235_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_218_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_217_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_216_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_215_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_214_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_213_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_212_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_211_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_210_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_208_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_209_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_223_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_222_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_221_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_220_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_219_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_170_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_169_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_168_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_167_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_166_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_165_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_164_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_163_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_162_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_160_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_161_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_175_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_174_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_173_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_172_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_171_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_154_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_153_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_152_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_151_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_150_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_149_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_148_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_147_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_146_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_144_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_145_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_159_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_158_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_157_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_156_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_155_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_202_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_201_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_200_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_199_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_198_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_197_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_196_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_195_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_194_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_192_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_193_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_207_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_206_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_205_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_204_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_203_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_186_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_185_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_184_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_183_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_182_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.26053),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_181_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.33422),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_180_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.40790),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_179_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.48159),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_178_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09885),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_176_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_177_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.64327),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_191_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_190_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_189_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_188_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_187_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.37106),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_250_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_249_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_248_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_247_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_246_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_245_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_244_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_243_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_242_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_240_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_241_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_255_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_254_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_253_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_252_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_251_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_346_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_345_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_344_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_343_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_342_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_341_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_340_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_339_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_338_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_336_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_337_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_351_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_350_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_349_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_348_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_347_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_330_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_329_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_328_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_327_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_326_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_325_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_324_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_323_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_322_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_320_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_321_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_335_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_334_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_333_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_332_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_331_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_282_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_281_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_280_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_279_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_278_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_277_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_276_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_275_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_274_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_272_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_273_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_287_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_286_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_285_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_284_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_283_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_266_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_265_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_264_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_263_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_262_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_261_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_260_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_259_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_258_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_256_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_257_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_271_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_270_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_269_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_268_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_267_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_314_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_313_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_312_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_311_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_310_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_309_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_308_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_307_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_306_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_304_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_305_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_319_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_318_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_317_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_316_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_315_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_2__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_298_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_297_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_296_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_295_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_calf_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_294_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.44475),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_293_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.51843),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_292_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.59212),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_291_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_lengthen_thigh_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.66580),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_290_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.24622),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_288_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_289_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_all_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.86433),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_303_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_foot_size_2_0/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.69527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_302_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_4/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_301_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_0_8/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_300_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_2/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)

GEN_HUMANOID_299_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/GenBot1K-v2/gen_humanoids/genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_6__Geo_scale_torso_1_6/usd_file/robot.usd",
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.55527),
        joint_pos={
            ".*torso.*": 0.00000,
            ".*shoulder.*": 0.00000,
            ".*elbow.*": 0.00000,
            ".*yaw.*": 0.00000,
            ".*roll.*": 0.00000,
            ".*pitch.*": -0.40000,
            ".*_knee_joint": 0.80000,
            ".*_knee_.*_joint": 0.00000,
            ".*ankle.*": -0.40000
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators_with_knee,
    prim_path=prim_path
)
