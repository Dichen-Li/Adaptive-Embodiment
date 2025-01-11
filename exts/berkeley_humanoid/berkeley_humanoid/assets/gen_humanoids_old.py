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
actuators = {

    # "legs": ImplicitActuatorCfg(
    #     joint_names_expr=['.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee.*joint', 'torso_joint'],
    #     effort_limit=300.00,
    #     velocity_limit=100.00,
    #     stiffness={
    #         ".*_hip_yaw_joint": 150.00,
    #         ".*_hip_roll_joint": 150.00,
    #         ".*_hip_pitch_joint": 200.00,
    #         ".*_knee.*joint": 200.00,
    #         "torso_joint": 200.00
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

    "all": ImplicitActuatorCfg(
        joint_names_expr=[
            '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee.*joint', 'torso_joint',
            '.*_ankle_joint',
            '.*_shoulder_joint', '.*_elbow_joint',
        ],
        effort_limit={
            '.*_hip_yaw_joint': 300, 
            '.*_hip_roll_joint': 300, 
            '.*_hip_pitch_joint': 300, 
            '.*_knee.*joint': 300,
            'torso_joint': 300,
            '.*_ankle_joint': 20,
            '.*_shoulder_joint': 300, 
            '.*_elbow_joint': 300,
        },
        velocity_limit={
            '.*_hip_yaw_joint': 100, 
            '.*_hip_roll_joint': 100, 
            '.*_hip_pitch_joint': 100, 
            '.*_knee.*joint': 100,
            'torso_joint': 100,
            '.*_ankle_joint': 100, # this is edited by tmu, original value is None
            '.*_shoulder_joint': 100, 
            '.*_elbow_joint': 100,
        },
        stiffness={
            ".*_hip_yaw_joint": 150.00,
            ".*_hip_roll_joint": 150.00,
            ".*_hip_pitch_joint": 200.00,
            '.*_knee.*joint': 200,
            "torso_joint": 200.00,
            '.*_ankle_joint': 20,
            ".*_shoulder_joint": 40.00,
            ".*_elbow_joint": 40.00,
        },
        damping={
            ".*_hip_yaw_joint": 5.00,
            ".*_hip_roll_joint": 5.00,
            ".*_hip_pitch_joint": 5.00,
            '.*_knee.*joint': 5.00,
            "torso_joint": 5.00,
            '.*_ankle_joint': 2.00,
            ".*_shoulder_joint": 10.00,
            ".*_elbow_joint": 10.00,
        },
        armature=0.01,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
    actuators=actuators,
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
        # "legs": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_hip_yaw_joint",
        #         ".*_hip_roll_joint",
        #         ".*_hip_pitch_joint",
        #         # ".*_knee_joint",
        #         "torso_joint",
        #     ],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_hip_yaw_joint": 150.0,
        #         ".*_hip_roll_joint": 150.0,
        #         ".*_hip_pitch_joint": 200.0,
        #         # ".*_knee_joint": 200.0,
        #         "torso_joint": 200.0,
        #     },
        #     damping={
        #         ".*_hip_yaw_joint": 5.0,
        #         ".*_hip_roll_joint": 5.0,
        #         ".*_hip_pitch_joint": 5.0,
        #         # ".*_knee_joint": 5.0,
        #         "torso_joint": 5.0,
        #     },
        #     armature={
        #         ".*_hip_yaw_joint": 0.01,
        #         ".*_hip_roll_joint": 0.01,
        #         ".*_hip_pitch_joint": 0.01,
        #         # ".*_knee_joint": 0.01,
        #         "torso_joint": 0.01,
        #     },
        # ),
        # "feet": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_ankle_joint"
        #     ],
        #     effort_limit=20,
        #     stiffness=20.0,
        #     damping=2.0,
        #     armature=0.01,
        # ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_joint",
        #         ".*_elbow_joint",
        #     ],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_shoulder_joint": 40.0,
        #         ".*_elbow_joint": 40.0,
        #     },
        #     damping={
        #         ".*_shoulder_joint": 10.0,
        #         ".*_elbow_joint": 10.0,
        #     },
        #     armature={
        #         ".*_shoulder_joint": 0.01,
        #         ".*_elbow_joint": 0.01,
        #     },
        # ),
        "all": ImplicitActuatorCfg(
            joint_names_expr=[
                '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', 'torso_joint',
                '.*_ankle_joint',
                '.*_shoulder_joint', '.*_elbow_joint',
            ],
            effort_limit={
                '.*_hip_yaw_joint': 300, 
                '.*_hip_roll_joint': 300, 
                '.*_hip_pitch_joint': 300, 
                'torso_joint': 300,
                '.*_ankle_joint': 20,
                '.*_shoulder_joint': 300, 
                '.*_elbow_joint': 300,
            },
            velocity_limit={
                '.*_hip_yaw_joint': 100, 
                '.*_hip_roll_joint': 100, 
                '.*_hip_pitch_joint': 100, 
                'torso_joint': 100,
                '.*_ankle_joint': 100, # this is edited by tmu, original value is None
                '.*_shoulder_joint': 100, 
                '.*_elbow_joint': 100,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.00,
                ".*_hip_roll_joint": 150.00,
                ".*_hip_pitch_joint": 200.00,
                "torso_joint": 200.00,
                '.*_ankle_joint': 20,
                ".*_shoulder_joint": 40.00,
                ".*_elbow_joint": 40.00,
            },
            damping={
                ".*_hip_yaw_joint": 5.00,
                ".*_hip_roll_joint": 5.00,
                ".*_hip_pitch_joint": 5.00,
                "torso_joint": 5.00,
                '.*_ankle_joint': 2.00,
                ".*_shoulder_joint": 10.00,
                ".*_elbow_joint": 10.00,
            },
            armature=0.01,
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
    actuators=actuators,
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
    actuators=actuators,
    prim_path=prim_path
)


"""
Generated humanoids 
"""

#################################################
# Generated humanoids (heavily edited by tmu)
#################################################

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

    "all": ImplicitActuatorCfg(
        joint_names_expr=[
            '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', 'torso_joint',
            '.*_ankle_joint',
            '.*_shoulder_joint', '.*_elbow_joint',
        ],
        effort_limit={
            '.*_hip_yaw_joint': 300, 
            '.*_hip_roll_joint': 300, 
            '.*_hip_pitch_joint': 300, 
            'torso_joint': 300,
            '.*_ankle_joint': 20,
            '.*_shoulder_joint': 300, 
            '.*_elbow_joint': 300,
        },
        velocity_limit={
            '.*_hip_yaw_joint': 100, 
            '.*_hip_roll_joint': 100, 
            '.*_hip_pitch_joint': 100, 
            'torso_joint': 100,
            '.*_ankle_joint': 100, # this is edited by tmu, original value is None
            '.*_shoulder_joint': 100, 
            '.*_elbow_joint': 100,
        },
        stiffness={
            ".*_hip_yaw_joint": 150.00,
            ".*_hip_roll_joint": 150.00,
            ".*_hip_pitch_joint": 200.00,
            "torso_joint": 200.00,
            '.*_ankle_joint': 20,
            ".*_shoulder_joint": 40.00,
            ".*_elbow_joint": 40.00,
        },
        damping={
            ".*_hip_yaw_joint": 5.00,
            ".*_hip_roll_joint": 5.00,
            ".*_hip_pitch_joint": 5.00,
            "torso_joint": 5.00,
            '.*_ankle_joint': 2.00,
            ".*_shoulder_joint": 10.00,
            ".*_elbow_joint": 10.00,
        },
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

    "all": ImplicitActuatorCfg(
        joint_names_expr=[
            '.*_hip_yaw_joint', '.*_hip_roll_joint', '.*_hip_pitch_joint', '.*_knee.*joint', 'torso_joint',
            '.*_ankle_joint',
            '.*_shoulder_joint', '.*_elbow_joint',
        ],
        effort_limit={
            '.*_hip_yaw_joint': 300, 
            '.*_hip_roll_joint': 300, 
            '.*_hip_pitch_joint': 300, 
            '.*_knee.*joint': 300,
            'torso_joint': 300,
            '.*_ankle_joint': 20,
            '.*_shoulder_joint': 300, 
            '.*_elbow_joint': 300,
        },
        velocity_limit={
            '.*_hip_yaw_joint': 100, 
            '.*_hip_roll_joint': 100, 
            '.*_hip_pitch_joint': 100, 
            '.*_knee.*joint': 100,
            'torso_joint': 100,
            '.*_ankle_joint': 100, # this is edited by tmu, original value is None
            '.*_shoulder_joint': 100, 
            '.*_elbow_joint': 100,
        },
        stiffness={
            ".*_hip_yaw_joint": 150.00,
            ".*_hip_roll_joint": 150.00,
            ".*_hip_pitch_joint": 200.00,
            '.*_knee.*joint': 200,
            "torso_joint": 200.00,
            '.*_ankle_joint': 20,
            ".*_shoulder_joint": 40.00,
            ".*_elbow_joint": 40.00,
        },
        damping={
            ".*_hip_yaw_joint": 5.00,
            ".*_hip_roll_joint": 5.00,
            ".*_hip_pitch_joint": 5.00,
            '.*_knee.*joint': 5.00,
            "torso_joint": 5.00,
            '.*_ankle_joint': 2.00,
            ".*_shoulder_joint": 10.00,
            ".*_elbow_joint": 10.00,
        },
        armature=0.01,
    ),
}

import os, json

base_dir = os.path.join(ISAAC_ASSET_DIR, "Robots/GenBot1K-v0/gen_humanoids")
robot_dirs = sorted(
    [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
)


for i, robot_dir in enumerate(robot_dirs):
    with open(f"{robot_dir}/train_cfg.json", "r") as f:
        train_cfg = json.load(f)

    robot_name = train_cfg.get("robot_name")
    cfg_name = f"{robot_name.upper()}_CFG"
    robot_idx = int(robot_name.split("_")[-1])
    init_height = train_cfg["drop_height"] + 0.01

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{robot_dir}/usd_file/robot.usd",
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_height),
            joint_pos={
                ".*": 0.00
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators=actuators_without_knee if robot_idx < 15 else actuators_with_knee,
        prim_path=prim_path
    )
    globals()[cfg_name] = cfg

