from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from berkeley_humanoid.tasks.direct.environments.agents.rsl_rl_ppo_cfg import HumanoidPPORunnerCfg


@configclass
class Gendog10PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Gendog9PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Gendog8PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Gendog7PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Gendog6PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Gendog5PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Gendog4PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Gendog3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Gendog2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8"

@configclass
class Gendog0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0"

@configclass
class Gendog1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2"

@configclass
class Gendog21PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Gendog20PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Gendog19PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Gendog18PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Gendog17PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Gendog16PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Gendog15PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Gendog14PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Gendog13PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8"

@configclass
class Gendog11PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0"

@configclass
class Gendog12PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2"

@configclass
class Gendog87PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog86PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog85PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog84PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog83PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog82PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog81PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog80PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog79PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_0_8"

@configclass
class Gendog77PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_0"

@configclass
class Gendog78PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_2"

@configclass
class Gendog76PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog75PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog74PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog73PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog72PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog71PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog70PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog69PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog68PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_0_8"

@configclass
class Gendog66PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_0"

@configclass
class Gendog67PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_2"

@configclass
class Gendog43PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog42PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog41PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog40PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog39PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog38PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog37PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog36PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog35PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog33PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog34PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog32PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog31PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog30PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog29PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog28PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog27PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog26PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog25PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog24PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog22PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog23PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog109PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog108PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog107PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog106PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog105PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog104PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog103PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog102PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog101PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog99PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog100PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog98PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog97PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog96PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog95PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog94PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog93PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog92PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog91PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog90PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog88PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog89PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog65PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog64PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog63PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog62PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog61PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog60PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog59PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog58PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog57PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog55PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog56PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog54PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog53PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog52PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog51PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog50PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog49PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog48PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog47PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog46PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog44PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog45PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog120PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Gendog119PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Gendog118PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Gendog117PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Gendog116PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Gendog115PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Gendog114PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Gendog113PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Gendog112PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8"

@configclass
class Gendog110PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0"

@configclass
class Gendog111PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2"

@configclass
class Gendog186PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog185PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog184PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog183PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog182PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog181PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog180PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog179PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog178PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_0_8"

@configclass
class Gendog176PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_0"

@configclass
class Gendog177PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_2"

@configclass
class Gendog175PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog174PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog173PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog172PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog171PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog170PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog169PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog168PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog167PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_0_8"

@configclass
class Gendog165PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_0"

@configclass
class Gendog166PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_2"

@configclass
class Gendog142PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog141PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog140PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog139PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog138PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog137PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog136PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog135PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog134PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog132PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog133PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog131PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog130PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog129PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog128PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog127PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog126PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog125PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog124PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog123PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog121PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog122PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog208PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog207PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog206PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog205PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog204PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog203PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog202PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog201PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog200PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog198PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog199PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog197PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog196PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog195PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog194PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog193PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog192PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog191PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog190PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog189PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog187PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog188PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog164PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog163PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog162PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog161PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog160PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog159PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog158PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog157PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog156PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog154PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog155PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog153PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog152PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog151PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog150PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog149PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog148PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog147PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog146PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog145PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog143PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog144PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog219PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Gendog218PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Gendog217PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Gendog216PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Gendog215PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Gendog214PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Gendog213PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Gendog212PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Gendog211PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8"

@configclass
class Gendog209PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0"

@configclass
class Gendog210PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2"

@configclass
class Gendog285PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog284PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog283PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog282PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog281PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog280PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog279PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog278PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog277PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_0_8"

@configclass
class Gendog275PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_0"

@configclass
class Gendog276PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_0_8__Geo_scale_all_1_2"

@configclass
class Gendog274PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog273PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog272PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog271PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog270PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog269PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog268PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog267PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog266PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_0_8"

@configclass
class Gendog264PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_0"

@configclass
class Gendog265PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl0_fr0_rl1_rr1_1_2__Geo_scale_all_1_2"

@configclass
class Gendog241PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog240PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog239PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog238PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog237PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog236PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog235PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog234PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog233PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog231PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog232PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog230PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog229PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog228PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog227PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog226PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog225PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog224PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog223PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog222PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog220PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog221PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl0_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog307PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog306PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog305PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog304PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog303PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog302PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog301PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog300PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog299PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog297PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog298PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog296PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog295PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog294PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog293PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog292PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog291PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog290PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog289PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog288PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog286PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog287PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr0_rl1_rr0_1_2__Geo_scale_all_1_2"

@configclass
class Gendog263PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Gendog262PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Gendog261PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Gendog260PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Gendog259PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Gendog258PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Gendog257PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Gendog256PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Gendog255PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_0_8"

@configclass
class Gendog253PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_0"

@configclass
class Gendog254PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_0_8__Geo_scale_all_1_2"

@configclass
class Gendog252PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Gendog251PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Gendog250PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Gendog249PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Gendog248PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Gendog247PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Gendog246PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Gendog245PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Gendog244PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_0_8"

@configclass
class Gendog242PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_0"

@configclass
class Gendog243PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "gendog__KneeNum_fl3_fr3_rl3_rr3__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_scale_all_1_2"
