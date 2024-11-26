from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from berkeley_humanoid.tasks.direct.environments.agents.rsl_rl_ppo_cfg import HumanoidPPORunnerCfg


@configclass
class Genhumanoid10PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid9PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid8PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid7PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid6PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid5PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid4PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid3PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid2PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8"

@configclass
class Genhumanoid0PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0"

@configclass
class Genhumanoid1PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2"

@configclass
class Genhumanoid14PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4"

@configclass
class Genhumanoid13PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8"

@configclass
class Genhumanoid12PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2"

@configclass
class Genhumanoid11PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6"

@configclass
class Genhumanoid25PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid24PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid23PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid22PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid21PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid20PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid19PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid18PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid17PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8"

@configclass
class Genhumanoid15PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0"

@configclass
class Genhumanoid16PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2"

@configclass
class Genhumanoid29PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4"

@configclass
class Genhumanoid28PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8"

@configclass
class Genhumanoid27PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2"

@configclass
class Genhumanoid26PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6"

@configclass
class Genhumanoid115PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid114PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid113PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid112PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid111PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid110PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid109PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid108PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid107PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid105PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid106PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid119PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid118PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid117PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid116PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid100PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid99PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid98PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid97PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid96PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid95PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid94PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid93PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid92PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid90PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid91PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid104PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid103PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid102PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid101PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid55PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid54PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid53PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid52PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid51PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid50PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid49PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid48PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid47PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid45PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid46PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid59PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid58PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid57PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid56PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid40PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid39PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid38PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid37PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid36PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid35PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid34PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid33PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid32PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid30PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid31PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid44PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid43PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid42PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid41PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid85PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid84PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid83PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid82PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid81PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid80PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid79PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid78PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid77PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid75PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid76PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid89PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid88PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid87PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid86PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid70PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid69PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid68PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid67PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid66PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid65PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid64PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid63PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid62PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid60PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid61PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid74PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid73PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid72PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid71PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid130PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid129PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid128PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid127PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid126PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid125PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid124PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid123PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid122PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8"

@configclass
class Genhumanoid120PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0"

@configclass
class Genhumanoid121PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2"

@configclass
class Genhumanoid134PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4"

@configclass
class Genhumanoid133PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8"

@configclass
class Genhumanoid132PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2"

@configclass
class Genhumanoid131PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6"

@configclass
class Genhumanoid220PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid219PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid218PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid217PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid216PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid215PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid214PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid213PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid212PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid210PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid211PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid224PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid223PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid222PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid221PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid205PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid204PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid203PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid202PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid201PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid200PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid199PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid198PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid197PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid195PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid196PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid209PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid208PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid207PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid206PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid160PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid159PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid158PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid157PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid156PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid155PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid154PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid153PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid152PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid150PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid151PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid164PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid163PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid162PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid161PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid145PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid144PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid143PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid142PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid141PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid140PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid139PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid138PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid137PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid135PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid136PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid149PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid148PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid147PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid146PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid190PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid189PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid188PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid187PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid186PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid185PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid184PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid183PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid182PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid180PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid181PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid194PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid193PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid192PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid191PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid175PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid174PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid173PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid172PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid171PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid170PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid169PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid168PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid167PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid165PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid166PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid179PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid178PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid177PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid176PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l2_r2__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid235PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid234PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid233PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid232PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid231PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid230PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid229PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid228PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid227PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_0_8"

@configclass
class Genhumanoid225PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_0"

@configclass
class Genhumanoid226PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_all_1_2"

@configclass
class Genhumanoid239PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_4"

@configclass
class Genhumanoid238PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_0_8"

@configclass
class Genhumanoid237PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_2"

@configclass
class Genhumanoid236PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r0_1_0__Geo_scale_torso_1_6"

@configclass
class Genhumanoid325PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid324PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid323PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid322PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid321PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid320PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid319PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid318PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid317PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid315PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid316PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid329PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid328PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid327PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid326PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid310PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid309PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid308PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid307PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid306PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid305PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid304PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid303PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid302PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid300PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid301PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid314PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid313PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid312PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid311PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l0_r1_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid265PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid264PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid263PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid262PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid261PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid260PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid259PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid258PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid257PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid255PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid256PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid269PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid268PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid267PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid266PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid250PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid249PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid248PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid247PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid246PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid245PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid244PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid243PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid242PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid240PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid241PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid254PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid253PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid252PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid251PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r0_1_2__Geo_scale_torso_1_6"

@configclass
class Genhumanoid295PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid294PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid293PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid292PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid291PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid290PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid289PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid288PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid287PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_0_8"

@configclass
class Genhumanoid285PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_0"

@configclass
class Genhumanoid286PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_all_1_2"

@configclass
class Genhumanoid299PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_4"

@configclass
class Genhumanoid298PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_0_8"

@configclass
class Genhumanoid297PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_2"

@configclass
class Genhumanoid296PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_0_8__Geo_scale_torso_1_6"

@configclass
class Genhumanoid280PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_4"

@configclass
class Genhumanoid279PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_0_8"

@configclass
class Genhumanoid278PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_2"

@configclass
class Genhumanoid277PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_calf_1_6"

@configclass
class Genhumanoid276PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_4"

@configclass
class Genhumanoid275PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_0_8"

@configclass
class Genhumanoid274PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_2"

@configclass
class Genhumanoid273PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_lengthen_thigh_1_6"

@configclass
class Genhumanoid272PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_0_8"

@configclass
class Genhumanoid270PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_0"

@configclass
class Genhumanoid271PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_all_1_2"

@configclass
class Genhumanoid284PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_4"

@configclass
class Genhumanoid283PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_0_8"

@configclass
class Genhumanoid282PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_2"

@configclass
class Genhumanoid281PPORunnerCfg(HumanoidPPORunnerCfg):
    experiment_name = "genhumanoid__KneeNum_l3_r3__ScaleJointLimit_l1_r1_1_2__Geo_scale_torso_1_6"
