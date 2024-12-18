python scripts/rsl_rl/run_distillation.py \
    --tasks Gendog10_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4 \
            Gendog100_gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2 \
            Gendog200_gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_0_8 \
            Gendog50_gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_4 \
    --model urma \
    --exp_name urma_10_100_200_50_randomized_additive_bs512_acc2 \
    --batch_size 512 \
    --lr 3e-4 \
    --num_workers 16 \
    --num_epochs 50 \
    --gradient_acc_steps 2
