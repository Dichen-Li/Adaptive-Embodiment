python scripts/rsl_rl/run_distillation.py \
    --train_set Genhexapod1_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2 \
            Genhexapod10_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_lengthen_calf_0_4 \
            Gendog10_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4 \
            Gendog100_gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2 \
            Genhumanoid100_genhumanoid__KneeNum_l1_r1__ScaleJointLimit_l0_r1_1_2__Geo_lengthen_calf_0_4 \
    --test_set Genhumanoid10_genhumanoid__KneeNum_l0_r0__ScaleJointLimit_l0_r0_1_0__Geo_lengthen_calf_0_4 \
    --model urma \
    --exp_name 0_test_small_bs256_lr3e-4_adamw \
    --batch_size 256 \
    --lr 3e-4 \
    --num_workers 16 \
    --num_epochs 10 \
    --gradient_acc_steps 1 \
    --use_amp 0_factor 1
