/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/run_distillation.py \
    --tasks Gendog0_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_0 \
            Gendog1_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_1_2 \
            Gendog2_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_scale_all_0_8 \
            Gendog3_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_6 \
            Gendog4_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_thigh_1_2 \
    --model urma \
    --exp_name urma_10_100_200_50_haxa1_randomized_additive_bs512_acc2 \
    --batch_size 512 \
    --lr 3e-4 \
    --num_workers 16 \
    --num_epochs 100 \
    --gradient_acc_steps 2
