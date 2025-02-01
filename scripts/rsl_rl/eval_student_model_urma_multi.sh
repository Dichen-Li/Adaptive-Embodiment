#!/bin/bash

# test set for Gendog
tasks=("Go2-Direct-v0" "Gendog12")
#tasks=("Gendog12")

#tasks=("Genhexapod197" "Genhexapod215" "Genhexapod20" "Genhexapod132" "Genhexapod261" "Genhexapod248" "Genhexapod207" "Genhexapod155" "Genhexapod244" "Genhexapod183" "Genhexapod298" "Genhexapod111" "Genhexapod258" "Genhexapod71" "Genhexapod144" "Genhexapod48" "Genhexapod316" "Genhexapod128" "Genhexapod272" "Genhexapod308" "Genhexapod75" "Genhexapod158" "Genhexapod50" "Genhexapod37" "Genhexapod169" "Genhexapod241" "Genhexapod286" "Genhexapod51" "Genhexapod181" "Genhexapod222" "Genhexapod161" "Genhexapod312" "Genhexapod327" "Genhexapod104" "Genhexapod282" "Genhexapod226" "Genhexapod266" "Genhexapod133" "Genhexapod31" "Genhexapod280" "Genhexapod7" "Genhexapod47" "Genhexapod204" "Genhexapod320" "Genhexapod0" "Genhexapod313" "Genhexapod252" "Genhexapod170" "Genhexapod124" "Genhexapod166" "Genhexapod32" "Genhexapod97" "Genhexapod290" "Genhexapod113" "Genhexapod122" "Genhexapod72" "Genhexapod278" "Genhexapod229" "Genhexapod46" "Genhexapod41" "Genhexapod163" "Genhexapod260" "Genhexapod250" "Genhexapod55" "Genhexapod154" "Genhexapod149" "Genhexapod63" "Genhexapod12")

checkpoint_path="log_dir/urma_3robot/urma_3robot_10epoch_jan29.pt"
log_file="log_dir/urma_3robot/eval_student_model_urma_detailed.json"
#checkpoint_path="log_dir/scaling_factor_0.5_v3_modelscale3_attempt2_bs256_acc1_clipv5.0_configv2_scratch_e5/best_model.pt"
#log_file="log_dir/scaling_factor_0.5_v3_modelscale3_attempt2_bs256_acc1_clipv5.0_configv2_scratch_e5/eval_student_model_urma.json"

trap "echo 'Process interrupted. Exiting...'; exit 1" SIGINT

total_time=0

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    num_done=$((i + 1))
    num_tasks=${#tasks[@]}

    echo -n "Evaluating embodiment $num_done out of $num_tasks: $task ... "

    start_time=$(date +%s)

    python scripts/rsl_rl/eval_student_model_urma.py \
        --task "$task" \
        --ckpt_path "$checkpoint_path" \
        --log_file "$log_file" \
        --headless
#        \
#        >/dev/null 2>&1

    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    total_time=$((total_time + runtime))

    echo -n "Done. "

    average_time=$((total_time / num_done))
    tasks_left=$((num_tasks - num_done))

    if [ $tasks_left -gt 0 ]; then
        estimate=$((average_time * tasks_left))
        current_time=$(date +%s)
        finish_time=$((current_time + estimate))
        echo "→ Expected completion: $(LC_ALL=en_US.UTF-8 date -d @"${finish_time}" +"%I:%M %p")"
    fi
done

echo "→ Evaluation complete!"
