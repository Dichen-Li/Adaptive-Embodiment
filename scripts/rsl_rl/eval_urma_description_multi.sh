#!/bin/bash
tasks=()
for i in $(seq 0 335); do
  tasks+=("Gendog$i")
done

checkpoint_path="log_dir/urma_3robot/urma_3robot_10epoch_jan29.pt"
description_log_file="log_dir/urma_3robot/policy_description.json"

log_file="log_dir/urma_3robot/eval_urma_description_log_$(date +"%Y%m%d_%H%M%S").log"

total_time=0

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    num_done=$((i + 1))
    num_tasks=${#tasks[@]}

    echo -n "Evaluating embodiment $num_done out of $num_tasks: $task ... " | tee -a "$log_file"

    start_time=$(date +%s)
    echo "Current working directory is $PWD" | tee -a "$log_file"
    
    python scripts/rsl_rl/eval_urma_description.py \
        --task "$task" \
        --ckpt_path "$checkpoint_path" \
        --description_log_file "$description_log_file" \
        --headless \
        >> "$log_file" 2>&1

    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    total_time=$((total_time + runtime))

    echo -n "Done. " | tee -a "$log_file"

    average_time=$((total_time / num_done))
    tasks_left=$((num_tasks - num_done))

    if [ $tasks_left -gt 0 ]; then
        estimate=$((average_time * tasks_left))
        current_time=$(date +%s)
        finish_time=$((current_time + estimate))
        echo "→ Expected completion: $(LC_ALL=en_US.UTF-8 date -d @"${finish_time}" +"%I:%M %p")" | tee -a "$log_file"
    fi
done

echo "→ Evaluation complete!" | tee -a "$log_file"
