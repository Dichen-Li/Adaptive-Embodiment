#!/bin/bash

tasks=("Gendog266" "Gendog252" "Gendog72" "Gendog71" "Gendog207" "Gendog104" "Gendog128" "Gendog272" "Gendog215" "Gendog248" "Gendog229" "Gendog280" "Gendog278" "Gendog32" "Gendog55" "Gendog75" "Gendog0" "Gendog282" "Gendog63" "Gendog47" "Gendog241" "Gendog222" "Gendog286" "Gendog181" "Gendog183" "Gendog97" "Gendog250" "Gendog20" "Gendog7" "Gendog258" "Gendog111" "Gendog41" "Gendog132" "Gendog113" "Gendog31" "Gendog290" "Gendog124" "Gendog46" "Gendog170" "Gendog48" "Gendog204" "Gendog260" "Gendog298" "Gendog226" "Gendog261" "Gendog37" "Gendog144" "Gendog244")
checkpoint_path="log_dir/scaling_factor_0.1_v3/best_model.pt"
log_file="log_dir/scaling_factor_0.1_v3/eval_student_model_urma.json"

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
        --headless \
        >/dev/null 2>&1

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
