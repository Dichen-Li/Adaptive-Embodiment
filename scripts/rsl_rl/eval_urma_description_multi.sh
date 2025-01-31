#!/bin/bash

# Default values (can be overridden by command-line arguments)
checkpoint_path="log_dir/urma_3robot/urma_3robot_10epoch_jan29.pt"
description_log_file="log_dir/urma_3robot/policy_description.json"
log_file="log_dir/urma_3robot/eval_urma_description_log_$(date +"%Y%m%d_%H%M%S").log"
start_index=0  # Default start index
end_index=307  # Default end index
robot_name="Gendog"  # Default base name

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path)
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        checkpoint_path="$2"
        shift 2
      else
        echo "Error: --checkpoint_path requires a value"
        exit 1
      fi
      ;;
    --description_log_file)
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        description_log_file="$2"
        shift 2
      else
        echo "Error: --description_log_file requires a value"
        exit 1
      fi
      ;;
    --log_file)
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        log_file="$2"
        shift 2
      else
        echo "Error: --log_file requires a value"
        exit 1
      fi
      ;;
    --start_index)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        start_index="$2"
        shift 2
      else
        echo "Error: --start_index requires a numeric value"
        exit 1
      fi
      ;;
    --end_index)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        end_index="$2"
        shift 2
      else
        echo "Error: --end_index requires a numeric value"
        exit 1
      fi
      ;;
    --robot_name)
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        robot_name="$2"
        shift 2
      else
        echo "Error: --robot_name requires a value"
        exit 1
      fi
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Ensure end_index is greater than start_index
if [[ "$end_index" -lt "$start_index" ]]; then
  echo "Error: --end_index ($end_index) must be greater than or equal to --start_index ($start_index)"
  exit 1
fi

# Define tasks dynamically based on provided indices and base name
tasks=()
for ((i=start_index; i<=end_index; i++)); do
  tasks+=("${robot_name}${i}")
done

total_time=0

# Run evaluation loop
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
