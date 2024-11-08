#!/bin/bash

# Check if tasks are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: bash train_batch.sh --tasks task1 task2 ..."
  exit 1
fi

# Parse tasks from command-line arguments
if [ "$1" == "--tasks" ]; then
  shift
  tasks=("$@")
else
  echo "Expected --tasks flag"
  exit 1
fi

# Print the tasks to be executed
echo "Tasks to be executed: ${tasks[@]}"

# Function to handle interrupt and exit
function stop_execution {
  echo "Training interrupted. Exiting."
  exit 1
}

# Trap SIGINT (Ctrl+C) and call stop_execution
trap stop_execution SIGINT

# Execute training for each specified task
for task in "${tasks[@]}"; do
  echo "Starting training for task: $task"
  "${ISAAC_LAB_PATH}/isaaclab.sh" -p scripts/rsl_rl/train.py --task "$task" --headless || stop_execution
  echo "Completed training for task: $task"
done
