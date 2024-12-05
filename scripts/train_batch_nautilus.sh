#!/bin/bash

# Check if tasks are provided
if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash train_batch.sh --tasks task1 task2 ... [additional arguments]"
  exit 1
fi

# Parse tasks and additional arguments
if [[ "$1" == "--tasks" ]]; then
  shift
  # Collect tasks until the first non-task flag
  while [[ "$#" -gt 0 && "$1" != --* ]]; do
    tasks+=("$1")
    shift
  done
else
  echo "Expected --tasks flag"
  exit 1
fi

# Remaining arguments are treated as additional keyword arguments
kwargs=("$@")

# Print the tasks to be executed and additional arguments
echo "Tasks to be executed: ${tasks[@]}"
echo "Additional arguments: ${kwargs[@]}"

# Define the log directory
log_dir="/bai-fast-vol/code/jobs-logs"
mkdir -p "$log_dir"  # Create the log directory if it doesn't exist

# Function to handle interrupt and exit
function stop_execution {
  echo "Training interrupted. Exiting."
  exit 1
}

# Trap SIGINT (Ctrl+C) and call stop_execution
trap stop_execution SIGINT

for task in "${tasks[@]}"; do
  # Generate a timestamp for the log file
  timestamp=$(date +"%Y%m%d_%H%M%S")
  log_file="${log_dir}/${task}_${timestamp}.log"

  # Construct the command to be executed
  cmd="/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task \"$task\" --headless ${kwargs[@]}"

  # Print the command being executed
  echo "Starting training for task: $task. Logging to $log_file"
  echo "Executing command: $cmd"

  # Execute the command with unbuffered output
  stdbuf -oL -eL /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task "$task" --headless "${kwargs[@]}" > "$log_file" 2>&1 || stop_execution

  echo "Completed training for task: $task. Log saved to $log_file"
done

