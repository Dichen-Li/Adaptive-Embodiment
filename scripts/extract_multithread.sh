#!/bin/bash

# Hyperparameter: Number of threads (parallel processes)
MAX_THREADS=16  # Change this to control the number of threads

# Specify the source directory and target directory
SOURCE_DIR="/mnt/hdd_0/expert_data/v2-tmu/logs"       # Directory containing .tar.xz files
TARGET_DIR="/mnt/hdd_0/expert_data/decompressed2/v2-tmu"  # Directory to extract files to

# Ensure the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory '$SOURCE_DIR' does not exist!"
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Create a function to extract a single file
extract_file() {
    local file=$1
    echo "Extracting $file to $TARGET_DIR ..."
    tar -xvJf "$file" -C "$TARGET_DIR"  # Extract into the target directory
    echo "Finished extracting $file!"
}

# Change to the source directory
cd "$SOURCE_DIR" || { echo "Failed to change to '$SOURCE_DIR' directory"; exit 1; }

# Initialize a counter for active threads
active_threads=0

# Iterate over all .tar.xz files
for file in *.tar.xz; do
    if [ -f "$file" ]; then
        # Start extraction in the background
        extract_file "$file" &
        active_threads=$((active_threads + 1))
        
        # Check if active threads have reached the maximum limit
        if [ "$active_threads" -ge "$MAX_THREADS" ]; then
            wait -n  # Wait for at least one process to finish
            active_threads=$((active_threads - 1))
        fi
    else
        echo "No .tar.xz files found in $SOURCE_DIR"
    fi
done

# Wait for any remaining background processes to complete
wait

echo "All files have been extracted to $TARGET_DIR!"

