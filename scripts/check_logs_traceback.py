import os
import re

# Define the root directory, log patterns, and folder index pattern
root_dir = "/bai-fast-vol/code/jobs-logs/prev"
folder_pattern = r"Genhumanoid(\d+)"  # Regex pattern for extracting indices from folder names
traceback_pattern = "Traceback"
error_pattern = "Error"

# Function to extract the numeric index from a folder name
def extract_index(folder_name, pattern):
    match = re.match(pattern, folder_name)
    return int(match.group(1)) if match else None

# Function to search for files with both "Traceback" and "Error"
def check_file_for_errors(file_path):
    has_traceback = False
    last_lines = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if traceback_pattern in line:
                    has_traceback = True
                if has_traceback and error_pattern in line:
                    last_lines = lines[-10:]  # Capture the last 5 lines
                    return True, last_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return False, last_lines

# Identify all log files matching the pattern and sort by index
log_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
log_file_indices = {
    extract_index(f, folder_pattern): f
    for f in log_files
    if extract_index(f, folder_pattern) is not None
}

# Sort log file indices
sorted_indices = sorted(log_file_indices.keys())

# Check each log file
error_logs = []
for idx in sorted_indices:
    log_file = log_file_indices[idx]
    log_file_path = os.path.join(root_dir, log_file)
    
    print(f"Scanning file: {log_file_path}")  # Debugging output
    has_error, last_lines = check_file_for_errors(log_file_path)
    if has_error:
        error_logs.append((idx, log_file, last_lines))

# Display the results
if error_logs:
    print("=== Logs with Errors ===")
    for idx, log_file, last_lines in error_logs:
        print(f"- File: {log_file} (Index: {idx})")
        print("  Last few lines:")
        for line in last_lines:
            print(f"    {line.strip()}")
        print()
else:
    print("No logs with errors found.")


