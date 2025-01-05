import os
import re
from datetime import datetime

# Define the root directory, threshold, index range, and folder name pattern
root_dir = "/bai-fast-vol/code/embodiment-scaling-law/logs/rsl_rl"
threshold = 3000  # Modify this if needed
index_range = range(0, 308)  # Define the range of indices to check
folder_pattern = r"Gendog(\d+)"  # Regex pattern for extracting indices from folder names

# Function to extract the numeric index from a folder name
def extract_index(folder_name, pattern):
    match = re.match(pattern, folder_name)
    return int(match.group(1)) if match else None

# Get all folder indices
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
folder_indices = {
    extract_index(f, folder_pattern): f
    for f in folders
    if extract_index(f, folder_pattern) is not None
}

# Identify missing, incomplete, and complete logs
missing_logs = [i for i in index_range if i not in folder_indices]
incomplete_logs = []
complete_logs = []

for idx in sorted(folder_indices.keys()):  # Iterate in index order
    folder = folder_indices[idx]
    folder_path = os.path.join(root_dir, folder)
    
    # Get the latest timestamp folder
    timestamp_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not timestamp_folders:
        incomplete_logs.append((idx, folder))
        continue

    # Sort timestamp folders by their date
    latest_folder = max(
        timestamp_folders, 
        key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S")
    )
    latest_folder_path = os.path.join(folder_path, latest_folder)

    # Check for `.pt` files in the latest timestamped folder
    pt_files = [f for f in os.listdir(latest_folder_path) if f.endswith(".pt")]
    valid_files = [f for f in pt_files if int(f.split("_")[1].split(".")[0]) > threshold]
    
    if not valid_files:  # No valid .pt files found
        incomplete_logs.append((idx, folder))
    else:
        complete_logs.append((idx, folder))

# Display the results
print("=== Missing Logs ===")
if missing_logs:
    print(f"Missing log indices: {missing_logs}")
else:
    print("No missing logs.")

print("\n=== Incomplete Logs ===")
if incomplete_logs:
    print(f"Incomplete logs (no .pt files after step {threshold}):")
    for idx, log in incomplete_logs:
        print(f"- Index {idx}: {log}")
else:
    print("No incomplete logs.")

print("\n=== Complete Logs ===")
if complete_logs:
    print(f"Complete logs (contain valid .pt files):")
    for idx, log in complete_logs:
        print(f"- Index {idx}: {log}")
else:
    print("No complete logs.")

