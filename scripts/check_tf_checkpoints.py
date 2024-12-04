import os
import re

# Define the root directory, threshold, and index range
root_dir = "/bai-fast-vol/code/embodiment-scaling-law/logs/rsl_rl"
threshold = 3000  # Modify if needed
index_range = range(0, 308)  # Define the range of indices to check


# Function to extract the numeric value from "Gendog{i}"
def extract_index(folder_name):
    match = re.match(r"Gendog(\d+)", folder_name)
    return int(match.group(1)) if match else None


# Get all folder indices
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
folder_indices = {extract_index(f): f for f in folders if extract_index(f) is not None}

# Identify missing, incomplete, and complete logs
missing_logs = [i for i in index_range if i not in folder_indices]
incomplete_logs = []
complete_logs = []

for idx, folder in folder_indices.items():
    folder_path = os.path.join(root_dir, folder)
    pt_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
    valid_files = [f for f in pt_files if int(f.split("_")[1].split(".")[0]) > threshold]

    if not valid_files:  # No valid .pt files found
        incomplete_logs.append(folder)
    else:
        complete_logs.append(folder)

# Display the results
print("=== Missing Logs ===")
if missing_logs:
    print(f"Logs missing for indices: {missing_logs}")
else:
    print("No missing logs.")

print("\n=== Incomplete Logs ===")
if incomplete_logs:
    print(f"Incomplete logs (no .pt files after step {threshold}):")
    for log in incomplete_logs:
        print(f"- {log}")
else:
    print("No incomplete logs.")

print("\n=== Complete Logs ===")
if complete_logs:
    print(f"Complete logs (contain valid .pt files):")
    for log in complete_logs:
        print(f"- {log}")
else:
    print("No complete logs.")
