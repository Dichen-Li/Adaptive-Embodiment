import os
import torch
import numpy as np

# Define the directory containing the data
log_dir = "log_dir/test"

# Get all files in the folder
files = os.listdir(log_dir)

# Filter files for _"new_ and _orig_
new_input_files = [f"batch_inputs_new_{i}.pt" for i in range(10)]
new_target_files = [f"batch_targets_new_{i}.pt" for i in range(10)]
orig_input_files = [f"batch_inputs_orig_{i}.pt" for i in range(10)]
orig_target_files = [f"batch_targets_orig_{i}.pt" for i in range(10)]

# Function to compare tensors
def compare_tensors(new_file, orig_file):
    if not os.path.exists(os.path.join(log_dir, orig_file)):
        print(f"Missing original file for: {new_file}")
        return

    # Load the data
    new_data = torch.load(os.path.join(log_dir, new_file))
    orig_data = torch.load(os.path.join(log_dir, orig_file))

    if isinstance(new_data, torch.Tensor):
        assert (new_data == orig_data).all()
    else:
        for x, y in zip(new_data, orig_data):
            assert torch.equal(x, y)

    # # Compare the data
    # try:
    #     if not torch.equal(new_data, orig_data):
    #         print(f"Mismatch found in {new_file} and {orig_file}")
    # except :
    #     import pdb; pdb.set_trace()

# Compare input files
print("Comparing input files...")
for new_file, orig_file in zip(new_input_files, orig_input_files):
    compare_tensors(new_file, orig_file)

# Compare target files
print("Comparing target files...")
for new_file, orig_file in zip(new_target_files, orig_target_files):
    compare_tensors(new_file, orig_file)

print("Comparison complete.")
