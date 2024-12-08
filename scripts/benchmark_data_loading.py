import numpy as np
import h5py
import time
import os

# Parameters
N = 1000000  # Number of rows
K = 1000     # Number of columns
split_factor = 5  # Number of splits (number of files)

# Generate a large random NumPy array
array = np.random.rand(N, K)

# Create a function to clean up test files
def cleanup(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)

# Benchmark saving and loading in one file
print("Testing single file...")
start_time = time.time()
with h5py.File("single_file.h5", "w") as h5f:
    h5f.create_dataset("dataset", data=array)
save_time_single = time.time() - start_time

start_time = time.time()
with h5py.File("single_file.h5", "r") as h5f:
    loaded_array_single = h5f["dataset"][:]
load_time_single = time.time() - start_time

cleanup(["single_file.h5"])

# Benchmark saving and loading in split files
print("Testing split files...")
split_files = [f"split_file_{i}.h5" for i in range(split_factor)]
split_arrays = np.array_split(array, split_factor, axis=0)  # Split the array

# Save split files
start_time = time.time()
for i, split_array in enumerate(split_arrays):
    with h5py.File(split_files[i], "w") as h5f:
        h5f.create_dataset("dataset", data=split_array)
save_time_split = time.time() - start_time

# Load split files
start_time = time.time()
loaded_splits = []
for split_file in split_files:
    with h5py.File(split_file, "r") as h5f:
        loaded_splits.append(h5f["dataset"][:])
loaded_array_split = np.concatenate(loaded_splits, axis=0)  # Recombine the array
load_time_split = time.time() - start_time

cleanup(split_files)

# Print results
print(f"Single File Save Time: {save_time_single:.4f} s, Load Time: {load_time_single:.4f} s")
print(f"Split Files Save Time: {save_time_split:.4f} s, Load Time: {load_time_split:.4f} s")

# Comparative Analysis
save_time_difference = save_time_split - save_time_single
load_time_difference = load_time_split - load_time_single

print("\n--- Comparative Analysis ---")
print(f"Saving split files took {'more' if save_time_difference > 0 else 'less'} time "
      f"by {abs(save_time_difference):.4f} seconds compared to a single file.")
print(f"Loading split files took {'more' if load_time_difference > 0 else 'less'} time "
      f"by {abs(load_time_difference):.4f} seconds compared to a single file.")

# Verify correctness
assert np.array_equal(array, loaded_array_single), "Single file load failed"
assert np.array_equal(array, loaded_array_split), "Split file load failed"

# Draw Conclusions
if save_time_split > save_time_single:
    print("Conclusion: Saving to a single file is faster.")
else:
    print("Conclusion: Saving to split files is faster.")

if load_time_split > load_time_single:
    print("Conclusion: Loading from a single file is faster.")
else:
    print("Conclusion: Loading from split files is faster.")

print("\nNOTE: Splitting files can be beneficial in scenarios requiring parallel I/O or when "
      "individual file sizes exceed system constraints.")
