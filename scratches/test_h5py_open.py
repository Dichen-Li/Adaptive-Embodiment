import h5py
import numpy as np
import psutil
import os
import time


# Create a sample .h5 file for testing
def create_sample_h5(file_path):
    with h5py.File(file_path, "w") as f:
        f.create_dataset("one_policy_observation", data=np.random.rand(1000, 100))
        f.create_dataset("actions", data=np.random.randint(0, 10, size=(1000,)))


def check_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Memory in MB


def test_memory_leak(file_path, iterations):
    initial_memory = check_memory_usage()
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")

    for i in range(iterations):
        with h5py.File(file_path, "r") as data_file:
            inputs = np.array(data_file["one_policy_observation"][:])
            targets = np.array(data_file["actions"][:])

        # Optionally, add a short delay to simulate realistic conditions
        time.sleep(0.01)

        # Log memory usage every 10 iterations
        if (i + 1) % 10 == 0:
            current_memory = check_memory_usage()
            print(f"Iteration {i + 1}: Memory Usage: {current_memory:.2f} MB")

    final_memory = check_memory_usage()
    print(f"Final Memory Usage: {final_memory:.2f} MB")

    memory_difference = final_memory - initial_memory
    print(f"Memory Difference After {iterations} Iterations: {memory_difference:.2f} MB")


if __name__ == "__main__":
    file_path = "test_data.h5"

    # Create the sample file if it doesn't exist
    if not os.path.exists(file_path):
        create_sample_h5(file_path)

    # Run the memory test with the specified number of iterations
    test_memory_leak(file_path, iterations=100000)

    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)