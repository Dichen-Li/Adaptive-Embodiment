import numpy as np
import time


def benchmark_ram(size=500_000_000, repeats=5):
    """
    Benchmarks RAM speed by performing sequential and random read/write operations.

    Args:
        size (int): Size of the array to test (number of elements).
        repeats (int): Number of times to repeat the test.

    Returns:
        dict: Average times for sequential and random read/write operations.
    """
    # Allocate a large array
    arr = np.empty(size, dtype=np.float32)  # Use 4 bytes per element to save memory

    sequential_write_times = []
    sequential_read_times = []
    random_write_times = []
    random_read_times = []

    for _ in range(repeats):
        # Sequential Write Test
        start_time = time.time()
        arr[:] = np.arange(size, dtype=np.float32)  # Fill array sequentially
        sequential_write_times.append(time.time() - start_time)

        # Sequential Read Test
        start_time = time.time()
        _ = arr.sum()  # Sequential read by summing elements
        sequential_read_times.append(time.time() - start_time)

        # Random Write Test
        indices = np.random.randint(0, size, size)  # Generate random indices
        values = np.random.rand(size).astype(np.float32)
        start_time = time.time()
        arr[indices] = values  # Random writes
        random_write_times.append(time.time() - start_time)

        # Random Read Test
        start_time = time.time()
        _ = arr[indices].sum()  # Random reads
        random_read_times.append(time.time() - start_time)

    results = {
        "sequential_write": np.mean(sequential_write_times),
        "sequential_read": np.mean(sequential_read_times),
        "random_write": np.mean(random_write_times),
        "random_read": np.mean(random_read_times),
    }

    print("RAM Benchmark Results (in seconds):")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    return results


if __name__ == "__main__":
    # Example: Test RAM performance with 500 million elements (~2 GB for float32)
    benchmark_ram(size=500_000_00, repeats=5)
