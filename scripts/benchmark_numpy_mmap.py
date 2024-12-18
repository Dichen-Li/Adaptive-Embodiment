import numpy as np
import os
import time

# Benchmark parameters
SMALL_FILE_SIZE = 10 * 1024  # 10 KB (small file)
LARGE_FILE_SIZE = 100 * 1024 * 1024 * 10  # 100 MB (large file)
NUM_READS = 100000  # Number of random reads for benchmarking
NUM_TRIALS = 5  # Number of trials to compute the mean benchmark time


def create_memmap_file(filename, size):
    """Create a file with random bytes and memory map it."""
    with open(filename, "wb") as f:
        f.write(np.random.bytes(size))
    print(f"Created file '{filename}' of size {size / (1024 ** 2):.2f} MB")


def benchmark_memmap_read(filename, num_reads, num_trials):
    """Benchmark reading random indices from a memory-mapped file over multiple trials."""
    # Memory map the file
    memmap_array = np.memmap(filename, dtype='ubyte', mode='r')
    size = len(memmap_array)

    # Generate random indices to read
    random_indices = np.random.randint(0, size, size=num_reads)

    # Run the benchmark for N trials
    times = []
    for trial in range(num_trials):
        start_time = time.perf_counter()
        for idx in random_indices:
            _ = memmap_array[idx]  # Perform read
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        print(f"  Trial {trial + 1}: {times[-1]:.4f} seconds")

    # Cleanup the memmap
    del memmap_array
    return np.mean(times)


def main():
    small_file = "small_file.dat"
    large_file = "large_file.dat"

    # Create small and large files
    create_memmap_file(small_file, SMALL_FILE_SIZE)
    create_memmap_file(large_file, LARGE_FILE_SIZE)

    print("\nBenchmarking small file...")
    small_file_time = benchmark_memmap_read(small_file, NUM_READS, NUM_TRIALS)
    print(f"Mean time to read {NUM_READS} indices from small file: {small_file_time:.4f} seconds")

    print("\nBenchmarking large file...")
    large_file_time = benchmark_memmap_read(large_file, NUM_READS, NUM_TRIALS)
    print(f"Mean time to read {NUM_READS} indices from large file: {large_file_time:.4f} seconds")

    # Cleanup files
    os.remove(small_file)
    os.remove(large_file)
    print("\nBenchmark complete. Files cleaned up.")

    # Compare results
    print("\n--- Benchmark Results ---")
    print(f"Mean Small file read time: {small_file_time:.4f} seconds")
    print(f"Mean Large file read time: {large_file_time:.4f} seconds")
    print(f"Speed difference ratio (large/small): {large_file_time / small_file_time:.2f}")


if __name__ == "__main__":
    main()
