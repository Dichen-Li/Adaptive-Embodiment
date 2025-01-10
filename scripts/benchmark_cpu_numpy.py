import numpy as np
import time


def test_cpu_speed(repeats=5, size=1000):
    """
    Tests CPU speed using matrix multiplication in NumPy.

    Args:
        repeats (int): Number of times to repeat the test.
        size (int): Size of the square matrices to multiply.

    Returns:
        float: Average computation time in seconds.
    """
    times = []

    for _ in range(repeats):
        # Generate two random matrices
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        # Time the matrix multiplication
        start_time = time.time()
        np.dot(A, B)
        elapsed_time = time.time() - start_time

        times.append(elapsed_time)

    average_time = np.mean(times)
    print(f"Average computation time over {repeats} runs: {average_time:.6f} seconds")
    return average_time


if __name__ == "__main__":
    # Example usage: Test CPU speed with 1000x1000 matrices, repeated 5 times
    test_cpu_speed(repeats=10, size=10000)
