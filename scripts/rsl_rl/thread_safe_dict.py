import psutil
import os
import time
import numpy as np
from collections import OrderedDict
import threading


# ThreadSafeDict implementation
class ThreadSafeDict:
    def __init__(self, max_size=128):
        self.dict = OrderedDict()
        self.lock = threading.RLock()
        self.max_size = max_size

    def get(self, key):
        with self.lock:
            if key in self.dict:
                self.dict.move_to_end(key)  # Mark as recently used
                return self.dict[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.dict:
                self.dict.move_to_end(key)  # Update usage order
            self.dict[key] = value
            if len(self.dict) > self.max_size:
                self.dict.popitem(last=False)  # Remove oldest item

    def delete(self, key):
        with self.lock:
            if key in self.dict:
                del self.dict[key]

    def clear(self):
        with self.lock:
            self.dict.clear()


# Function to monitor memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


# Test the ThreadSafeDict with large NumPy arrays
def test_memory_leak():
    print("Starting memory leak test with large NumPy arrays...")
    max_size = 100  # Maximum number of items in cache
    cache = ThreadSafeDict(max_size=max_size)
    array_size = (1000, 10000)  # Large NumPy array size (1 million elements)

    initial_memory = memory_usage()
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")

    # Insert large NumPy arrays into the cache
    for i in range(1_000_000):
        large_array = np.random.rand(*array_size)  # Create a large NumPy array
        cache.put(i, large_array)  # Insert into cache

        if i % 100 == 0:  # Check memory usage periodically
            current_memory = memory_usage()
            print(f"Iteration {i}, Memory Usage: {current_memory:.2f} MB")

            # # Stop test if memory usage grows excessively (indicates a leak)
            # if current_memory - initial_memory > 500:  # Allowable growth of 500 MB
            #     print("Potential memory leak detected!")
            #     break

    final_memory = memory_usage()
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print("Memory leak test complete.")


if __name__ == "__main__":
    test_memory_leak()
