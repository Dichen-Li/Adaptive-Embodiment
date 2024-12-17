from multiprocessing import Manager
from collections import OrderedDict
import time

# Set up the dictionaries
manager = Manager()
shared_dict = manager.dict({str(i): i for i in range(10000)})
local_dict = OrderedDict({str(i): i for i in range(10000)})

# Benchmark for OrderedDict
start_time = time.time()
for i in range(10000):
    _ = local_dict[str(i)]
print(f"OrderedDict access time: {time.time() - start_time:.6f} seconds")

# Benchmark for Manager.dict()
start_time = time.time()
for i in range(10000):
    _ = shared_dict[str(i)]
print(f"Manager.dict access time: {time.time() - start_time:.6f} seconds")
