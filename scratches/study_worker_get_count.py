import torch
from torch.utils.data import Dataset, DataLoader
import os
import gc
import psutil

class TestDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.get_count = 0
        self.memory_usage = []  # To track memory usage as an artifact
        worker_info = torch.utils.data.get_worker_info()
        print(f"[INIT] Worker {worker_info.id if worker_info else 'Main Process'} initialized with get_count = {self.get_count}")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.get_count += 1

        # Create dummy variables to simulate memory usage
        dummy_data = [0] * (10**6)  # 1 MB of data
        self.memory_usage.append(dummy_data)

        # Print memory usage before and after garbage collection
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB

        if self.get_count % 5 == 0:
            gc.collect()
            memory_after = process.memory_info().rss / (1024 ** 2)  # Memory in MB
            print(f"[GC] Worker {torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'Main Process'}: Garbage collection invoked after {self.get_count} items.\n"
                  f"[Memory] Before GC: {memory_before:.2f} MB, After GC: {memory_after:.2f} MB")
        else:
            print(f"[Memory] Current memory usage: {memory_before:.2f} MB")

        worker_info = torch.utils.data.get_worker_info()
        print(f"[GET] Worker {worker_info.id if worker_info else 'Main Process'}: get_count = {self.get_count}")
        return index

# Number of samples in the dataset
data_size = 20

# Number of workers in DataLoader
num_workers = 2

# Create dataset and dataloader
data = TestDataset(size=data_size)
data_loader = DataLoader(data, batch_size=1, num_workers=num_workers)

# Iterate through the data loader
for _ in data_loader:
    pass
