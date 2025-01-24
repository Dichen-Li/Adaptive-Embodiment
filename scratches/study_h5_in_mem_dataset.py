import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class InMemoryDataset(Dataset):
    def __init__(self, h5_files, dataset_key):
        """
        Args:
            h5_files (list of str): List of paths to .h5 files.
            dataset_key (str): Key to access the dataset within the .h5 files.
        """
        self.data = []  # Store all data in memory

        # Load data into memory
        load_start = time.time()
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as f:
                self.data.append(f[dataset_key][:])
        self.data = np.concatenate(self.data, axis=0)  # Combine all files into one array
        self.num_samples, self.sample_length = self.data.shape[0], self.data.shape[1]
        print(f"Total data loaded: {self.data.shape}, Load time: {time.time() - load_start:.4f}s")

    def __len__(self):
        return self.num_samples * self.sample_length

    def __getitem__(self, idx):
        batch_idx, sample_idx = divmod(idx, self.sample_length)
        data = self.data[batch_idx, sample_idx]  # Retrieve the slice

        # Convert to tensor
        return torch.tensor(data, dtype=torch.float32)

# Create dummy .h5 files
def create_dummy_h5_files(file_paths, dataset_key):
    for file_path in file_paths:
        with h5py.File(file_path, 'w') as f:
            data = np.random.rand(100, 4096, 100).astype(np.float32)  # Shape (100, 4096, 100)
            f.create_dataset(dataset_key, data=data, chunks=True, compression="gzip")

# Main script
if __name__ == "__main__":
    import time

    # Parameters
    dataset_key = "my_dataset"
    h5_files = [f"dummy_data_{i}.h5" for i in range(5)]
    batch_size = 256

    # Uncomment to create dummy data
    # create_dummy_h5_files(h5_files, dataset_key)

    # Initialize dataset and dataloader
    dataset = InMemoryDataset(h5_files, dataset_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=True)

    # Iterate through dataloader with tqdm
    for batch in tqdm(dataloader, desc="Processing batches"):
        pass