import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class EfficientH5Dataset(Dataset):
    def __init__(self, h5_files, dataset_key):
        """
        Args:
            h5_files (list of str): List of paths to .h5 files.
            dataset_key (str): Key to access the dataset within the .h5 files.
        """
        self.h5_files = h5_files
        self.dataset_key = dataset_key
        self.index_map = []  # Map from global index to (file_idx, local_idx)
        self.file_lens = []  # Cache lengths of datasets in each file

        # Precompute index mapping
        for file_idx, file_path in enumerate(self.h5_files):
            with h5py.File(file_path, 'r') as f:
                length = f[self.dataset_key].shape[0] * f[self.dataset_key].shape[1]  # Treat first two dims as batch
                self.file_lens.append(length)
                self.index_map.extend([(file_idx, local_idx) for local_idx in range(length)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.index_map[idx]
        file_path = self.h5_files[file_idx]

        # Lazy load data from file
        with h5py.File(file_path, 'r', swmr=True) as f:  # Use SWMR for thread safety
            data = f[self.dataset_key]
            batch_idx, sample_idx = divmod(local_idx, data.shape[1])
            data = data[batch_idx, sample_idx]

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
    # Parameters
    dataset_key = "my_dataset"
    h5_files = [f"dummy_data_{i}.h5" for i in range(1)]
    batch_size = 256

    # Create dummy data
    create_dummy_h5_files(h5_files, dataset_key)

    # Initialize dataset and dataloader
    dataset = EfficientH5Dataset(h5_files, dataset_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=True)

    # Iterate through dataloader with tqdm
    for batch in tqdm(dataloader, desc="Processing batches"):
        pass
