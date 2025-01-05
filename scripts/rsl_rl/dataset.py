import os
import random

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from itertools import chain

from thread_safe_dict import ThreadSafeDict
from utils import AverageMeter


class DatasetSaver:
    # List of attribute names to retrieve from the environment
    ENV_ATTRS = [
        "nr_dynamic_joint_observations",
        "single_dynamic_joint_observation_length",
        "dynamic_joint_observation_length",
        "dynamic_joint_description_size",
        "joint_positions_update_obs_idx",
        "joint_velocities_update_obs_idx",
        "joint_previous_actions_update_obs_idx",
        "trunk_angular_vel_update_obs_idx",
        "goal_velocity_update_obs_idx",
        "projected_gravity_update_obs_idx",
    ]

    def __init__(self, record_path, env, max_steps_per_file, buffer_size=1):
        """
        Initialize the DatasetSaver to handle HDF5 datasets.

        Args:
            record_path (str): Directory to store HDF5 files.
            max_steps_per_file (int): Maximum number of steps per HDF5 file.
            buffer_size (int): Number of HDF5 files to accumulate before saving.
            env: Instance of the locomotion environment.
        """
        self.record_path = record_path
        self.max_steps_per_file = max_steps_per_file
        self.buffer_size = buffer_size
        self.current_file_index = 0

        # Buffers for accumulating data
        self.obs_buffer = []
        self.actions_buffer = []

        # Ensure the record directory exists
        os.makedirs(self.record_path, exist_ok=True)
        print(f"Recording path set to: {self.record_path}")

        # Save environment metadata, including the number of samples and parallel environments
        self._save_environment_metadata(env)

    def _save_environment_metadata(self, env):
        """
        Save specific environment metadata as a YAML file.

        Args:
            env: Instance of the locomotion environment.
        """
        # Dynamically retrieve attribute values
        metadata = {attr: getattr(env.unwrapped, attr, None) for attr in self.ENV_ATTRS}

        # Add additional metadata for sample and environment counts
        metadata["steps_per_file"] = self.max_steps_per_file
        metadata["parallel_envs"] = env.num_envs  # Assuming `num_envs` is an attribute of the environment

        # Save metadata to a YAML file
        metadata_file_path = os.path.join(self.record_path, "metadata.yaml")
        with open(metadata_file_path, "w") as metadata_file:
            yaml.dump(metadata, metadata_file, default_flow_style=False)

        print(f"Environment metadata saved to: {metadata_file_path}")

    def _save_to_files(self):
        """
        Save the accumulated data in the buffers to multiple HDF5 files,
        checking for None/NaN values before saving.
        """
        num_files = len(self.obs_buffer) // self.max_steps_per_file

        for i in range(num_files):
            # Prepare file-specific data
            start_idx = i * self.max_steps_per_file
            end_idx = (i + 1) * self.max_steps_per_file
            obs_data = np.array(self.obs_buffer[start_idx:end_idx])
            action_data = np.array(self.actions_buffer[start_idx:end_idx])

            # Check for None or NaN values
            if np.any(obs_data == None) or np.any(action_data == None):  # Check for None
                raise ValueError(f"Found None values in observation or action data for file {i}")
            if np.isnan(obs_data).any() or np.isnan(action_data).any():  # Check for NaN
                raise ValueError(f"Found NaN values in observation or action data for file {i}")

            # Save to a new HDF5 file
            file_name = f"obs_actions_{self.current_file_index:05d}.h5"
            file_path = os.path.join(self.record_path, file_name)
            with h5py.File(file_path, "w") as file:
                file.create_dataset("one_policy_observation", data=obs_data, dtype="float32")
                file.create_dataset("actions", data=action_data, dtype="float32")

            print(f"[INFO]: Saved {len(obs_data)} steps to {file_path}")
            self.current_file_index += 1

        # Remove saved data from buffers
        self.obs_buffer = self.obs_buffer[num_files * self.max_steps_per_file:]
        self.actions_buffer = self.actions_buffer[num_files * self.max_steps_per_file:]

    def save_data(self, one_policy_observation, actions):
        """
        Accumulate data in buffers and save to multiple files when buffer is full.

        Args:
            one_policy_observation (np.ndarray): Observation data.
            actions (np.ndarray): Action data.
        """
        # Append data to buffers
        self.obs_buffer.append(one_policy_observation)
        self.actions_buffer.append(actions)

        # Check if we need to save files
        total_steps = len(self.obs_buffer)
        if total_steps >= self.buffer_size * self.max_steps_per_file:
            self._save_to_files()

    def close(self):
        """
        Save any remaining data in the buffers to files.
        """
        if self.obs_buffer or self.actions_buffer:
            print("[INFO]: Flushing remaining data to files.")
            self._save_to_files()

    def __del__(self):
        """
        Destructor to ensure proper cleanup.
        """
        self.close()

#
# class LocomotionDatasetSingle:
#     def __init__(self, folder_path):
#         """
#         Initialize the LocomotionDataset.
#
#         Args:
#             folder_path (str): Path to the folder containing HDF5 files and metadata.
#         """
#         self.folder_path = folder_path
#         self.metadata = self._load_metadata()
#         self.inputs = []
#         self.targets = []
#
#     def _load_metadata(self):
#         """
#         Load metadata from the YAML file in the dataset folder.
#
#         Returns:
#             dict: Metadata containing environment parameters.
#         """
#         metadata_path = os.path.join(self.folder_path, "metadata.yaml")
#         if not os.path.exists(metadata_path):
#             raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
#
#         with open(metadata_path, "r") as metadata_file:
#             metadata = yaml.safe_load(metadata_file)
#         print(f"[INFO]: Loaded metadata from {metadata_path}")
#         return metadata
#
#     def _load_hdf5_files(self):
#         """
#         Load data from all HDF5 files in the folder, sorted numerically by index,
#         and check for None/NaN values in the data.
#         """
#         hdf5_files = sorted(
#             [f for f in os.listdir(self.folder_path) if f.endswith(".h5")],
#             key=lambda x: int(x.split('_')[-1].split('.')[0])  # Extract the integer index
#         )
#
#         if not hdf5_files:
#             raise FileNotFoundError(f"No HDF5 files found in folder: {self.folder_path}")
#
#         for file_name in hdf5_files:
#             file_path = os.path.join(self.folder_path, file_name)
#             # print(f"[INFO]: Loading file {file_path}")
#
#             with h5py.File(file_path, "r") as data_file:
#                 inputs = data_file["one_policy_observation"][:]
#                 targets = data_file["actions"][:]
#
#                 # Check for None or NaN values
#                 if np.any(inputs == None) or np.any(targets == None):  # Check for None
#                     raise ValueError(f"None values found in file: {file_path}")
#                 if np.isnan(inputs).any() or np.isnan(targets).any():  # Check for NaN
#                     raise ValueError(f"NaN values found in file: {file_path}")
#
#                 self.inputs.append(inputs)
#                 self.targets.append(targets)
#
#         # Concatenate data from all files
#         self.inputs = np.concatenate(self.inputs, axis=0)
#         self.targets = np.concatenate(self.targets, axis=0)
#         print(f"[INFO]: Loaded data from {len(hdf5_files)} HDF5 files.")
#
#     def get_data_loader(self, batch_size=8, shuffle=True):
#         """
#         Create a DataLoader for the dataset.
#
#         Args:
#             batch_size (int): Batch size for the DataLoader.
#             shuffle (bool): Whether to shuffle the dataset.
#
#         Returns:
#             DataLoader: DataLoader object for the dataset.
#         """
#         if not self.inputs or not self.targets:
#             self._load_hdf5_files()
#
#         dataset = TensorDataset(
#             torch.tensor(self.inputs, dtype=torch.float32),
#             torch.tensor(self.targets, dtype=torch.float32),
#         )
#         return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#     def get_dynamic_joint_params(self):
#         """
#         Retrieve dynamic joint parameters from the metadata.
#
#         Returns:
#             dict: Dynamic joint parameters.
#         """
#         return {
#             "nr_dynamic_joint_observations": self.metadata["nr_dynamic_joint_observations"],
#             "single_dynamic_joint_observation_length": self.metadata["single_dynamic_joint_observation_length"],
#             "dynamic_joint_observation_length": self.metadata["dynamic_joint_observation_length"],
#             "dynamic_joint_description_size": self.metadata["dynamic_joint_description_size"],
#         }


class LocomotionDataset(Dataset):
    def __init__(self, folder_paths, max_files_in_memory, train_mode, val_ratio):
        """
        Initialize the LocomotionDataset.

        The dataset uses memoization for data loading, i.e., once a data file is loaded it will also be stored in cache
        Note that DataLoader spawns num_workers DataSet classes, each for one thread, so the cache will not
        be shared across multiple threads.
        TODO: Consider implementing a cache sharable across all threads to avoid duplicate IO and memory usage
        Edit: The TODO doesn't look feasible, because Manager().dict() can be shared across threads but the reading
        is 100x slower due to communication. Seems that a trade-off is necessary

        The dataset sampler also guarantees that the samples in one batch are all from one .h5 file, which
        may reduce randomness in data but increases reading speed and ease implementation.
        Thus, an implicit assumption is that one .h5 file contains at least one batch of data.

        Args:
            folder_paths (list): List of paths to dataset folders.
            max_files_in_memory (int): Maximum number of files to keep in memory.
        """
        self.folder_paths = folder_paths
        self.max_files_in_memory = max_files_in_memory
        self.metadata_list = [self._load_metadata(folder_path) for folder_path in folder_paths]
        self.train_mode = train_mode
        self.val_ratio = val_ratio
        self.file_indices = {}
        self.total_samples = 0

        # Thread-safe cache for loaded files
        # global global_cache    # we must use global reference, otherwise every thread will spawn its own cache
        self.cache = ThreadSafeDict(max_size=max_files_in_memory)

        # Map file indices and prepare dataset structure
        self._prepare_file_indices()

        # Compute hit rate for caching, for debugging purpose
        # self.cache_hit_count = AverageMeter()
        self.cache_query_time = AverageMeter()

        # record the number of times the data index is not in the worker's scope
        self.counter_not_in_scope = 0

        # Verbose output
        print(f"[INFO]: Initialized dataset with {len(self)} samples from {len(self.folder_paths)} folders. "
              f"\n\tGot {len(self.folder_idx_to_file_name)} robots: {list(self.folder_idx_to_file_name.values())}")

    def _load_metadata(self, folder_path):
        """
        Load metadata from the YAML file in a dataset folder.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            dict: Metadata containing environment parameters.
        """
        metadata_path = os.path.join(folder_path, "metadata.yaml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"[ERROR]: Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        return metadata

    def _prepare_file_indices(self):
        """
        Create a mapping of file indices to (folder_idx, file_idx).
        Segregates files into train/validation sets based on val_ratio.
        """
        self.folder_idx_to_file_name = {}
        for folder_idx, folder_path in enumerate(self.folder_paths):
            # example path: 'logs/rsl_rl/Gendog10_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4/2024-12-15_15-19-08/h5py_record'
            self.folder_idx_to_file_name[folder_idx] = folder_path.split("/")[-3]
            metadata = self.metadata_list[folder_idx]
            hdf5_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".h5")],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            total_files = len(hdf5_files)
            num_val_files = int(total_files * self.val_ratio)
            if num_val_files == 0:
                print(f"Num of val files {num_val_files} is 0, given total_files {total_files} "
                      f"and val_ratio {self.val_ratio}. Will use one file as validation set")
                num_val_files = 1

            # Choose files based on train/val mode
            if self.train_mode:
                selected_files = hdf5_files[num_val_files:]  # Remaining files for training
            else:
                selected_files = hdf5_files[:num_val_files]  # First files for validation
            # Process selected files
            for file_idx, file_name in enumerate(selected_files):
                key = (folder_idx, file_idx)
                steps_per_file = metadata["steps_per_file"]
                parallel_envs = metadata["parallel_envs"]
                self.file_indices[key] = (folder_path, file_name, steps_per_file, parallel_envs)
                self.total_samples += steps_per_file * parallel_envs

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return self.total_samples

    def _load_file(self, folder_idx, file_idx):
        """
        Load a specific HDF5 file into memory.

        Args:
            folder_idx (int): Index of the folder.
            file_idx (int): Index of the file within the folder.

        Returns:
            tuple: (inputs, targets) from the HDF5 file.
        """
        folder_path, file_name, steps_per_file, parallel_envs = self.file_indices[(folder_idx, file_idx)]
        file_path = os.path.join(folder_path, file_name)
        # print(f"[INFO]: Loading file {file_path}")

        with h5py.File(file_path, "r") as data_file:
            inputs = np.array(data_file["one_policy_observation"][:])
            targets = np.array(data_file["actions"][:])

        # Validate shapes
        if inputs.shape != (steps_per_file, parallel_envs, inputs.shape[-1]):
            raise ValueError(f"[ERROR]: Input shape mismatch in file {file_path}.")
        if targets.shape != (steps_per_file, parallel_envs, targets.shape[-1]):
            raise ValueError(f"[ERROR]: Target shape mismatch in file {file_path}.")

        return inputs, targets

    def _cache_file(self, folder_idx, file_idx):
        """
        Cache a specific file, loading it if not already cached.

        Args:
            folder_idx (int): Folder index.
            file_idx (int): File index within the folder.

        Returns:
            tuple: (inputs, targets) from the file.
        """
        cache_key = (folder_idx, file_idx)
        cached_file = self.cache.get(cache_key)
        if cached_file is None:
            # print(f"can't find cached file for {cache_key}")
            inputs, targets = self._load_file(folder_idx, file_idx)
            self.cache.put(cache_key, (inputs, targets))
            return inputs, targets
            # self.cache_hit_count.update(0, 1)
        # else:
            # self.cache_hit_count.update(1, 1 / self.batch_size)
            # print(f"[INFO]: Cached file found {cache_key}")
            # if self.cache_hit_count.count % 1000000:
            #     print(f"[INFO]: Cache hit rate: {self.cache_hit_count.avg}")
        return self.cache.get(cache_key)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (tuple): A four-tuple (folder_idx, file_idx, step, env) specifying the sample.

        Returns:
            tuple: Transformed input and target for the sample.
        """
        # # print(index)
        # # print(f"[DEBUG] Cache memory location: {id(self.cache)}, {index}")
        #
        # # Expecting index as (folder_idx, file_idx, step, env)
        folder_idx, file_idx, step, env = index

        # To make sure the worker does not get sample that should be processed by other workers
        # This shouldn't happen at all, in theory.
        worker_id = torch.utils.data.get_worker_info().id
        if (folder_idx, file_idx) not in self.worker_idx_to_folder_file_idx[worker_id]:
            # cache_keys = self.cache.keys()
            # folder_idx, file_idx = random.choice(list(cache_keys))
            # self.counter_not_in_scope += 1
            # if self.counter_not_in_scope % 1000 == 0:
            #     print(f"[ERROR]: Got index {index} but it's not from the expected files for the {self.counter_not_in_scope}th time. "
            #              f"Worker id: {worker_id}, files: {self.worker_idx_to_folder_file_idx[worker_id]}")
            raise ValueError(f"[ERROR]: Got index {index} but it's not from the expected files. "
                             f"Worker id: {worker_id}, files: {self.worker_idx_to_folder_file_idx[worker_id]}")

        # Load data from cache or file
        # import time
        # start_time = time.time()
        inputs, targets = self._cache_file(folder_idx, file_idx)
        # self.cache_query_time.update(time.time() - start_time, 1)
        # if self.cache_query_time.count % self.batch_size == 0:
        #     # print(f'cache query time for a batch on avg = {self.cache_query_time.avg * self.batch_size}')
        #     keys = self.cache.keys()
        #     file_indices = set([(x[0], x[1]) for x in keys])
        #     print(f'file indices: {file_indices}')

        # Retrieve specific sample within the file
        input_sample = inputs[step, env]
        target_sample = targets[step, env]

        # Get metadata for transformation
        metadata = self.metadata_list[folder_idx]

        # Transform the sample
        transformed_sample = self._transform_sample(input_sample, target_sample, metadata)
        # print([x.shape for x in transformed_sample])
        # import ipdb; ipdb.set_trace()

        # folder_idx, file_idx, step, env = index
        # transformed_sample = [torch.zeros(8, 18), torch.zeros(8, 3), torch.zeros(16), torch.zeros(8)]

        # Return the transformed components
        return transformed_sample[:-1], transformed_sample[-1], self.folder_idx_to_file_name[folder_idx]

    def _transform_sample(self, input_sample, target_sample, metadata):
        """
        Transform a single input and target sample into its components.

        Args:
            input_sample (np.ndarray): The input sample (shape: [D]).
            target_sample (np.ndarray): The target sample (shape: [T]).
            metadata (dict): Metadata for this sample.

        Returns:
            tuple: Transformed components.
        """
        state = torch.tensor(input_sample, dtype=torch.float32)  # Shape: (320,)
        target = torch.tensor(target_sample, dtype=torch.float32)  # Shape: (12,)

        # Dynamic Joint Data Transformation
        dynamic_joint_observation_length = metadata["dynamic_joint_observation_length"]
        nr_dynamic_joint_observations = metadata["nr_dynamic_joint_observations"]
        single_dynamic_joint_observation_length = metadata["single_dynamic_joint_observation_length"]
        dynamic_joint_description_size = metadata["dynamic_joint_description_size"]

        dynamic_joint_combined_state = state[..., :dynamic_joint_observation_length]  # Focus only on last dim
        dynamic_joint_combined_state = dynamic_joint_combined_state.view(
            nr_dynamic_joint_observations, single_dynamic_joint_observation_length
        )
        dynamic_joint_description = dynamic_joint_combined_state[..., :dynamic_joint_description_size]
        dynamic_joint_state = dynamic_joint_combined_state[..., dynamic_joint_description_size:]

        # General Policy State Transformation
        trunk_angular_vel_update_obs_idx = metadata["trunk_angular_vel_update_obs_idx"]
        goal_velocity_update_obs_idx = metadata["goal_velocity_update_obs_idx"]
        projected_gravity_update_obs_idx = metadata["projected_gravity_update_obs_idx"]
        general_policy_state = state[..., trunk_angular_vel_update_obs_idx+goal_velocity_update_obs_idx+projected_gravity_update_obs_idx]
        general_policy_state = torch.cat((general_policy_state, state[..., -7:]), dim=-1) # gains_and_action_scaling_factor; mass; robot_dimensions

        # Return transformed inputs and target
        return (
            dynamic_joint_description,  # Shape: (nr_dynamic_joint_observations, dynamic_joint_description_size)
            dynamic_joint_state,  # Shape: (nr_dynamic_joint_observations, remaining_length)
            general_policy_state,  # Shape: (<concatenated_dim>)
            target  # Shape: (12,)
        )

    def collate_fn(self, batch):
        """
        Collate function to combine samples into a batch.

        Args:
            batch (list): List of samples, where each sample is a 2-tuple:
                          (inputs, target).

        Returns:
            tuple: A 2-tuple where:
                - The first element is a tuple of batched inputs (stacked by component).
                - The second element is the batched target tensor.
        """
        # Split batch into inputs and targets
        inputs, targets, robot_names = zip(*batch)  # inputs: list of tuples, targets: list of tensors
        assert len(set(robot_names)) == 1, f"got different robot names in a batch: {set(robot_names)}"

        # Transpose the inputs to group by component
        inputs_by_component = zip(*inputs)  # Converts list of tuples into tuples of components

        # Stack each component of the inputs
        batched_inputs = tuple(torch.stack(components) for components in inputs_by_component)

        # Stack the targets
        batched_targets = torch.stack(targets)

        return batched_inputs, batched_targets, robot_names[0]

    def get_batch_indices(self, batch_size, shuffle=True, num_workers=1):
        """
        Generate all indices for the dataset, ensuring (1) batches are interleaved across workers,
        (2) each worker processes batches from a specific file in sequence, and (3) adjacent batches
        for one worker come from the same file.

        The expectation is:
        Worker 1: [batch from K1.h5] [batch from K1.h5] ... [batch from K3.h5] [batch from K3.h5]
        Worker 2: [batch from K2.h5] [batch from K2.h5] ... [batch from K4.h5] [batch from K4.h5]
        This way, workers won't process the same .h5 and every .h5 won't be read from disk more than once.

        Args:
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): Number of workers in the DataLoader.

        Returns:
            list: A list of indices, interleaved for workers, where each sublist contains indices for a batch.
        """
        self.batch_size = batch_size
        file_samples = {}  # Dictionary to hold batches per file

        # Collect batches for each file
        for (folder_idx, file_idx), (_, _, steps_per_file, parallel_envs) in self.file_indices.items():
            indices = [(folder_idx, file_idx, step, env)
                       for step in range(steps_per_file)
                       for env in range(parallel_envs)]
            if shuffle:
                np.random.shuffle(indices)  # Shuffle indices within the file

            # Split indices into batches and store in file_batches
            file_samples[(folder_idx, file_idx)] = [
                indices[i:i + batch_size] for i in range(0, len(indices), batch_size)
            ]

        # Shuffle the order of files
        file_keys = list(file_samples.keys())
        if shuffle:
            np.random.shuffle(file_keys)

        # Distribute batches across workers in an interleaved manner
        # samples_per_worker will be a list of lists, where the inner list contains lists of samples, with
        # each list from one file
        # also record the mapping between worker and file keys
        file_sample_lists_per_worker = [[] for _ in range(num_workers)]
        self.worker_idx_to_folder_file_idx = {worker_idx: set() for worker_idx in range(num_workers)}
        for i, key in enumerate(file_keys):
            file_sample_lists_per_worker[i % num_workers].append(file_samples[key])
            self.worker_idx_to_folder_file_idx[i % num_workers].add(key)

        # Duplicate samples so that every worker has the same number of samples
        # otherwise the workers that finish their job earlier will be assigned to join
        # other worker's job queue, which may create the condition where multiple workers
        # read the same file from the disk, making the system incredibly slow
        duplicates = 0
        max_files_per_worker = max(len(worker_batches) for worker_batches in file_sample_lists_per_worker)
        for worker_idx, worker_samples in enumerate(file_sample_lists_per_worker):
            while len(worker_samples) < max_files_per_worker:
                sampled_file_samples = random.choice(file_sample_lists_per_worker[worker_idx])  # sample a file sample list
                worker_samples.append(sampled_file_samples)
                duplicates += len(sampled_file_samples)     # record number of duplicates for logging
            file_sample_lists_per_worker[worker_idx] = worker_samples

        # flatten the inner 2-layer nested lists into one 1-layer list
        samples_per_worker = [list(chain(*worker_sample_lists)) for worker_sample_lists in file_sample_lists_per_worker]
        assert all(len(samples) == len(samples_per_worker[0]) for samples in samples_per_worker), \
            (f"the number of samples for workers differ: {[len(samples) for samples in samples_per_worker]}"
             f"This function assumes all files have the same number of samples. Is this true?")

        # Interleave batches from all workers to form the final sequence
        final_samples = []
        max_samples_per_worker = max(len(worker_samples) for worker_samples in samples_per_worker)
        for i in range(max_samples_per_worker):
            for worker_idx, samples in enumerate(samples_per_worker):
                final_samples.append(samples[i])

        print(f'[INFO]: duplicates due to resample: {duplicates} out of {len(final_samples)} samples '
              f'({duplicates/len(final_samples)*100:.2f}%)')

        return final_samples

    def get_data_loader(self, batch_size, shuffle=True, num_workers=16, **kwargs):
        """
        Create a DataLoader for the dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: The configured DataLoader instance.
        """

        if num_workers > len(self.file_indices):
            import warnings
            warnings.warn(f"num_workers={num_workers} should not exceed the number of files to read={len(self.file_indices)}, "
             f"as this would cause torch DataLoader to be extremely slow with our dataset implementation. "
             f"This is likely due to multiple threads reading the same file. "
             f"I will set num_workers to {min(num_workers, len(self.file_indices))}")
            num_workers = min(num_workers, len(self.file_indices))        # 2 is a safe number, tested

        return DataLoader(
            self,
            batch_sampler=self.get_batch_indices(batch_size, shuffle, num_workers),
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            **kwargs
        )


"""
The below dataset load all files into the memory, which is used as a reference for the fastest data loader. 
"""

# import os
# import h5py
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import yaml
#
#
# class LocomotionDataset(Dataset):
#     def __init__(self, folder_paths, train_mode, val_ratio, **kwargs):
#         """
#         Dataset class that loads all data into memory during initialization.
#
#         Args:
#             folder_paths (list): List of paths to dataset folders.
#             train_mode (bool): If True, loads training set; otherwise loads validation set.
#             val_ratio (float): Ratio of files to be used for validation.
#         """
#         self.folder_paths = folder_paths
#         self.train_mode = train_mode
#         self.val_ratio = val_ratio
#         self.metadata_list = [self._load_metadata(folder_path) for folder_path in folder_paths]
#         self.data = []
#
#         # Load all data into memory
#         self._load_all_files()
#
#         print(f"[INFO]: Loaded {len(self.data)} samples into memory.")
#
#     def _load_metadata(self, folder_path):
#         """
#         Load metadata from the YAML file in a dataset folder.
#         """
#         metadata_path = os.path.join(folder_path, "metadata.yaml")
#         if not os.path.exists(metadata_path):
#             raise FileNotFoundError(f"[ERROR]: Metadata file not found at {metadata_path}")
#
#         with open(metadata_path, "r") as metadata_file:
#             metadata = yaml.safe_load(metadata_file)
#         return metadata
#
#     def _load_all_files(self):
#         """
#         Load all .h5 files from the given folder paths into memory.
#         """
#         for folder_idx, folder_path in enumerate(self.folder_paths):
#             metadata = self.metadata_list[folder_idx]
#             hdf5_files = sorted(
#                 [f for f in os.listdir(folder_path) if f.endswith(".h5")],
#                 key=lambda x: int(x.split('_')[-1].split('.')[0])
#             )
#
#             # Split files into train/val sets
#             total_files = len(hdf5_files)
#             num_val_files = max(1, int(total_files * self.val_ratio))
#             selected_files = hdf5_files[num_val_files:] if self.train_mode else hdf5_files[:num_val_files]
#
#             # Load each file into memory
#             for file_name in selected_files:
#                 file_path = os.path.join(folder_path, file_name)
#                 with h5py.File(file_path, "r") as data_file:
#                     inputs = np.array(data_file["one_policy_observation"][:])
#                     targets = np.array(data_file["actions"][:])
#
#                 steps_per_file = metadata["steps_per_file"]
#                 parallel_envs = metadata["parallel_envs"]
#
#                 # Validate shapes
#                 if inputs.shape != (steps_per_file, parallel_envs, inputs.shape[-1]):
#                     raise ValueError(f"[ERROR]: Input shape mismatch in file {file_path}.")
#                 if targets.shape != (steps_per_file, parallel_envs, targets.shape[-1]):
#                     raise ValueError(f"[ERROR]: Target shape mismatch in file {file_path}.")
#
#                 # Flatten the data and store it in self.data
#                 for step in range(steps_per_file):
#                     for env in range(parallel_envs):
#                         self.data.append((inputs[step, env], targets[step, env]))
#
#     def __len__(self):
#         """
#         Get the total number of samples in the dataset.
#         """
#         return len(self.data)
#
#     def __getitem__(self, index):
#         """
#         Retrieve a single sample from the dataset.
#
#         Args:
#             index (int): Index of the sample.
#
#         Returns:
#             tuple: Transformed input and target.
#         """
#         input_sample, target_sample = self.data[index]
#         transformed_sample = self._transform_sample(input_sample, target_sample, self.metadata_list[0])
#         return transformed_sample[:-1], transformed_sample[-1]
#
#     def _transform_sample(self, input_sample, target_sample, metadata):
#         """
#         Transform a single input and target sample into its components.
#
#         Args:
#             input_sample (np.ndarray): The input sample (shape: [D]).
#             target_sample (np.ndarray): The target sample (shape: [T]).
#             metadata (dict): Metadata for this sample.
#
#         Returns:
#             tuple: Transformed components.
#         """
#         state = torch.tensor(input_sample, dtype=torch.float32)  # Shape: (320,)
#         target = torch.tensor(target_sample, dtype=torch.float32)  # Shape: (12,)
#
#         # Dynamic Joint Data Transformation
#         dynamic_joint_observation_length = metadata["dynamic_joint_observation_length"]
#         nr_dynamic_joint_observations = metadata["nr_dynamic_joint_observations"]
#         single_dynamic_joint_observation_length = metadata["single_dynamic_joint_observation_length"]
#         dynamic_joint_description_size = metadata["dynamic_joint_description_size"]
#
#         dynamic_joint_combined_state = state[..., :dynamic_joint_observation_length]  # Focus only on last dim
#         dynamic_joint_combined_state = dynamic_joint_combined_state.view(
#             nr_dynamic_joint_observations, single_dynamic_joint_observation_length
#         )
#         dynamic_joint_description = dynamic_joint_combined_state[..., :dynamic_joint_description_size]
#         dynamic_joint_state = dynamic_joint_combined_state[..., dynamic_joint_description_size:]
#
#         # Dynamic Foot Data Transformation
#         dynamic_foot_observation_length = metadata["dynamic_foot_observation_length"]
#         nr_dynamic_foot_observations = metadata["nr_dynamic_foot_observations"]
#         single_dynamic_foot_observation_length = metadata["single_dynamic_foot_observation_length"]
#         dynamic_foot_description_size = metadata["dynamic_foot_description_size"]
#
#         dynamic_foot_start = dynamic_joint_observation_length
#         dynamic_foot_end = dynamic_foot_start + dynamic_foot_observation_length
#         dynamic_foot_combined_state = state[..., dynamic_foot_start:dynamic_foot_end]  # Focus only on last dim
#         dynamic_foot_combined_state = dynamic_foot_combined_state.view(
#             nr_dynamic_foot_observations, single_dynamic_foot_observation_length
#         )
#         dynamic_foot_description = dynamic_foot_combined_state[..., :dynamic_foot_description_size]
#         dynamic_foot_state = dynamic_foot_combined_state[..., dynamic_foot_description_size:]
#
#         # General Policy State Transformation
#         general_policy_state = torch.cat([state[..., -17:-8], state[..., -7:]], dim=-1)
#
#         # Return transformed inputs and target
#         return (
#             dynamic_joint_description,  # Shape: (nr_dynamic_joint_observations, dynamic_joint_description_size)
#             dynamic_joint_state,  # Shape: (nr_dynamic_joint_observations, remaining_length)
#             dynamic_foot_description,  # Shape: (nr_dynamic_foot_observations, dynamic_foot_description_size)
#             dynamic_foot_state,  # Shape: (nr_dynamic_foot_observations, remaining_length)
#             general_policy_state,  # Shape: (<concatenated_dim>)
#             target  # Shape: (12,)
#         )
#
#     def get_data_loader(self, batch_size, shuffle=True, num_workers=0):
#         """
#         Create a DataLoader for the dataset.
#
#         Args:
#             batch_size (int): Batch size.
#             shuffle (bool): Whether to shuffle the data.
#             num_workers (int): Number of worker processes.
#
#         Returns:
#             DataLoader: Configured DataLoader.
#         """
#         return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
