import os
import h5py
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class DatasetSaver:
    # List of attribute names to retrieve from the environment
    ENV_ATTRS = [
        "nr_dynamic_joint_observations",
        "single_dynamic_joint_observation_length",
        "dynamic_joint_observation_length",
        "dynamic_joint_description_size",
        "nr_dynamic_foot_observations",
        "single_dynamic_foot_observation_length",
        "dynamic_foot_observation_length",
        "dynamic_foot_description_size",
        "joint_positions_update_obs_idx",
        "joint_velocities_update_obs_idx",
        "joint_previous_actions_update_obs_idx",
        "foot_ground_contact_update_obs_idx",
        "foot_time_since_last_ground_contact_update_obs_idx",
        "trunk_linear_vel_update_obs_idx",
        "trunk_angular_vel_update_obs_idx",
        "goal_velocity_update_obs_idx",
        "projected_gravity_update_obs_idx",
        "height_update_obs_idx",
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

        # Save environment metadata
        self._save_environment_metadata(env)

    def _save_environment_metadata(self, env):
        """
        Save specific environment metadata as a YAML file.

        Args:
            env: Instance of the locomotion environment.
        """
        # Dynamically retrieve attribute values
        metadata = {attr: getattr(env.unwrapped, attr, None) for attr in self.ENV_ATTRS}

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



class LocomotionDatasetSingle:
    def __init__(self, folder_path):
        """
        Initialize the LocomotionDataset.

        Args:
            folder_path (str): Path to the folder containing HDF5 files and metadata.
        """
        self.folder_path = folder_path
        self.metadata = self._load_metadata()
        self.inputs = []
        self.targets = []

    def _load_metadata(self):
        """
        Load metadata from the YAML file in the dataset folder.

        Returns:
            dict: Metadata containing environment parameters.
        """
        metadata_path = os.path.join(self.folder_path, "metadata.yaml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        print(f"[INFO]: Loaded metadata from {metadata_path}")
        return metadata

    def _load_hdf5_files(self):
        """
        Load data from all HDF5 files in the folder, sorted numerically by index,
        and check for None/NaN values in the data.
        """
        hdf5_files = sorted(
            [f for f in os.listdir(self.folder_path) if f.endswith(".h5")],
            key=lambda x: int(x.split('_')[-1].split('.')[0])  # Extract the integer index
        )

        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in folder: {self.folder_path}")

        for file_name in hdf5_files:
            file_path = os.path.join(self.folder_path, file_name)
            print(f"[INFO]: Loading file {file_path}")
            
            with h5py.File(file_path, "r") as data_file:
                inputs = data_file["one_policy_observation"][:]
                targets = data_file["actions"][:]

                # Check for None or NaN values
                if np.any(inputs == None) or np.any(targets == None):  # Check for None
                    raise ValueError(f"None values found in file: {file_path}")
                if np.isnan(inputs).any() or np.isnan(targets).any():  # Check for NaN
                    raise ValueError(f"NaN values found in file: {file_path}")

                self.inputs.append(inputs)
                self.targets.append(targets)

        # Concatenate data from all files
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        print(f"[INFO]: Loaded data from {len(hdf5_files)} HDF5 files.")

    def get_data_loader(self, batch_size=8, shuffle=True):
        """
        Create a DataLoader for the dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: DataLoader object for the dataset.
        """
        if not self.inputs or not self.targets:
            self._load_hdf5_files()

        dataset = TensorDataset(
            torch.tensor(self.inputs, dtype=torch.float32),
            torch.tensor(self.targets, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dynamic_joint_params(self):
        """
        Retrieve dynamic joint parameters from the metadata.

        Returns:
            dict: Dynamic joint parameters.
        """
        return {
            "nr_dynamic_joint_observations": self.metadata["nr_dynamic_joint_observations"],
            "single_dynamic_joint_observation_length": self.metadata["single_dynamic_joint_observation_length"],
            "dynamic_joint_observation_length": self.metadata["dynamic_joint_observation_length"],
            "dynamic_joint_description_size": self.metadata["dynamic_joint_description_size"],
        }
    
    def get_dynamic_foot_params(self):
        """
        Retrieve dynamic foot parameters from the metadata.

        Returns:
            dict: Dynamic foot parameters.
        """
        return {
            "nr_dynamic_foot_observations": self.metadata["nr_dynamic_foot_observations"],
            "single_dynamic_foot_observation_length": self.metadata["single_dynamic_foot_observation_length"],
            "dynamic_foot_observation_length": self.metadata["dynamic_foot_observation_length"],
            "dynamic_foot_description_size": self.metadata["dynamic_foot_description_size"],
        }


import os
import yaml
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class LocomotionDataset:
    def __init__(self, folder_paths):
        """
        Initialize the LocomotionDataset.

        Args:
            folder_paths (list): List of paths to folders containing HDF5 files and metadata.
        """
        self.folder_paths = folder_paths
        self.metadata_list = [self._load_metadata(folder_path) for folder_path in folder_paths]
        self.inputs = []
        self.targets = []
        self.metadata_indices = []

        self._load_all_data()

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
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        print(f"[INFO]: Loaded metadata from {metadata_path}")
        return metadata

    def _load_all_data(self):
        """
        Load data from all specified folders.
        """
        for idx, folder_path in enumerate(self.folder_paths):
            hdf5_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".h5")],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )

            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in folder: {folder_path}")

            for file_name in hdf5_files:
                file_path = os.path.join(folder_path, file_name)
                print(f"[INFO]: Loading file {file_path}")

                with h5py.File(file_path, "r") as data_file:
                    inputs = data_file["one_policy_observation"][:]
                    targets = data_file["actions"][:]

                    # Check for None or NaN values
                    if np.any(inputs == None) or np.any(targets == None):  # Check for None
                        raise ValueError(f"None values found in file: {file_path}")
                    if np.isnan(inputs).any() or np.isnan(targets).any():  # Check for NaN
                        raise ValueError(f"NaN values found in file: {file_path}")

                    self.inputs.append(inputs)
                    self.targets.append(targets)
                    self.metadata_indices.extend([idx] * len(inputs))

        # Concatenate data from all files
        import ipdb; ipdb.set_trace()
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        print(f"[INFO]: Loaded data from {len(self.folder_paths)} folders.")

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Get a transformed sample from the dataset at a specific index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Transformed inputs and targets for the sample.
        """
        state = torch.tensor(self.inputs[index], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        metadata = self.metadata_list[self.metadata_indices[index]]

        # Dynamic Joint Data Transformation
        dynamic_joint_observation_length = metadata["dynamic_joint_observation_length"]
        nr_dynamic_joint_observations = metadata["nr_dynamic_joint_observations"]
        single_dynamic_joint_observation_length = metadata["single_dynamic_joint_observation_length"]
        dynamic_joint_description_size = metadata["dynamic_joint_description_size"]

        dynamic_joint_combined_state = state[:dynamic_joint_observation_length].view(
            (-1, nr_dynamic_joint_observations, single_dynamic_joint_observation_length)
        )
        dynamic_joint_description = dynamic_joint_combined_state[:, :, :dynamic_joint_description_size]
        dynamic_joint_state = dynamic_joint_combined_state[:, :, dynamic_joint_description_size:]

        # Dynamic Foot Data Transformation
        dynamic_foot_observation_length = metadata["dynamic_foot_observation_length"]
        nr_dynamic_foot_observations = metadata["nr_dynamic_foot_observations"]
        single_dynamic_foot_observation_length = metadata["single_dynamic_foot_observation_length"]
        dynamic_foot_description_size = metadata["dynamic_foot_description_size"]

        dynamic_foot_combined_state = state[dynamic_joint_observation_length:dynamic_joint_observation_length + dynamic_foot_observation_length].view(
            (-1, nr_dynamic_foot_observations, single_dynamic_foot_observation_length)
        )
        dynamic_foot_description = dynamic_foot_combined_state[:, :, :dynamic_foot_description_size]
        dynamic_foot_state = dynamic_foot_combined_state[:, :, dynamic_foot_description_size:]

        # General Policy State Transformation
        general_policy_state = torch.cat([state[-17:-8], state[-7:]], dim=0)

        import ipdb; ipdb.set_trace()

        # sizes torch.Size([320, 12, 18]), torch.Size([320, 12, 3]), torch.Size([320, 4, 10]), 
        # torch.Size([320, 4, 2]), torch.Size([16, 320]), torch.Size([4096, 12])

        # Return transformed inputs and target
        return (dynamic_joint_description, dynamic_joint_state,
                dynamic_foot_description, dynamic_foot_state,
                general_policy_state, target)

    def get_data_loader(self, batch_size=8, shuffle=True):
        """
        Create a DataLoader for the dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: DataLoader object for the dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
