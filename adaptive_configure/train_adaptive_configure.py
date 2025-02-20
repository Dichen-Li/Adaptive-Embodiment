import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import h5py
from tqdm.auto import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__))+"/scripts/rsl_rl")
from utils import one_policy_observation_to_inputs
from adaptive_network import AdaptiveMLP, data_loader, train_model

class Metadata:
    def __init__(self, nr_dynamic_joint_observations, single_dynamic_joint_observation_length,
                 dynamic_joint_observation_length, dynamic_joint_description_size,
                 trunk_angular_vel_update_obs_idx, goal_velocity_update_obs_idx,
                 projected_gravity_update_obs_idx):
        self.nr_dynamic_joint_observations = nr_dynamic_joint_observations
        self.single_dynamic_joint_observation_length = single_dynamic_joint_observation_length
        self.dynamic_joint_observation_length = dynamic_joint_observation_length
        self.dynamic_joint_description_size = dynamic_joint_description_size
        self.trunk_angular_vel_update_obs_idx = trunk_angular_vel_update_obs_idx
        self.goal_velocity_update_obs_idx = goal_velocity_update_obs_idx
        self.projected_gravity_update_obs_idx = projected_gravity_update_obs_idx

def load_robot_data(h5py_path, device='cuda'):
    """
    Load urma observation data and parse data for one robot.
    Input: h5py_path
    Output: dynamic_joint_state, targets; dynamic_joint_description, general_policy_state
    """
    # Load h5py file
    h5py_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), h5py_path)
    max_parallel_envs_per_file = 4096
    with h5py.File(h5py_dir, "r") as data_file:
        inputs = np.array(data_file["one_policy_observation"][:, :max_parallel_envs_per_file])
        targets = np.array(data_file["actions"][:, :max_parallel_envs_per_file])

    # Define the metadata parameters for urma observation
    metadata = Metadata(
        nr_dynamic_joint_observations=12,
        single_dynamic_joint_observation_length=21,
        dynamic_joint_observation_length=252,
        dynamic_joint_description_size=18,
        trunk_angular_vel_update_obs_idx= [252, 253, 254],  # Example indices
        goal_velocity_update_obs_idx=[255, 256, 257],
        projected_gravity_update_obs_idx=[258, 259, 260]
    )

    # Parse urma observation
    inputs_tensor = torch.tensor(inputs)
    dynamic_joint_description = []
    dynamic_joint_state = []
    general_policy_state = []
    for i in range(inputs.shape[0]):
        (
            dynamic_joint_description_i,
            dynamic_joint_state_i,
            general_policy_state_i
        ) = one_policy_observation_to_inputs(inputs_tensor[i], metadata, device)
        dynamic_joint_description.append(dynamic_joint_description_i)
        dynamic_joint_state.append(dynamic_joint_state_i)
        general_policy_state.append(general_policy_state_i)

    dynamic_joint_description = torch.stack((dynamic_joint_description), dim=0)
    dynamic_joint_state = torch.stack((dynamic_joint_state), dim=0)
    general_policy_state = torch.stack((general_policy_state), dim=0)
    
    return dynamic_joint_state, targets, dynamic_joint_description, general_policy_state

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    h5py_file = 'logs/rsl_rl/Genhexapod2_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/2024-12-18_10-38-06/h5py_record/obs_actions_00000.h5'

    (
        dynamic_joint_state,
        targets,
        dynamic_joint_description,
        general_policy_state
    ) = load_robot_data(h5py_file, device)
    
    # Prepare data
    train_loader = data_loader(dynamic_joint_state, targets, dynamic_joint_description, general_policy_state, batch_size=16, device=device)
    # Initialize model
    model = AdaptiveMLP().to(device)
    print("Starting training...")
    # Train model
    model = train_model(model, train_loader, num_epochs=1000, device=device)

if __name__ == "__main__":
    main()