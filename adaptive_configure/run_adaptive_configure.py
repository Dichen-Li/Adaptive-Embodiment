import os
import h5py
import json
import numpy as np
import torch
from adaptive_network import AdaptiveConfigureNet

def load_robot_data(name, obs_actions_path, configure_path):
    """Load observation, action, and configuration data for a robot"""
    # Load observations and actions
    # Load only first 128 timesteps for memory efficiency
    with h5py.File(obs_actions_path, "r") as data_file:
        inputs = np.array(data_file["one_policy_observation"][:, :128])
        targets = np.array(data_file["actions"][:, :128])
    
    # Load configuration
    with open(configure_path, 'r') as f:
        data_file = json.load(f)
        configure = np.array(data_file[name]['dynamic_joint_description'])
    
    return inputs, targets, configure

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = AdaptiveConfigureNet(
        obs_dim=268,  # From the data shape
        action_dim=12,  # From the data shape
        config_dim=18,  # From the data shape
        hidden_dim=256
    ).to(device)
    
    # Load saved model weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "saved_models/adaptive_configure_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully")

    # Data paths for Genhexapod1
    robot_data = [
        {
            'name': 'Genhexapod1',
            'obs_actions': '/home/dichen/Documents/dichen/Adaptive-Embodiment/project/version0/embodiment-scaling-law/logs/rsl_rl/Genhexapod1_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2/2024-12-18_01-17-38/h5py_record/obs_actions_00000.h5',
            'configure': '/home/dichen/Documents/dichen/Adaptive-Embodiment/project/version0/embodiment-scaling-law/exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v2/configure/Genhexapod0_307_averageEnv_policy_description.json'
        },
                {
            'name': 'Genhexapod2',
            'obs_actions': '/home/dichen/Documents/dichen/Adaptive-Embodiment/project/version0/embodiment-scaling-law/logs/rsl_rl/Genhexapod2_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/2024-12-18_10-38-06/h5py_record/obs_actions_00000.h5',
            'configure': '/home/dichen/Documents/dichen/Adaptive-Embodiment/project/version0/embodiment-scaling-law/exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v2/configure/Genhexapod0_307_averageEnv_policy_description.json'
        }
    ]
    
    robot = robot_data[0]
    # Load data
    print(f"Loading data for {robot['name']}...")
    obs, actions, config = load_robot_data(
        robot['name'],
        robot['obs_actions'],
        robot['configure']
    )
    
    # Process data in batches
    batch_size = 32
    outputs = []
    
    for i in range(0, len(obs), batch_size):
        # Get batch
        batch_obs = torch.FloatTensor(obs[i:i+batch_size]).to(device)
        batch_actions = torch.FloatTensor(actions[i:i+batch_size]).to(device)
        
        # Run inference
        with torch.no_grad():
            batch_output = model(batch_obs, batch_actions)
            outputs.append(batch_output.cpu())  # Move to CPU to save GPU memory
    
    # Combine all outputs
    output = torch.cat(outputs, dim=0)
    
    # Print shapes
    print("\nShape Information:")
    print(f"Input observation shape: {obs.shape}")
    print(f"Input actions shape: {actions.shape}")
    print(f"Output configuration shape: {output.shape}")
    print(f"\nOutput represents: [batch_size, num_joints(12), config_dim(18)]")
    
    # Print first output configuration for verification
    print("\nFirst output configuration (shape and values):")
    print(f"Shape: {output[0].shape}")
    print(f"Values:\n{output[0]}")

if __name__ == "__main__":
    main()
