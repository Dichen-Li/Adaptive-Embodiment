import os
import h5py
import json
import numpy as np
import torch
from adaptive_network import AdaptiveConfigureNet, train_model

def load_robot_data(xa_path, configure_path, robot_name):
    """Load observation, action, and configuration data for a robot"""
    max_parallel_envs_per_file = 4096
    
    # Load observations and actions
    with h5py.File(xa_path, "r") as data_file:
        inputs = np.array(data_file["one_policy_observation"][:, :max_parallel_envs_per_file])
        targets = np.array(data_file["actions"][:, :max_parallel_envs_per_file])
    
    # Load configuration
    with open(configure_path, 'r') as f:
        data_file = json.load(f)
        configure = np.array(data_file[robot_name]['dynamic_joint_description'])
    
    return inputs, targets, configure

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths for both robots
    robot_data = [
        {
            'name': 'Genhexapod2',
            'xa_path': os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                'logs/rsl_rl/Genhexapod2_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_0_8/2024-12-18_10-38-06/h5py_record/obs_actions_00000.h5'),
            'configure_path': os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                'exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v2/configure/Genhexapod0_307_averageEnv_policy_description.json')
        }
        # Add Genhexapod1 data paths when available
    ]
    
    # Load data for both robots
    train_obs = []
    train_actions = []
    train_configs = []
    
    for robot in robot_data:
        print(f"Loading data for {robot['name']}...")
        obs, actions, config = load_robot_data(
            robot['xa_path'],
            robot['configure_path'],
            robot['name']
        )
        train_obs.append(obs)
        train_actions.append(actions)
        train_configs.append(config)
        print(f"Loaded shapes - Obs: {obs.shape}, Actions: {actions.shape}, Config: {config.shape}")
    
    # Initialize model
    model = AdaptiveConfigureNet(
        obs_dim=268,  # From the data shape
        action_dim=12,  # From the data shape
        config_dim=18,  # From the data shape
        hidden_dim=256
    ).to(device)
    
    print("\nStarting training...")
    # Train model
    trained_model = train_model(
        model=model,
        train_obs=train_obs,
        train_actions=train_actions,
        train_configs=train_configs,
        num_epochs=100,
        batch_size=4,
        device=device
    )
    
    # Save trained model
    save_path = "adaptive_configure_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'obs_dim': 268,
            'action_dim': 12,
            'config_dim': 18,
            'hidden_dim': 256
        }
    }, save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
