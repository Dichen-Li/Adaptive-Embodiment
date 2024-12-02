# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# Import extensions to set up environment tasks
# import berkeley_humanoid.tasks  # noqa: F401


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np
from berkeley_humanoid.tasks.direct.locomotion.locomotion_env import LocomotionEnv

env = LocomotionEnv()
state = env
# Create policy and critic state masks
policy_general_state_mask = np.zeros(state.shape)
critic_general_state_mask = np.zeros(state.shape)
general_state_for_policy_names = [
    "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
    "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "p_gain", "d_gain", "action_scaling_factor",
    "mass",
    "robot_length", "robot_width", "robot_height"
]
general_state_for_critic_names = [
    "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
    "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
    "goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "height_0",
    "p_gain", "d_gain", "action_scaling_factor",
    "mass",
    "robot_length", "robot_width", "robot_height"
]
observation_name_to_ids = self.env.call("observation_name_to_id")
for env_id in range(self.nr_train_envs):
    for name in general_state_for_policy_names:
        self.policy_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1

    for name in general_state_for_critic_names:
        self.critic_general_state_mask[env_id, observation_name_to_ids[env_id][name]] = 1
self.policy_general_state_mask = np.array(self.policy_general_state_mask, dtype=bool)
self.critic_general_state_mask = np.array(self.critic_general_state_mask, dtype=bool)

dummy_dynamic_joint_combined_state = state[:self.train_env_ids[1], :self.dynamic_joint_observation_lengths[self.train_env_ids[0]]].reshape((-1, self.nr_dynamic_joint_observations[self.train_env_ids[0]], self.single_dynamic_joint_observation_length))
dummy_dynamic_joint_description = dummy_dynamic_joint_combined_state[:, :, :self.dynamic_joint_description_size]
dummy_dynamic_joint_state = dummy_dynamic_joint_combined_state[:, :, self.dynamic_joint_description_size:]

dummy_dynamic_foot_combined_state = state[:self.train_env_ids[1], self.dynamic_joint_observation_lengths[self.train_env_ids[0]]:self.dynamic_joint_observation_lengths[self.train_env_ids[0]] + self.dynamic_foot_observation_lengths[self.train_env_ids[0]]].reshape((-1, self.nr_dynamic_foot_observations[self.train_env_ids[0]], self.single_dynamic_foot_observation_length))
dummy_dynamic_foot_description = dummy_dynamic_foot_combined_state[:, :, :self.dynamic_foot_description_size]
dummy_dynamic_foot_state = dummy_dynamic_foot_combined_state[:, :, self.dynamic_foot_description_size:]

dummy_general_policy_state = state[:self.train_env_ids[1], self.policy_general_state_mask[self.train_env_ids[0]]]
dummy_general_critic_state = state[:self.train_env_ids[1], self.critic_general_state_mask[self.train_env_ids[0]]]

def main():

    # define the device = 'cuda:0'
    model_device = agent_cfg.device

    import h5py
    import numpy as np
    # Define the file path
    h5py_record_file_path = os.path.join(log_dir, "h5py_record", "obs_actions.h5")
    if not os.path.exists(h5py_record_file_path):
        print(f"[INFO]: h5py_record_file_path not found")
        env.close()
        return
    # Load h5py input output data
    data = h5py.File(h5py_record_file_path, "a")
    inputs = data["one_policy_observation"]
    targets = data["actions"]
    # Convert to PyTorch tensors
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32).to(model_device)
    targets = torch.tensor(np.array(targets), dtype=torch.float32).to(model_device)

    from torch.utils.data import DataLoader, TensorDataset
    # Create DataLoader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(runner.alg.actor_critic.parameters(), lr=0.001)

    # Store epoch losses in a list during training
    epoch_losses = []

    # Training loop
    num_epochs = 1000
    print("[INFO] Starting supervised training.")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(model_device)
            batch_targets = batch_targets.to(model_device)
            # Forward passd
            # Forward pass: process each input individually
            predictions = []
            for single_input in batch_inputs:
                single_prediction = runner.alg.actor_critic.actor(single_input)  # Add batch dimension
                predictions.append(single_prediction)

            predictions = torch.stack(predictions, dim=0)  # Combine into batch predictions
            loss = criterion(predictions, batch_targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(data_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.6f}")
    
    loss_log_flag = 0
    if loss_log_flag:
        import matplotlib.pyplot as plt
        # Plotting the loss curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
        plt.title('Training Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    # Save the new trained model
    save_path = os.path.join(log_dir, "h5py_record/supervised_model.pt")
    torch.save(runner.alg.actor_critic.state_dict(), save_path)
    print(f"[INFO] Supervised trained model saved to {save_path}.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
