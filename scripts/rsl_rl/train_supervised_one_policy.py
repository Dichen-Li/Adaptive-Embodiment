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

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np


def main():

    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Load training dataset
    import h5py
    import numpy as np
    # Define the file path
    h5py_record_file_path = os.path.join(log_dir, "h5py_record", "obs_actions.h5")
    if not os.path.exists(h5py_record_file_path):
        print(f"[INFO]: h5py_record_file_path not found")
        return
    # Load h5py input output data
    data = h5py.File(h5py_record_file_path, "a")
    inputs = data["one_policy_observation"]
    targets = data["actions"]
    import ipdb; ipdb.set_trace()

    # Create DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32))
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define model, optimizer and loss
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))
    import silver_badger_torch
    # define the device = 'cuda:0'
    model_device = 'cuda:0'
    policy = silver_badger_torch.policy.get_policy(model_device)
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    
    # Start training loop
    import time
    start_time = time.time()
    # Store epoch losses in a list during training
    epoch_losses = []
    # Training loop
    num_epochs = 100
    print("[INFO] Starting supervised training.")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(model_device)
            batch_targets = batch_targets.to(model_device)
            batch_predictions = []
            for single_input in batch_inputs:
                state: torch.tensor = single_input

                nr_dynamic_joint_observations = 12
                single_dynamic_joint_observation_length = 21
                dynamic_joint_observation_lengths = 12 * 21
                dynamic_joint_description_size = 18

                dynamic_joint_combined_state = state[:, :dynamic_joint_observation_lengths].view((-1, nr_dynamic_joint_observations, single_dynamic_joint_observation_length))
                dynamic_joint_description = dynamic_joint_combined_state[:, :, :dynamic_joint_description_size]
                dynamic_joint_state = dynamic_joint_combined_state[:, :, dynamic_joint_description_size:]

                nr_dynamic_foot_observations = 4
                single_dynamic_foot_observation_length = 12
                dynamic_foot_observation_lengths = 4 * 12
                dynamic_foot_description_size = 10

                dynamic_foot_combined_state = state[:, dynamic_joint_observation_lengths:dynamic_joint_observation_lengths + dynamic_foot_observation_lengths].view((-1, nr_dynamic_foot_observations, single_dynamic_foot_observation_length))
                dynamic_foot_description = dynamic_foot_combined_state[:, :, :dynamic_foot_description_size]
                dynamic_foot_state = dynamic_foot_combined_state[:, :, dynamic_foot_description_size:]

                policy_general_state_mask = torch.arange(303, 320, device = 'cpu')
                policy_general_state_mask = policy_general_state_mask[policy_general_state_mask != 312]

                general_policy_state = state[:, policy_general_state_mask]


                # Forward pass: process each input individually
                import ipdb; ipdb.set_trace()
                single_prediction = policy(dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_policy_state)
                batch_predictions.append(single_prediction)

            batch_predictions = torch.stack(batch_predictions)
            loss = criterion(batch_predictions, batch_targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(data_loader))
        elapsed_time = time.time()-start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.6f}, Elapsed time: {elapsed_time:.4f}")
    
    loss_log_flag = 1
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
    torch.save(policy.state_dict(), save_path)
    print(f"[INFO] Supervised trained model saved to {save_path}.")

if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
