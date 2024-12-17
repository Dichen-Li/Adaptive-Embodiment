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
from rsl_rl.env.isaac_lab_vec_env import IsaacLabVecEnvWrapper

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = IsaacLabVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # define the device = 'cuda:0'
    model_device = agent_cfg.device

    import h5py
    import numpy as np
    # Define the file path
    h5py_record_file_path = os.path.join(log_dir, "h5py_record_simple_obs", "obs_actions.h5")
    if not os.path.exists(h5py_record_file_path):
        print(f"[INFO]: h5py_record_file_path not found")
        env.close()
        return
    # Load h5py input output data
    data = h5py.File(h5py_record_file_path, "a")
    inputs = data["observations"]
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
    runner.train_mode()
    # runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

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
    # save_path is ./logs/rsl_rl/task_name/pt_save_actor_critic/supervised_model.pt
    save_path = os.path.join(log_dir, "pt_save_actor_critic")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    save_path = os.path.join(save_path, "supervised_model.pt")
    from utils import ensure_unique_save_path
    save_path = ensure_unique_save_path(save_path)
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
