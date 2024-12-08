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
from datetime import datetime
import tqdm
from torch.utils.tensorboard import SummaryWriter
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

parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run.")
parser.add_argument("--batch_size", type=int, default=4090*8, help="Batch size. 4096*16 takes 10G")
parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    help="Name of the experiment. Default is the current date and time.")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs.")
parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for torch data loder.")
parser.add_argument("--max_files_in_memory", type=int, default=1, help="Max number of data files in memory.")
parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set size.")

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


def main():
    """Train with RSL-RL agent."""
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
    env = RslRlVecEnvWrapper(env)

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

    from utils import get_most_recent_h5py_record_path, save_checkpoint, AverageMeter
    from dataset import LocomotionDataset

    task = args_cli.task
    # Dataset paths
    dataset_dirs = [get_most_recent_h5py_record_path("logs/rsl_rl", task)]

    # Training dataset
    train_dataset = LocomotionDataset(
        folder_paths=dataset_dirs,
        train_mode=True,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory
    )
    train_loader = train_dataset.get_data_loader(
        batch_size=args_cli.batch_size, shuffle=True, num_workers=args_cli.num_workers
    )

    # # Validation dataset
    # val_dataset = LocomotionDataset(
    #     folder_paths=dataset_dirs,
    #     train_mode=False,
    #     val_ratio=args_cli.val_ratio,
    #     max_files_in_memory=args_cli.max_files_in_memory
    # )
    # val_loader = val_dataset.get_data_loader(
    #     batch_size=args_cli.batch_size, shuffle=False, num_workers=args_cli.num_workers
    # )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(runner.alg.actor_critic.parameters(), lr=0.001)
    runner.train_mode()

    # save_path is ./logs/rsl_rl/task_name/pt_save_actor_critic
    save_path = os.path.join(log_dir, "pt_save_actor_critic")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    
    writer = SummaryWriter(log_dir=save_path)
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    best_val_loss = float("inf")

    # Training loop
    num_epochs = args_cli.num_epochs
    checkpoint_interval=args_cli.checkpoint_interval
    print("[INFO] Starting supervised training.")
    for epoch in range(num_epochs):
        runner.train_mode()
        train_loss_meter.reset()
        with tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_inputs, batch_targets in pbar:
                batch_inputs = [x.to(model_device) for x in batch_inputs]
                batch_targets = batch_targets.to(model_device)

                (
                    dynamic_joint_description,
                    dynamic_joint_state,
                    dynamic_foot_description,
                    dynamic_foot_state,
                    general_policy_state,
                ) = batch_inputs

                # Preprocessing the batch_inputs to adapt to the original actor_critic model
                sliced_1 = general_policy_state[:, :12] # base_lin_vel, base_ang_vel, projected_gravity and target_x_y_yaw_rel from general_policy_state
                swapped = torch.cat([sliced_1[:, :6], sliced_1[:, 9:12], sliced_1[:, 6:9]], dim=-1) # base_lin_vel, base_ang_vel, arget_x_y_yaw_rel and projected_gravity

                sliced_2 = dynamic_joint_state  # dynamic_joint_state (batch_num, 15, 3)
                split = torch.split(sliced_2, 1, dim=-1)  # Split into 3 tensors along last axis
                concatenated = torch.cat(split, dim=1).squeeze(-1)  # Concatenate along the second axis. This is joint_pos_rel, joint_vel_rel and action(previous)

                # The order of a actor model obs input is base_lin_vel, base_ang_vel, projected_gravity, target_x_y_yaw_rel, joint_pos_rel, joint_vel_rel and action(previous)
                processed_batch_inputs = torch.cat([swapped, concatenated], dim=1)

                # Forward passd
                batch_predictions = runner.alg.actor_critic.actor(processed_batch_inputs)  # Add batch dimension

                loss = criterion(batch_predictions, batch_targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update training loss tracker
                train_loss_meter.update(loss.item(), n=batch_targets.size(0))

                # Update progress bar with current loss
                pbar.set_postfix({"Loss": f"{train_loss_meter.avg:.3f}"})

        # Save checkpoints periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(runner.alg.actor_critic.actor, optimizer, epoch + 1, save_path)

        # Save the best model based on validation loss
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            save_checkpoint(runner.alg.actor_critic.actor, optimizer, epoch + 1, save_path, is_best=True)
    

    # release memory
    del train_dataset, train_loader

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
