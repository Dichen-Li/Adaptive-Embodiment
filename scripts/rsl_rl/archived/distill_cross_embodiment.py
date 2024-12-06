import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip
from dataset import LocomotionDatasetSingle
from datetime import datetime

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
# parser.add_argument(
#     "--tasks", 
#     nargs="+",  # Allows multiple values to be passed
#     type=str, 
#     default=None, 
#     help="List of tasks to process."
# )
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to run")
parser.add_argument(
    "--exp_name", 
    type=str, 
    default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),  # Default to current date and time
    help="Name of the experiment. Default is the current date and time."
)

# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)

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

# from rsl_rl.runners import OnPolicyRunner

import berkeley_humanoid.tasks  # noqa: F401

# from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# from omni.isaac.lab.utils.dict import print_dict
# from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import os
import torch
import numpy as np

def get_most_recent_h5py_record_path(base_path, task_name):
    """
    Finds the most recent folder for a given task and returns the path to its h5py_record subfolder.

    Args:
        base_path (str): The base directory where task folders are located.
        task_name (str): The name of the task to search for.

    Returns:
        str: Path to the h5py_record folder of the most recent task folder.
    """
    task_path = os.path.join(base_path, task_name)
    
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task folder '{task_name}' not found at {base_path}")

    # Find all subdirectories in the task folder
    subdirectories = [
        d for d in os.listdir(task_path) 
        if os.path.isdir(os.path.join(task_path, d)) and d.replace("_", "-").replace("-", "").isdigit()
    ]
    
    if not subdirectories:
        raise FileNotFoundError(f"No subfolders found for task '{task_name}' in {task_path}")

    # Sort directories by datetime in descending order
    subdirectories.sort(key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"), reverse=True)
    most_recent_folder = subdirectories[0]

    # Construct the path to h5py_record
    h5py_record_path = os.path.join(task_path, most_recent_folder, "h5py_record")
    
    if not os.path.exists(h5py_record_path):
        raise FileNotFoundError(f"h5py_record folder not found in '{os.path.join(task_path, most_recent_folder)}'")

    return h5py_record_path


def main():

    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # import ipdb; ipdb.set_trace()
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    locomotion_dataset = LocomotionDatasetSingle(folder_path=os.path.join(log_dir, "h5py_record"))

    data_loader = locomotion_dataset.get_data_loader(batch_size=args_cli.batch_size)

    dynamic_joint_params = locomotion_dataset.get_dynamic_joint_params()
    dynamic_foot_params = locomotion_dataset.get_dynamic_foot_params()

    print("[INFO]: Dynamic Joint Parameters:", dynamic_joint_params)
    print("[INFO]: Dynamic Foot Parameters:", dynamic_foot_params)

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
    for epoch in range(args_cli.num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(model_device)
            batch_targets = batch_targets.to(model_device)
            batch_predictions = []
            for single_input in batch_inputs:
                state: torch.tensor = single_input

                dynamic_joint_params = locomotion_dataset.get_dynamic_joint_params()
                nr_dynamic_joint_observations = dynamic_joint_params['nr_dynamic_joint_observations']
                single_dynamic_joint_observation_length = dynamic_joint_params['single_dynamic_joint_observation_length']
                dynamic_joint_observation_length = dynamic_joint_params['dynamic_joint_observation_length']
                dynamic_joint_description_size = dynamic_joint_params['dynamic_joint_description_size']

                dynamic_joint_combined_state = state[:, :dynamic_joint_observation_length].view((-1, nr_dynamic_joint_observations, single_dynamic_joint_observation_length))
                dynamic_joint_description = dynamic_joint_combined_state[:, :, :dynamic_joint_description_size]
                dynamic_joint_state = dynamic_joint_combined_state[:, :, dynamic_joint_description_size:]

                dynamic_foot_params = locomotion_dataset.get_dynamic_foot_params()
                nr_dynamic_foot_observations = dynamic_foot_params['nr_dynamic_foot_observations']
                single_dynamic_foot_observation_length = dynamic_foot_params['single_dynamic_foot_observation_length']
                dynamic_foot_observation_length = dynamic_foot_params['dynamic_foot_observation_length']
                dynamic_foot_description_size = dynamic_foot_params['dynamic_foot_description_size']

                dynamic_foot_combined_state = state[:, dynamic_joint_observation_length:dynamic_joint_observation_length + dynamic_foot_observation_length].view((-1, nr_dynamic_foot_observations, single_dynamic_foot_observation_length))
                dynamic_foot_description = dynamic_foot_combined_state[:, :, :dynamic_foot_description_size]
                dynamic_foot_state = dynamic_foot_combined_state[:, :, dynamic_foot_description_size:]

                # policy_general_state_mask = torch.arange(303, 320, device = 'cpu')
                # policy_general_state_mask = policy_general_state_mask[policy_general_state_mask != 312]

                # we just need a few last elements in the state as the "general_policy_state"
                general_policy_state = torch.cat([state[:, -17:-8], state[:, -7:]], dim=1)

                # Forward pass: process each input individually
                single_prediction = policy(dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_policy_state)
                print(dynamic_joint_description.shape, dynamic_joint_state.shape, dynamic_foot_description.shape, dynamic_foot_state.shape, general_policy_state.shape, single_prediction.shape)
                import ipdb; ipdb.set_trace()
                # humanoid: torch.Size([4096, 15, 18]) torch.Size([4096, 15, 3]) torch.Size([4096, 2, 10]) torch.Size([4096, 2, 2]) torch.Size([4096, 16]) torch.Size([4096, 15])
                # quadruped: torch.Size([4096, 12, 18]) torch.Size([4096, 12, 3]) torch.Size([4096, 4, 10]) torch.Size([4096, 4, 2]) torch.Size([4096, 16]) torch.Size([4096, 12])
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
