import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args
from dataset import DatasetSaver  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=20, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps per environment")

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

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


import os
import torch

import torch
from rsl_rl.modules import ActorCritic

import torch

from silver_badger_torch.policy import get_policy

class SimpleOnePolicyRunner:
    """A simple runner to handle inference using the one policy."""

    def __init__(self, device: str ="cpu"):
        """
        Initialize the one policy runner.

        Args:
            device (str): The device for computation ('cpu' or 'cuda').
        """

        policy = get_policy(device)
        self.policy = policy
        self.device = device

    def get_inference_policy(self, device: str ='cpu'):
        """
        Prepare and return the inference-ready policy network.

        Returns:
            nn.Module: The policy network set to evaluation mode and moved to the correct device.
        """
        self.device  = device
        self.policy.eval()  # Ensure evaluation mode
        self.policy.to(self.device)  # Ensure the policy is on the correct device
        print("[INFO] Inference policy is ready.")
        return self.policy

    def load(self, checkpoint_path, optimizer=None):
        """
        Load the policy network and optimizer state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            optimizer (torch.optim.Optimizer, optional): The optimizer to restore, if provided.

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        print(f"[INFO] Checkpoint loaded from {checkpoint_path}")
        return checkpoint["epoch"]



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load previously trained model
    one_policy_runner = SimpleOnePolicyRunner(device=model_device)
    one_policy_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = one_policy_runner.get_inference_policy(device=model_device)

    # Reset environment and start simulation
    obs, observations = env.get_observations()
    one_policy_observation = observations["observations"]["one_policy"]
    curr_timestep = 0

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Agent stepping
            actions = policy(obs)

            # Reshape and save data
            dataset_manager.save_data(
                one_policy_observation=one_policy_observation.cpu().numpy(),
                actions=actions.cpu().numpy(),
            )
            
            # Environment stepping
            obs, _, _, extra = env.step(actions)
            one_policy_observation = extra["observations"]["one_policy"]

            curr_timestep += 1

            if curr_timestep > args_cli.steps:
                break

    # if args_cli.video:
    #     if h5py_timestep >= args_cli.video_length:
    #         break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
