import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a student model in simulation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps per environment")
parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
parser.add_argument("--model_is_actor", action="store_true", default=False,
                    help="Indicate if the supervised model is actor=True/one_policy=False.")
parser.add_argument("--ckpt_path", type=str, default=None, help="Store the specified policy file directory.")

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

from rsl_rl.modules import ActorCritic

# Ready for defining the policy package
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))

from rsl_rl.env import VecEnv
from silver_badger_torch.policy import get_policy
from utils import one_policy_observation_to_inputs


def find_newest_best_checkpoint(log_dir: str) -> tuple:
    """
    Find the newest 'best_model.pt' checkpoint in the latest experiment folder.

    Args:
        log_dir (str): Base directory where experiment folders are located.

    Returns:
        str: Full path to the 'best_model.pt' checkpoint.

    Raises:
        FileNotFoundError: If no experiment folder or checkpoint is found.
    """
    # Step 1: Find the newest experiment folder
    experiment_folders = [
        f for f in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, f))
    ]
    experiment_folders.sort(reverse=True)  # Sort by name (newest first based on timestamp naming)
    if not experiment_folders:
        raise FileNotFoundError("[ERROR] No experiment folders found in the log directory.")

    newest_folder = os.path.join(log_dir, experiment_folders[0])
    print(f"[INFO] Found newest experiment folder: {newest_folder}")

    # Step 2: Locate the best checkpoint
    best_checkpoint_path = os.path.join(newest_folder, "best_model.pt")
    if not os.path.isfile(best_checkpoint_path):
        raise FileNotFoundError(f"[ERROR] No 'best_model.pt' found in {newest_folder}")

    return (newest_folder, best_checkpoint_path)


class InferenceOnePolicyRunner:
    """A simple runner to handle inference using the one policy."""

    def __init__(self, env: VecEnv, device: str = "cpu", model_is_actor: bool = False):
        """
        Initialize the one policy runner.

        Args:
            device (str): The device for computation ('cpu' or 'cuda').
        """
        if model_is_actor:
            from supervised_actor.policy import get_policy
            policy = get_policy(env.unwrapped.nr_dynamic_joint_observations, device)
        else:
            from silver_badger_torch.policy import get_policy
            policy = get_policy(device)

        self.policy = policy
        self.device = device
        self.env = env

    def get_inference_policy(self, device: str = 'cpu'):
        """
        Prepare and return the inference-ready policy network.

        Returns:
            nn.Module: The policy network set to evaluation mode and moved to the correct device.
        """
        self.device = device
        self.policy.eval()  # Ensure evaluation mode
        self.policy.to(self.device)  # Ensure the policy is on the correct device
        print("[INFO] Inference policy is ready.")
        return self.policy

    def load(self, best_checkpoint_path: str, optimizer=None):
        """
        Load the policy network from the specified checkpoint path.

        Args:
            best_checkpoint_path (str): Full path to the 'best_model.pt' checkpoint.
            optimizer (torch.optim.Optimizer, optional): Optimizer to restore, if provided.

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        if self.policy is None:
            self.policy = get_policy(self.device)  # Initialize policy if not already initialized
        self.policy.load_state_dict(checkpoint["state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        print(f"[INFO] Checkpoint loaded successfully from {best_checkpoint_path}")

        return checkpoint["epoch"]

    def infer(self, one_policy_observation: torch.Tensor):
        """
        Preprocess the input one_policy_observation and run inference with the policy network.

        Args:
            one_policy_observation (torch.Tensor): The input one_policy_observation tensor.

        Returns:
            torch.Tensor: The action predicted by the policy.
        """
        (
            dynamic_joint_description,
            dynamic_joint_state,
            dynamic_foot_description,
            dynamic_foot_state,
            general_policy_state
        ) = one_policy_observation_to_inputs(one_policy_observation, self.env.unwrapped, self.device)
        # Feed processed data into the policy network
        with torch.no_grad():
            action = self.policy(
                dynamic_joint_description,
                dynamic_joint_state,
                dynamic_foot_description,
                dynamic_foot_state,
                general_policy_state
            )

        return action


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    policy_root_directory = args_cli.log_dir
    (policy_folder_directory, policy_file_path) = find_newest_best_checkpoint(policy_root_directory)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(policy_folder_directory, "one_policy_videos", args_cli.task),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Specify the policy file directory if needed (instead of loading the newest one)
    if args_cli.ckpt_path != parser.get_default('policy_file_directory'):
        policy_file_path = args_cli.ckpt_path

    # Create one policy runner for inference and load trained model parameters
    print(f"[INFO]: Loading model checkpoint from: {policy_file_path}")
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load previously trained model
    one_policy_runner = InferenceOnePolicyRunner(env, device=model_device, model_is_actor=args_cli.model_is_actor)
    one_policy_runner.load(policy_file_path)

    # Reset environment and start simulation
    obs, observations = env.get_observations()
    one_policy_observation = observations["observations"]["urma_obs"]
    curr_timestep = 0
    video_timestep = 0

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Agent stepping
            actions = one_policy_runner.infer(one_policy_observation)

            # Environment stepping
            obs, _, _, extra = env.step(actions)
            one_policy_observation = extra["observations"]["urma_obs"]

            curr_timestep += 1

            if curr_timestep > args_cli.steps:
                break
            if args_cli.video:
                video_timestep += 1
                # Exit the play loop after recording one video
                if video_timestep > args_cli.video_length:
                    break
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
