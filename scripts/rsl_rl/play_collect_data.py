import argparse
import os
import torch
import tqdm
import cli_args
from dataset import DatasetSaver  # isort: skip
import numpy as np
import gymnasium as gym
from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstration data for policy distillation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=20, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps per environment")

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

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
import ipdb; ipdb.set_trace()
from utils import RewardDictLogger


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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Initialize DatasetManager
    data_record_path = os.path.join(log_dir, "h5py_record")
    dataset_manager = DatasetSaver(
        record_path=data_record_path,
        max_steps_per_file=200,
        env=env
    )

    # Reset environment and start simulation
    obs, observations = env.get_observations()
    one_policy_observation = observations["observations"]["urma_obs"]
    # curr_timestep = 0
    actions_std = None

    from utils import RewardDictLogger
    reward_dict_logger = RewardDictLogger(args_cli.num_envs)

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            for curr_timestep in tqdm.tqdm(range(args_cli.steps)):
                # Agent stepping
                actions = policy(obs)

                # Reshape and save data
                dataset_manager.save_data(
                    one_policy_observation=one_policy_observation.cpu().numpy(),
                    actions=actions.cpu().numpy(),
                )

                # To expand the training data distribution, we should inject noise when rolling out teacher policy
                # Please don't use randomized actions as ground truth!

                # First, record action std
                if actions_std is None:
                    actions_std = actions.std(0)       # compute std for every joint
                    print(f'[INFO] Action std recorded: {actions_std}')

                # Apply strong randomization 1/20 of the time so there is still quite some clean data
                if np.random.randn() < 0.05:
                    # actions = actions * (torch.randn_like(actions) * actions_std + 1)
                    actions += torch.randn_like(actions) * actions_std.unsqueeze(0).repeat(args_cli.num_envs, 1) * 0.9

                # Stepping the environment
                obs, rewards, dones, extra = env.step(actions)
                one_policy_observation = extra["observations"]["urma_obs"]

                # log reward
                reward_dict_logger.update(env, rewards, dones)
                reward_dict_logger.print(curr_timestep, 'sum')

            # break the simulation loop
            break

    # if args_cli.video:
    #     if h5py_timestep >= args_cli.video_length:
    #         break

    # log reward statistics
    reward_dict_logger.write_to_yaml(os.path.join(data_record_path, "reward_dict.yaml"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
