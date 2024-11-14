# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL and record motion data."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pandas as pd
import os
import torch
import gymnasium as gym

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

import csv

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and record motion data.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import berkeley_humanoid.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


def main():
    """Play with RSL-RL agent and record motion data."""
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

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    checkpoint_interval = 100  # Adjust this value as needed
    print("[INFO] Starting simulation with data recording.")
    # Initialize CSV file for data recording
    csv_file_path = "motion_data.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Define header based on parsed data structure
        writer.writerow(["timestamp",
                         "position_x", "position_y", "position_z",
                         "orientation_x", "orientation_y", "orientation_z", "orientation_w",
                         "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                         "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                         *["joint_position_" + str(i) for i in range(1, 13)],
                         *["joint_velocity_" + str(i) for i in range(1, 13)],
                         *["sensor_" + str(i) for i in range(1, 12)],
                         "action"])
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            obs, _ = env.get_observations()

            position = obs[0:3]
            orientation = obs[3:7]
            linear_velocity = obs[7:10]
            angular_velocity = obs[10:13]
            joint_positions = obs[13:25]
            joint_velocities = obs[25:37]
            sensors = obs[37:48]
            timestamp = timestep * 1  # Ensure `env.dt` is defined for timestep duration

            with open(csv_file_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    timestamp,
                    *position,
                    *orientation,
                    *linear_velocity,
                    *angular_velocity,
                    *joint_positions,
                    *joint_velocities,
                    *sensors,
                    actions.cpu().numpy().tolist()  # Convert actions to list if it's a tensor
                ])
            
            # Env stepping
            obs, _, _, _ = env.step(actions)

            
            timestep += 1


        # Exit the loop if video length is reached
        if args_cli.video and timestep == args_cli.video_length:
            break


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
