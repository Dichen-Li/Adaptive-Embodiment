import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate checkpoint in simulation")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs.")
parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

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

    log_dir = args_cli.log_dir
    resume_path = os.path.join(log_dir, "best_model.pt")

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

    # if we want to override this model with a customized model
    # from rsl_rl.modules import ActorCritic
    # metadata = train_dataset.metadata_list[0]
    # actor_critic = ActorCritic(316, 316, 12)
    # runner.alg.actor_critic = actor_critic.cuda()

    # Load the checkpoint
    print(f"[INFO] Loading model pt file from {resume_path}")
    checkpoint = torch.load(resume_path, map_location=agent_cfg.device)
    runner.alg.actor_critic.actor.load_state_dict(checkpoint["state_dict"])

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # define the device = 'cuda:0'
    model_device = agent_cfg.device
    # policy = runner.get_inference_policy(device=model_device)
    
    from utils import one_policy_observation_to_inputs
    # reset environment
    obs, observations = env.get_observations()
    one_policy_observation = observations["observations"]["one_policy"]

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # # Run model with a subset of the URMA observation
            # # Change one policy observation into actor obs input (trunk linear velocity changed)
            # (
            #     dynamic_joint_description,
            #     dynamic_joint_state,
            #     dynamic_foot_description,
            #     dynamic_foot_state,
            #     general_policy_state,
            # ) = one_policy_observation_to_inputs(one_policy_observation, env.unwrapped, model_device)

            # # Preprocessing the batch_inputs to adapt to the original actor_critic model
            # sliced_1 = general_policy_state[:, :12] # base_lin_vel, base_ang_vel, projected_gravity and target_x_y_yaw_rel from general_policy_state
            # swapped = torch.cat([sliced_1[:, :6], sliced_1[:, 9:12], sliced_1[:, 6:9]], dim=-1) # base_lin_vel, base_ang_vel, arget_x_y_yaw_rel and projected_gravity
            #
            # sliced_2 = dynamic_joint_state  # dynamic_joint_state (batch_num, 15, 3)
            # split = torch.split(sliced_2, 1, dim=-1)  # Split into 3 tensors along last axis
            # concatenated = torch.cat(split, dim=1).squeeze(-1)  # Concatenate along the second axis. This is joint_pos_rel, joint_vel_rel and action(previous)
            #
            # # The order of a actor model obs input is base_lin_vel, base_ang_vel, projected_gravity, target_x_y_yaw_rel, joint_pos_rel, joint_vel_rel and action(previous)
            # processed_batch_inputs = torch.cat([swapped, concatenated], dim=1)

            # # agent stepping
            # actions = runner.alg.actor_critic.actor(processed_batch_inputs)3

            # run model with full URMA observation
            one_input = torch.cat([
                component.flatten(1) for component in one_policy_observation_to_inputs(one_policy_observation, env.unwrapped, model_device)
            ], dim=1)
            actions = runner.alg.actor_critic.actor(one_input)

            # env stepping
            obs, _, _, extra = env.step(actions)
            one_policy_observation = extra["observations"]["one_policy"]

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
