import os
from datetime import datetime

import torch


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the statistics."""
        self.val = 0          # Current value
        self.avg = 0          # Average value
        self.sum = 0          # Sum of all values
        self.count = 0        # Number of updates

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int): The weight of this value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        """String representation of the average and current value."""
        return f"Val: {self.val:.4f}, Avg: {self.avg:.4f}"


class RewardDictLogger:
    def __init__(self):
        """
        Initializes the RewardDictLogger to track reward statistics.
        """
        self.reward_avg_meter_dict = {}

    def update(self, env):
        """
        Updates the reward statistics from the environment's reward dictionary.
        """
        reward_dict = env.env.env.reward_dict  # Access reward dictionary
        for key, values in reward_dict.items():
            if key not in self.reward_avg_meter_dict:
                self.reward_avg_meter_dict[key] = AverageMeter()
            mean_reward = values.mean().item()
            self.reward_avg_meter_dict[key].update(mean_reward)

    def print(self, curr_timestep, mode="avg"):
        """
        Prints the current reward statistics in a clean format.

        Args:
            curr_timestep (int): Current timestep.
            mode (str): "sum" to print cumulative sums, "avg" to print averages.
        """
        assert mode in ["sum", "avg"], "Mode must be 'sum' or 'avg'"

        # Header
        print("[INFO] =========================================================")
        print(f"[INFO] Timestep: {curr_timestep}")
        print(f"[INFO] Reward {'Sums' if mode == 'sum' else 'Averages'}:")

        # Per-key reward printout
        for key, meter in self.reward_avg_meter_dict.items():
            value = meter.sum if mode == "sum" else meter.avg
            print(f"  [INFO] {key:<20}: {value:.2f}")

        # Total reward
        if mode == "sum":
            total_reward = sum(meter.sum for meter in self.reward_avg_meter_dict.values())
            print(f"[INFO] Total Reward       : {total_reward:.2f}")
        elif mode == "avg":
            total_avg = sum(meter.avg for meter in self.reward_avg_meter_dict.values())
            print(f"[INFO] Total Average      : {total_avg:.2f}")

        print("[INFO] =========================================================")


def get_most_recent_h5py_record_path(base_path, task_name):
    """Find the most recent folder for a given task and return the path to its `h5py_record` subfolder."""
    task_path = os.path.join(base_path, task_name)

    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task folder '{task_name}' not found at {base_path}")

    subdirectories = [
        d for d in os.listdir(task_path)
        if os.path.isdir(os.path.join(task_path, d)) and d.replace("_", "-").replace("-", "").isdigit()
    ]

    if not subdirectories:
        raise FileNotFoundError(f"No subfolders found for task '{task_name}' in {task_path}")

    subdirectories.sort(key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"), reverse=True)
    most_recent_folder = subdirectories[0]

    h5py_record_path = os.path.join(task_path, most_recent_folder, "h5py_record")
    if not os.path.exists(h5py_record_path):
        raise FileNotFoundError(f"h5py_record folder not found in '{os.path.join(task_path, most_recent_folder)}'")

    return h5py_record_path


def save_checkpoint(policy, optimizer, epoch, log_dir, is_best=False):
    """Save the model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved to {checkpoint_path}")

    if is_best:
        best_checkpoint_path = os.path.join(log_dir, "best_model.pt")
        torch.save(checkpoint, best_checkpoint_path)
        print(f"[INFO] Best model saved to {best_checkpoint_path}")


def one_policy_observation_to_inputs(one_policy_observation: torch.tensor, metadata, device):
    """
        Transform one policy observation into 5 inputs that a one policy model accept
        Args:
            one_policy_observation (tensor): The one policy observation. eg. For GenDog1, the size is 340 on the last dimension.
            meta: could be anything that provide the metadata of robot joint and foot numbers. By default, pass in env.unwrapped.
            device: 
        """
    # Dynamic Joint Observations
    nr_dynamic_joint_observations = metadata.nr_dynamic_joint_observations
    single_dynamic_joint_observation_length = metadata.single_dynamic_joint_observation_length
    dynamic_joint_observation_length = metadata.dynamic_joint_observation_length
    dynamic_joint_description_size = metadata.dynamic_joint_description_size

    dynamic_joint_combined_state = one_policy_observation[:, :dynamic_joint_observation_length].view((-1, nr_dynamic_joint_observations, single_dynamic_joint_observation_length))
    dynamic_joint_description = dynamic_joint_combined_state[:, :, :dynamic_joint_description_size]
    dynamic_joint_state = dynamic_joint_combined_state[:, :, dynamic_joint_description_size:]

    # Dynamic Foot Observations
    nr_dynamic_foot_observations = metadata.nr_dynamic_foot_observations
    single_dynamic_foot_observation_length = metadata.single_dynamic_foot_observation_length
    dynamic_foot_observation_length = metadata.dynamic_foot_observation_length
    dynamic_foot_description_size = metadata.dynamic_foot_description_size

    dynamic_foot_combined_state = one_policy_observation[:, dynamic_joint_observation_length:dynamic_joint_observation_length + dynamic_foot_observation_length].view((-1, nr_dynamic_foot_observations, single_dynamic_foot_observation_length))
    dynamic_foot_description = dynamic_foot_combined_state[:, :, :dynamic_foot_description_size]
    dynamic_foot_state = dynamic_foot_combined_state[:, :, dynamic_foot_description_size:]

    policy_general_state_start_index = dynamic_joint_observation_length + dynamic_foot_observation_length
    policy_general_state_end_index = one_policy_observation.shape[1]
    policy_general_state_mask = torch.arange(policy_general_state_start_index, policy_general_state_end_index, device=device)
    # exclude truck_linear_vel and height # 20->16
    policy_exlucion_index = torch.tensor((metadata.trunk_linear_vel_update_obs_idx + metadata.height_update_obs_idx), device=device)
    policy_general_state_mask = policy_general_state_mask[~torch.isin(policy_general_state_mask, policy_exlucion_index)]
    general_policy_state = one_policy_observation[:, policy_general_state_mask]

    # # General Policy State Mask
    # policy_general_state_mask = torch.arange(303, 320, device=self.device)
    # policy_general_state_mask = policy_general_state_mask[policy_general_state_mask != 312]

    # general_policy_state = one_policy_observation[:, policy_general_state_mask]
    inputs = (
        dynamic_joint_description,
        dynamic_joint_state,
        dynamic_foot_description,
        dynamic_foot_state,
        general_policy_state
    )
    return inputs

def ensure_unique_save_path(save_path):
    base_name, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base_name}_{counter}{ext}"
        counter += 1
    return save_path