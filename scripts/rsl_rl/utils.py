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
