import argparse
import os
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter  
from dataset import LocomotionDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))
import silver_badger_torch
import tqdm


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--tasks", nargs="+", type=str, default=None, help="List of tasks to process.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run.")
    parser.add_argument("--batch_size", type=int, default=4090*16, help="Batch size.")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Name of the experiment. Default is the current date and time.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for torch data loder. ")
    parser.add_argument("--max_files_in_memory", type=int, default=8, help="Max number of data files in memory.")
    return parser.parse_args()

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


def train(policy, criterion, optimizer, data_loader, num_epochs, model_device, log_dir, checkpoint_interval):
    """Training loop with TensorBoard logging and checkpoint saving."""
    writer = SummaryWriter(log_dir=log_dir)
    loss_meter = AverageMeter()
    best_loss = float("inf")

    print("[INFO] Starting supervised training.")

    for epoch in range(num_epochs):
        loss_meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs}.")

        for batch_inputs, batch_targets in tqdm.tqdm(data_loader):
            # Move data to device
            batch_inputs = [x.to(model_device) for x in batch_inputs] 
            batch_targets = batch_targets.to(model_device)

            # Unpack dataset-specific transformed inputs
            (
                dynamic_joint_description,
                dynamic_joint_state,
                dynamic_foot_description,
                dynamic_foot_state,
                general_policy_state,
            ) = batch_inputs

            # Forward pass
            # print(dynamic_joint_description.shape, dynamic_joint_state.shape, dynamic_foot_description.shape, 
            #       dynamic_foot_state.shape, general_policy_state.shape)
            batch_predictions = policy(
                dynamic_joint_description,
                dynamic_joint_state,
                dynamic_foot_description,
                dynamic_foot_state,
                general_policy_state,
            )
            loss = criterion(batch_predictions, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss tracker
            loss_meter.update(loss.item(), n=batch_targets.size(0))

        # Log loss to TensorBoard
        writer.add_scalar("Loss/train", loss_meter.avg, epoch + 1)

        # Save checkpoints periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(policy, optimizer, epoch + 1, log_dir)

        # Save the best model
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(policy, optimizer, epoch + 1, log_dir, is_best=True)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_meter.avg:.6f}, Best Loss: {best_loss:.6f}")

    writer.close()
    print("[INFO] Training completed. TensorBoard logs saved.")


def main():
    args_cli = parse_arguments()

    # Prepare log directory and dataset paths
    log_dir = os.path.join(args_cli.log_dir, args_cli.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    dataset_dirs = [get_most_recent_h5py_record_path("logs/rsl_rl", task) for task in args_cli.tasks]
    dataset = LocomotionDataset(folder_paths=dataset_dirs, max_files_in_memory=args_cli.max_files_in_memory)
    data_loader = dataset.get_data_loader(batch_size=args_cli.batch_size, shuffle=True,
                                          num_workers=args_cli.num_workers)

    # Define model, optimizer, and loss
    from silver_badger_torch.policy import get_policy

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = get_policy(model_device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)

    # Train the policy
    train(
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=data_loader,
        num_epochs=args_cli.num_epochs,
        model_device=model_device,
        log_dir=log_dir,
        checkpoint_interval=args_cli.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
