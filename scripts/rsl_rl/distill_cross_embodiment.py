import argparse
import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_most_recent_h5py_record_path, save_checkpoint, AverageMeter
from dataset import LocomotionDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))
import tqdm


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--tasks", nargs="+", type=str, default=None, help="List of tasks to process.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run.")
    parser.add_argument("--batch_size", type=int, default=4090*8, help="Batch size.")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        help="Name of the experiment. Default is the current date and time.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for torch data loder. ")
    parser.add_argument("--max_files_in_memory", type=int, default=8, help="Max number of data files in memory.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set size.")
    return parser.parse_args()


def train(policy, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, model_device,
          log_dir, checkpoint_interval):
    """Training loop with validation, TensorBoard logging, and checkpoint saving."""
    writer = SummaryWriter(log_dir=log_dir)
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    best_val_loss = float("inf")

    print("[INFO] Starting supervised training.")

    for epoch in range(num_epochs):
        # Training phase
        policy.train()
        train_loss_meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Training.")

        for batch_inputs, batch_targets in tqdm.tqdm(train_loader, desc="Training"):
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

            # Update training loss tracker
            train_loss_meter.update(loss.item(), n=batch_targets.size(0))

        # Log training loss to TensorBoard
        writer.add_scalar("Train/loss", train_loss_meter.avg, epoch + 1)
        writer.add_scalar("Train/lr", scheduler.get_lr(), epoch + 1)

        # Validation phase
        policy.eval()
        val_loss_meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Validation.")

        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm.tqdm(val_loader, desc="Validation"):
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
                batch_predictions = policy(
                    dynamic_joint_description,
                    dynamic_joint_state,
                    dynamic_foot_description,
                    dynamic_foot_state,
                    general_policy_state,
                )
                loss = criterion(batch_predictions, batch_targets)

                # Update validation loss tracker
                val_loss_meter.update(loss.item(), n=batch_targets.size(0))

        # Log validation loss to TensorBoard
        writer.add_scalar("Val/loss", val_loss_meter.avg, epoch + 1)

        # Step the LR scheduler
        scheduler.step()

        # Save checkpoints periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(policy, optimizer, epoch + 1, log_dir)

        # Save the best model based on validation loss
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            save_checkpoint(policy, optimizer, epoch + 1, log_dir, is_best=True)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_meter.avg:.6f}, "
              f"Val Loss: {val_loss_meter.avg:.6f}, Best Val Loss: {best_val_loss:.6f}")

    writer.close()
    print("[INFO] Training completed. TensorBoard logs saved.")


def main():
    args_cli = parse_arguments()

    # Prepare log directory
    log_dir = os.path.join(args_cli.log_dir, args_cli.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Dataset paths
    dataset_dirs = [get_most_recent_h5py_record_path("logs/rsl_rl", task) for task in args_cli.tasks]

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

    # Validation dataset
    val_dataset = LocomotionDataset(
        folder_paths=dataset_dirs,
        train_mode=False,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory
    )
    val_loader = val_dataset.get_data_loader(
        batch_size=args_cli.batch_size, shuffle=False, num_workers=args_cli.num_workers
    )

    # Define model, optimizer, and loss
    from silver_badger_torch.policy import get_policy

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = get_policy(model_device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_cli.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_cli.num_epochs)

    # Train the policy
    train(
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args_cli.num_epochs,
        model_device=model_device,
        log_dir=log_dir,
        checkpoint_interval=args_cli.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
