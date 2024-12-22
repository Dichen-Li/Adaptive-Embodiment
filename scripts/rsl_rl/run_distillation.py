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
    parser = argparse.ArgumentParser(description="Train an agent using supervised learning.")
    parser.add_argument("--tasks", nargs="+", type=str, default=None, required=True,
                        help="List of tasks to process.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. 4096*16 takes 10G")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Name of the experiment. If provided, the current date and time will be appended. "
                             "Default is the current date and time.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
    parser.add_argument("--lr", type=float, default=3e-4, help="unit learning rate (for a batch size of 512)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for torch data loder.")
    parser.add_argument("--max_files_in_memory", type=int, default=1, help="Max number of data files in memory.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set size.")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of batches before one gradient update.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["urma", "rsl_rl_actor", "naive_actor"],
        help="Model type."
    )

    args = parser.parse_args()

    # Handle experiment name
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.exp_name:
        args.exp_name = f"{current_datetime}_{args.exp_name}"
    else:
        args.exp_name = current_datetime

    # Process learning rate
    # args.lr = args.lr * (args.batch_size / 4096)
    args.lr = args.lr * (args.batch_size / 512)

    # For learning rate:
    # 1.25e-4 seems to work well with batch size 4096; however, it seems good to use 3e-4 with batch size 512

    return args


def get_meter_dict_avg(meter_dicts):
    return sum([meter.avg for meter in meter_dicts.values()])/len(meter_dicts)


def train(policy, criterion, optimizer, scheduler, train_dataset, val_dataset, num_epochs, model_device,
          log_dir, checkpoint_interval, model, gradient_acc_steps, batch_size, num_workers):
    """Training loop with validation, TensorBoard logging, and checkpoint saving."""
    writer = SummaryWriter(log_dir=log_dir)
    train_loss_meters = {}
    val_loss_meters = {}
    best_val_loss = float("inf")

    print("[INFO] Starting supervised training.")
    for epoch in range(num_epochs):
        # Training phase
        policy.train()
        for meter in train_loss_meters.values():
            meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Training.")

        train_dataloader = train_dataset.get_data_loader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

        with tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for index, (batch_inputs, batch_targets, data_source_name) in enumerate(pbar):

                # times = []
                # import time
                # start_time = time.time()

                # Move data to device
                batch_inputs = [x.to(model_device) for x in batch_inputs]
                batch_targets = batch_targets.to(model_device)

                # end_time = time.time()
                # times.append(end_time - start_time)
                # start_time = time.time()

                if model == 'urma':
                    batch_predictions = policy(*batch_inputs)
                else:
                    one_input = torch.cat([
                        component.flatten(1) for component in batch_inputs
                    ], dim=1)
                    batch_predictions = policy(one_input)

                # end_time = time.time()
                # times.append(end_time - start_time)
                # start_time = time.time()

                loss = criterion(batch_predictions, batch_targets)

                # end_time = time.time()
                # times.append(end_time - start_time)
                # start_time = time.time()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # end_time = time.time()
                # times.append(end_time - start_time)
                # start_time = time.time()

                # backward only when we have accumulated gradients for enough batches
                if index % gradient_acc_steps == gradient_acc_steps - 1:
                    optimizer.step()

                # end_time = time.time()
                # times.append(end_time - start_time)
                # start_time = time.time()

                # Update training loss tracker
                if data_source_name not in train_loss_meters:
                    train_loss_meters[data_source_name] = AverageMeter()
                train_loss_meters[data_source_name].update(loss.item(), n=batch_targets.size(0))

                # Update progress bar with current loss
                pbar.set_postfix({"Loss": f"{get_meter_dict_avg(train_loss_meters):.4f}"})

                # end_time = time.time()
                # times.append(end_time - start_time)
                # if index % 1000 == 0:
                #     print(f'times: {times}')

        # Log training loss to TensorBoard
        for robot_name, meter in train_loss_meters.items():
            writer.add_scalar(f"Train/loss/{robot_name}", meter.avg, epoch + 1)
        writer.add_scalar("Train/loss/avg", get_meter_dict_avg(train_loss_meters), epoch + 1)
        writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], epoch + 1)

        # Validation phase
        policy.eval()
        for meter in val_loss_meters.values():
            meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Validation.")

        val_dataloader = val_dataset.get_data_loader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        with torch.no_grad():
            with tqdm.tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for index, (batch_inputs, batch_targets, data_source_name) in enumerate(pbar):
                    # Move data to device
                    batch_inputs = [x.to(model_device) for x in batch_inputs]
                    batch_targets = batch_targets.to(model_device)

                    if model == 'urma':
                        batch_predictions = policy(*batch_inputs)
                    else:
                        one_input = torch.cat([
                            component.flatten(1) for component in batch_inputs
                        ], dim=1)
                        batch_predictions = policy(one_input)

                    loss = criterion(batch_predictions, batch_targets)

                    # Update validation loss tracker
                    if data_source_name not in val_loss_meters:
                        val_loss_meters[data_source_name] = AverageMeter()
                    val_loss_meters[data_source_name].update(loss.item(), n=batch_targets.size(0))

                    # Update progress bar with current loss
                    pbar.set_postfix(
                        {"Loss": f"{get_meter_dict_avg(val_loss_meters):.4f}"})

        # Log validation loss to TensorBoard
        for robot_name, meter in val_loss_meters.items():
            writer.add_scalar(f"Val/loss/{robot_name}", meter.avg, epoch + 1)
        writer.add_scalar("Val/loss/avg", get_meter_dict_avg(val_loss_meters), epoch + 1)

        # Step the LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoints periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(policy, optimizer, epoch + 1, log_dir)

        # Save the best model based on validation loss
        if get_meter_dict_avg(val_loss_meters) < best_val_loss:
            best_val_loss = get_meter_dict_avg(val_loss_meters)
            save_checkpoint(policy, optimizer, epoch + 1, log_dir, is_best=True)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {get_meter_dict_avg(train_loss_meters):.6f}, "
              f"Val Loss: {get_meter_dict_avg(val_loss_meters):.6f}, Best Val Loss: {best_val_loss:.6f}")

    writer.close()
    print("[INFO] Training completed. TensorBoard logs saved.")


def main():
    args_cli = parse_arguments()

    # Prepare log directory
    log_dir = os.path.join(args_cli.log_dir, args_cli.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Dataset paths
    assert args_cli.tasks is not None, f"Please specify value for arg --tasks"
    dataset_dirs = [get_most_recent_h5py_record_path("logs/rsl_rl", task) for task in args_cli.tasks]

    # Training dataset
    train_dataset = LocomotionDataset(
        folder_paths=dataset_dirs,
        train_mode=True,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory
    )

    # Validation dataset
    val_dataset = LocomotionDataset(
        folder_paths=dataset_dirs,
        train_mode=False,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory
    )

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer, and loss
    if args_cli.model == 'urma':
        from silver_badger_torch.policy import get_policy
        policy = get_policy(model_device)

        # # load checkpoint if needed
        # checkpoint_path = "/home/albert/github/embodiment-scaling-law-sim2real/log_dir/2024-12-20_01-36-32_debug/best_model.pt"
        # checkpoint = torch.load(checkpoint_path, map_location=model_device)
        # policy.load_state_dict(checkpoint["state_dict"], strict=False)

        # from supervised_actor.policy import get_policy
        # metadata = train_dataset.metadata_list[0]
        # nr_dynamic_joint_observations = metadata['nr_dynamic_joint_observations']
        # policy = get_policy(nr_dynamic_joint_observations, model_device)

    elif args_cli.model == 'rsl_rl_actor':
        # use a simple MLP from RSL-RL
        from rsl_rl.modules import ActorCritic
        # metadata = train_dataset.metadata_list[0]
        # import ipdb; ipdb.set_trace()
        actor_critic = ActorCritic(376, 376, 12)
        policy = actor_critic.actor.to(model_device)
    elif args_cli.model == 'naive_actor':
        from naive_actor import ActorMLP
        policy = ActorMLP(input_dim=316, hidden_dim=256, output_dim=12,
                          activation=torch.nn.LeakyReLU(negative_slope=0.03)).to(model_device)
    else:
        raise NotImplementedError(f'model type {args_cli.model} not implemented')

    print('policy architecture:\n', policy)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_cli.num_epochs)
    # scheduler = None

    # Train the policy
    train(
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args_cli.num_epochs,
        model_device=model_device,
        log_dir=log_dir,
        checkpoint_interval=args_cli.checkpoint_interval,
        model=args_cli.model,
        gradient_acc_steps=args_cli.gradient_acc_steps,
        batch_size=args_cli.batch_size,
        num_workers=args_cli.num_workers
    )


if __name__ == "__main__":
    main()
