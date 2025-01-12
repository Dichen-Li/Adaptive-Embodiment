import os
import torch
from torch.cuda.amp import GradScaler, autocast
torch.backends.cudnn.benchmark = True

from torch.utils.tensorboard import SummaryWriter
import time
from utils import get_most_recent_h5py_record_path, save_checkpoint, AverageMeter, save_args_to_yaml, compute_gradient_norm
from dataset import LocomotionDataset_tmu
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')))
import tqdm
import argparse
import yaml

import time
from collections import defaultdict


class NonOverlappingTimeProfiler(object):
    def __init__(self):
        self.time_cost = defaultdict(float)
        self.tic = time.time()

    def end(self, key):
        toc = time.time()
        self.time_cost[key] += toc - self.tic
        self.tic = toc

    def reset(self):
        self.time_cost.clear()
        self.tic = time.time()

    def read(self):
        tot_time = sum(self.time_cost.values())
        ratio = {f'{k}_ratio': v / tot_time for k, v in self.time_cost.items()}
        return {**self.time_cost, **ratio, **{'total': tot_time}}
    
    def dump_to_writer(self, writer: SummaryWriter, global_step):
        time_stat = self.read()
        writer.add_scalar("time/SPS", global_step / time_stat.pop('total'), global_step)
        for k, v in time_stat.items():
            if k.endswith('ratio'):
                writer.add_scalar(f"time/{k}", v, global_step)
            else:
                writer.add_scalar(f"time/{k}_SPS", global_step / v, global_step)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an agent using supervised learning.")

    # Define arguments with defaults
    parser.add_argument("--train_set", nargs="+", type=str, default=None, required=True,
                        help="List of robot names as the training set.")
    parser.add_argument("--train_merged_h5", type=str, required=True)
    parser.add_argument("--test_set", nargs="+", type=str, default=None, required=False,
                        help="List of robot names as the test set.")
    parser.add_argument("--test_merged_h5", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to run.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size. 4096*16 takes 10G")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Name of the experiment. If provided, the current date and time will be appended. "
                             "Default is the current date and time.")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--log_dir", type=str, default="log_dir", help="Base directory for logs and checkpoints.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Unit learning rate (for a batch size of 512)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for torch data loader.")
    parser.add_argument("--max_files_in_memory", type=int, default=1, help="Max number of data files in memory.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set size.")
    parser.add_argument("--gradient_acc_steps", type=int, default=1,
                        help="Number of batches before one gradient update.")
    parser.add_argument("--h5_repeat_factor", type=int, default=1, help="Number of times we repeat one h5 file consecutively in one epoch.")
    parser.add_argument("--use_amp", type=int, default=0, help="Whether to use automatic mixed precision.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["urma", "rsl_rl_actor", "naive_actor"],
        help="Model type."
    )
    # Add argument for YAML configuration
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--dataset_dir", type=str, default="/media/t7-ssd/Data/logs/rsl_rl", help="Directory containing the dataset.")

    args = parser.parse_args()

    # Load and override arguments from YAML file if specified
    if args.config:
        print(f'Loading configuration from {args.config}. Specified params will be overridden.')

        with open(args.config, 'r') as file:
            config_args = yaml.safe_load(file)

        for key, value in config_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Fill missing arguments with defaults
    for action in parser._actions:
        if action.dest == "help":
            continue
        if getattr(args, action.dest) is None and action.default is not None:
            setattr(args, action.dest, action.default)

    return args


def get_meter_dict_avg(meter_dicts):
    if len(meter_dicts) == 0:
        return 0
    return sum([meter.avg for meter in meter_dicts.values()])/len(meter_dicts)


def train(policy, criterion, optimizer, scheduler, train_dataset, val_dataset, test_dataset, num_epochs, model_device,
          log_dir, checkpoint_interval, model, gradient_acc_steps, batch_size, num_workers, use_amp):
    """Training loop with validation, TensorBoard logging, and checkpoint saving."""
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=log_dir)
    train_loss_meters = {}
    val_loss_meters = {}
    test_loss_meters = {}
    best_val_loss = float("inf")

    print("[INFO] Starting supervised training.")
    timer = NonOverlappingTimeProfiler()
    grad_step_cnt = 0
    for epoch in range(num_epochs):
        # Training phase
        policy.train()
        for meter in train_loss_meters.values():
            meter.reset()
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Training.")

        train_dataloader = train_dataset.get_data_loader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=25 if num_workers > 0 else None
        )
        timer.end('prepare')

        with tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            iteration_start_time = time.time()
            for index, data_batch in enumerate(pbar):
                grad_step_cnt += 1
                if grad_step_cnt % 100 == 0:
                    timer.dump_to_writer(writer, grad_step_cnt)
                
                batch_inputs = [
                    data_batch['dynamic_joint_description'],
                    data_batch['dynamic_joint_state'],
                    data_batch['general_state'],
                ]
                batch_targets = data_batch['target']
                data_source_name = 'mix'
                io_times = data_batch['io_time']
                processing_times = data_batch['data_processing_time']

                iteration = index + epoch * len(train_dataloader)
                # dataloader_time = time.time() - iteration_start_time

                # Move data to device
                # start_time = time.time()
                batch_inputs = [x.to(model_device) for x in batch_inputs]
                batch_targets = batch_targets.to(model_device)
                # move_cuda_time = time.time() - start_time
                timer.end('data_loading')

                # start_time = time.time()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    if model == 'urma':
                        batch_predictions = policy(*batch_inputs)
                    else:
                        raise NotImplementedError

                    loss = criterion(batch_predictions, batch_targets)
                # forward_time = time.time() - start_time
                timer.end('forward')

                # Backward pass and optimization
                # start_time = time.time()

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Only update optimizer after accumulating enough gradients
                grad_norm = None  # Initialize gradient norm
                if index % gradient_acc_steps == gradient_acc_steps - 1:
                    if use_amp:
                        # Compute gradient norm before unscaled gradients are modified
                        # scaler.unscale_(optimizer)  # Unscale gradients for logging (optional with AMP)
                        grad_norm = compute_gradient_norm(policy)  # Log this value
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Compute gradient norm before optimizer step
                        torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=5.0)
                        grad_norm = compute_gradient_norm(policy)  # Log this value
                        optimizer.step()

                    optimizer.zero_grad()  # Zero gradients after optimization step

                timer.end('backward')
                # backward_time = time.time() - start_time

                # Update training loss tracker
                if data_source_name not in train_loss_meters:
                    train_loss_meters[data_source_name] = AverageMeter()
                train_loss_meters[data_source_name].update(loss.item(), n=batch_targets.size(0))

                # Update progress bar with current loss
                pbar.set_postfix({"Loss": f"{get_meter_dict_avg(train_loss_meters):.4f}"})

                # Log times
                writer.add_scalar("Train/times/io_per_thread", io_times.mean().item(), iteration)
                writer.add_scalar("Train/times/data_processing_per_thread", processing_times.mean().item(), iteration)

                # Log loss and lr by iteration
                writer.add_scalar("Train/loss-iter/avg", loss.item(), iteration)
                writer.add_scalar("Train/lr-iter", optimizer.param_groups[0]['lr'], iteration)
                if grad_norm is not None:
                    writer.add_scalar("Train/grad_norm-iter", grad_norm, iteration)

                # Step the LR scheduler by iteration
                if scheduler is not None:
                    scheduler.step()

                iteration_start_time = time.time()  # start time of next iteration

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
            batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=25
        )

        with torch.no_grad():
            with tqdm.tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for index, data_batch in enumerate(pbar):
                    batch_inputs = [
                        data_batch['dynamic_joint_description'],
                        data_batch['dynamic_joint_state'],
                        data_batch['general_state'],
                    ]
                    batch_targets = data_batch['target']
                    data_source_name = 'mix'
                    # io_times = data_batch['io_time']
                    # processing_times = data_batch['processing_time']

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

                    if index > 0.1 * len(val_dataloader):
                        break

        # Log validation loss to TensorBoard
        for robot_name, meter in val_loss_meters.items():
            writer.add_scalar(f"Val/loss/{robot_name}", meter.avg, epoch + 1)
        writer.add_scalar("Val/loss/avg", get_meter_dict_avg(val_loss_meters), epoch + 1)
        timer.end('validation')

        if len(test_dataset) > 0:
            # Test phase
            for meter in test_loss_meters.values():
                meter.reset()
            print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs} - Test.")

            test_dataloader = test_dataset.get_data_loader(
                batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=25
            )

            with torch.no_grad():
                with tqdm.tqdm(test_dataloader, desc=f"Test Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                    for index, (batch_inputs, batch_targets, data_source_name, _, _) in enumerate(pbar):
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
                        if data_source_name not in test_loss_meters:
                            test_loss_meters[data_source_name] = AverageMeter()
                        test_loss_meters[data_source_name].update(loss.item(), n=batch_targets.size(0))

                        # Update progress bar with current loss
                        pbar.set_postfix(
                            {"Loss": f"{get_meter_dict_avg(test_loss_meters):.4f}"})

                        if index > 0.1 * len(test_dataloader):
                            break

            timer.end('test')
            # Log validation loss to TensorBoard
            for robot_name, meter in test_loss_meters.items():
                writer.add_scalar(f"Test/loss/{robot_name}", meter.avg, epoch + 1)
            writer.add_scalar("Test/loss/avg", get_meter_dict_avg(test_loss_meters), epoch + 1)

        # Save checkpoints periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(policy, optimizer, epoch + 1, log_dir)
            timer.end('checkpoint')

        # Save the best model based on validation loss
        if get_meter_dict_avg(val_loss_meters) < best_val_loss:
            best_val_loss = get_meter_dict_avg(val_loss_meters)
            save_checkpoint(policy, optimizer, epoch + 1, log_dir, is_best=True)
            timer.end('checkpoint')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {get_meter_dict_avg(train_loss_meters):.6f}, "
              f"Val Loss: {get_meter_dict_avg(val_loss_meters):.6f}, Best Val Loss: {best_val_loss:.6f}, "
              f"Test Loss: {get_meter_dict_avg(test_loss_meters):.6f}")

    writer.close()
    print("[INFO] Training completed. TensorBoard logs saved.")


def main():
    args_cli = parse_arguments()

    import wandb
    os.environ["WANDB_API_KEY"] = "44713a60b687b7a3dbe558ae6ef945cbeacb756e"

    wandb.init(
        project='esl_distillation',
        entity=None,
        sync_tensorboard=True,
        config=vars(args_cli),
        name=str(args_cli.exp_name).replace(os.path.sep, "__"),
        save_code=True,
    )

    # Prepare log directory
    log_dir = os.path.join(args_cli.log_dir, args_cli.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save args to a YAML file for reproducibility
    config_save_path = os.path.join(log_dir, "config.yaml")
    save_args_to_yaml(args_cli, config_save_path)
    print(f"[INFO] Config saved to {config_save_path}")

    # Dataset paths
    assert args_cli.train_set is not None, f"Please specify value for arg --train_set"
    train_set_paths = [get_most_recent_h5py_record_path(args_cli.dataset_dir, task) for task in args_cli.train_set]
    if args_cli.test_set:
        test_set_paths = [get_most_recent_h5py_record_path(args_cli.dataset_dir, task) for task in args_cli.test_set]
        assert args_cli.test_merged_h5 is not None
    else:
        test_set_paths = list()
        print(f'[INFO] No test set provided.')

    # Training dataset
    train_dataset = LocomotionDataset_tmu(
        merged_h5_path=args_cli.train_merged_h5,
        folder_paths=train_set_paths,
        train_mode=True,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory,
        h5_repeat_factor=args_cli.h5_repeat_factor
    )

    # Validation dataset
    val_dataset = LocomotionDataset_tmu(
        merged_h5_path=args_cli.train_merged_h5,
        folder_paths=train_set_paths,
        train_mode=False,
        val_ratio=args_cli.val_ratio,
        max_files_in_memory=args_cli.max_files_in_memory,
        h5_repeat_factor=1
    )

    # Test dataset
    test_dataset = LocomotionDataset_tmu(
        merged_h5_path=args_cli.test_merged_h5,
        folder_paths=test_set_paths,
        train_mode=False,
        val_ratio=args_cli.val_ratio,       # only use a proportion of the data as the test set
        max_files_in_memory=args_cli.max_files_in_memory,
        h5_repeat_factor=1
    )

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer, and loss
    if args_cli.model == 'urma':
        from silver_badger_torch.policy import get_policy
        policy = get_policy(model_device)

        # # load checkpoint if needed
        # checkpoint_path = "/home/albert/github/embodiment-scaling-law-sim2real/log_dir/scaling_factor_0.1_v3_modelscale3_attempt2/best_model.pt"
        # checkpoint = torch.load(checkpoint_path, map_location=model_device)
        # policy.load_state_dict(checkpoint["state_dict"], strict=True)
        # print(f'[INFO] Policy loaded from {checkpoint_path}\n\n')
        # import ipdb; ipdb.set_trace()

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
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args_cli.lr,
        weight_decay=5e-4,
        betas=(0.95, 0.999)  # Adjusted betas for smoother gradients
    )
    # optimizer = torch.optim.SGD(policy.parameters(), lr=args_cli.lr, momentum=0.95, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args_cli.num_epochs*len(train_dataset)/args_cli.batch_size)
    # scheduler = None

    # Train the policy
    train(
        policy=policy,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_epochs=args_cli.num_epochs,
        model_device=model_device,
        log_dir=log_dir,
        checkpoint_interval=args_cli.checkpoint_interval,
        model=args_cli.model,
        gradient_acc_steps=args_cli.gradient_acc_steps,
        batch_size=args_cli.batch_size,
        num_workers=args_cli.num_workers,
        use_amp=bool(args_cli.use_amp)
    )


if __name__ == "__main__":
    main()
