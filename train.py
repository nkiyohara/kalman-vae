import argparse
import os
from datetime import datetime
from typing import Literal

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from bouncing_ball.dataloaders.bouncing_data import BouncingBallDataLoader
from config import TrainingConfig
from evaluation import evaluate
from kalman_vae import KalmanVariationalAutoencoder
from sample_control import SampleControl


def setup_dataloaders(root_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Set up the training and testing data loaders.
    """
    dataloader_train = DataLoader(
        BouncingBallDataLoader(root_dir=f"{root_dir}/train"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sequence_first_collate_fn,
    )

    dataloader_test = DataLoader(
        BouncingBallDataLoader(root_dir=f"{root_dir}/test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_first_collate_fn,
    )

    return dataloader_train, dataloader_test


def setup_model_optimizer_scheduler(
    dataloader_train: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[KalmanVariationalAutoencoder, optim.Optimizer, lr_scheduler.LRScheduler]:
    """
    Set up the model, optimizer, and scheduler.
    """
    # Get first batch to setup model
    for batch in dataloader_train:
        first_batch = batch
        break

    kvae = (
        KalmanVariationalAutoencoder(
            image_size=first_batch.shape[3:],
            image_channels=first_batch.shape[2],
            a_dim=config.a_dim,
            z_dim=config.z_dim,
            K=config.K,
            decoder_type=config.decoder_type,
            dynamics_parameter_network=config.dynamics_parameter_network,
        )
        .to(dtype=dtype)
        .to(device)
    )

    optimizer = optim.Adam(kvae.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.learning_rate_decay)

    return kvae, optimizer, scheduler


def run_epoch(
    dataloader: DataLoader,
    kvae: KalmanVariationalAutoencoder,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    mode: Literal["train", "test"],
    epoch: int,
    sample_control: SampleControl,
) -> tuple[float, dict]:
    """
    Run training or testing for one epoch.
    """
    if mode == "train":
        kvae.train()
    else:
        kvae.eval()

    total_loss = 0.0
    metrics = {
        "reconstruction": 0.0,
        "regularization": 0.0,
        "kalman_observation_log_likelihood": 0.0,
        "kalman_state_transition_log_likelihood": 0.0,
        "kalman_posterior_log_likelihood": 0.0,
    }
    n_batches = len(dataloader)

    for data in dataloader:
        data = (data > 0.5).to(dtype=dtype).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == "train"):
            elbo, info = kvae.elbo(
                xs=data,
                reconst_weight=config.reconst_weight,
                regularization_weight=config.regularization_weight,
                kalman_weight=config.kalman_weight,
                kl_weight=config.kl_weight,
                batch_operation=config.batch_operation,
                sequence_operation=config.sequence_operation,
                symmetrize_covariance=config.symmetrize_covariance,
                burn_in=config.burn_in,
                learn_weight_model=(epoch >= config.warmup_epochs),
                sample_control=sample_control,
            )
            loss = -elbo
            if mode == "train":
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        for key in metrics:
            metrics[key] += info[key] / n_batches  # Accumulate the average

    average_loss = total_loss / n_batches
    return average_loss, metrics


def save_checkpoint(
    kvae: KalmanVariationalAutoencoder,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    epoch: int,
    train_loss: float,
    test_loss: float,
    checkpoint_dir: str,
) -> None:
    """
    Save the model checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": kvae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        },
        os.path.join(checkpoint_dir, f"state-{epoch}.pth"),
    )


def sequence_first_collate_fn(batch: list) -> torch.Tensor:
    """
    Collate function for the DataLoader.
    """
    data = torch.Tensor(np.stack(batch, axis=0))
    data = data.permute(1, 0, 2, 3, 4)
    return data


def train(config: TrainingConfig) -> None:
    """
    Main training function.
    """
    # Initialize wandb
    wandb.init(project=config.project_name, name=config.name, config=config._asdict())

    # Setup device and dtype
    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    # Setup DataLoaders
    dataloader_train, dataloader_test = setup_dataloaders(
        config.data_root_dir, config.batch_size
    )

    # Setup Model, Optimizer, Scheduler
    kvae, optimizer, scheduler = setup_model_optimizer_scheduler(
        dataloader_train, config, device, dtype
    )

    # Setup Sample Control
    sample_control_train = SampleControl()
    sample_control_test = SampleControl(
        encoder="mean", decoder="mean", state_transition="mean", observation="mean"
    )

    # Training Loop
    for epoch in tqdm(range(config.epochs)):
        train_loss, train_metrics = run_epoch(
            dataloader=dataloader_train,
            kvae=kvae,
            optimizer=optimizer,
            config=config,
            device=device,
            dtype=dtype,
            mode="train",
            epoch=epoch,
            sample_control=sample_control_train,
        )
        test_loss, test_metrics = run_epoch(
            dataloader=dataloader_test,
            kvae=kvae,
            optimizer=optimizer,
            config=config,
            device=device,
            dtype=dtype,
            mode="test",
            epoch=epoch,
            sample_control=sample_control_test,
        )

        if config.evaluation_interval > 0:
            if epoch % config.evaluation_interval == 0:
                random_masking, continuous_masking = evaluate(
                    dataloader=dataloader_test,
                    kvae=kvae,
                    sample_control=sample_control_test,
                    checkpoint_dir=config.checkpoint_dir,
                    epoch=epoch,
                    device=device,
                    dtype=dtype,
                )

                wandb.log(
                    {
                        "random_masking": wandb.Table(dataframe=random_masking),
                        "continuous_masking": wandb.Table(dataframe=continuous_masking),
                        "epoch": epoch,
                    }
                )

        # Log losses and metrics
        wandb.log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "epoch": epoch,
            }
        )

        # Scheduler Step and Checkpoint Saving
        if (epoch > 0) & (epoch % config.scheduler_step == 0):
            scheduler.step()

        save_checkpoint(
            kvae=kvae,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            checkpoint_dir=config.checkpoint_dir,
        )


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a Kalman Variational Autoencoder"
    )

    data_group = parser.add_argument_group("Data settings")
    data_group.add_argument(
        "--data_root_dir",
        type=str,
        default="bouncing_ball/datasets/bouncing-ball",
        help="Root directory of the data",
    )
    data_group.add_argument("--batch_size", type=int, default=128, help="Batch size")

    model_group = parser.add_argument_group("Model settings")
    model_group.add_argument(
        "--z_dim", type=int, default=4, help="Dimension of latent space z"
    )
    model_group.add_argument(
        "--a_dim", type=int, default=2, help="Dimension of encoded space a"
    )
    model_group.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of matrices for calculating the weighted average in the observation and transition matrices",
    )
    model_group.add_argument(
        "--dynamics_parameter_network",
        type=str,
        default="lstm",
        choices=["mlp", "lstm"],
        help="Type of dynamics parameter network",
    )
    model_group.add_argument(
        "--decoder_type",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "gaussian"],
        help="Type of decoder",
    )
    model_group.add_argument(
        "--reconst_weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    model_group.add_argument(
        "--regularization_weight",
        type=float,
        default=0.3,
        help="Weight for regularization loss",
    )
    model_group.add_argument(
        "--symmetrize_covariance",
        type=bool,
        default=False,
        help="Symmetrize the covariance matrix",
    )
    model_group.add_argument(
        "--kalman_weight", type=float, default=1.0, help="Weight for Kalman loss"
    )
    model_group.add_argument(
        "--kl_weight",
        type=float,
        default=0.0,
        help="Weight for KL loss (when training VAE individually)",
    )

    train_group = parser.add_argument_group("Training settings")
    train_group.add_argument(
        "--epochs", type=int, default=80, help="Number of training epochs"
    )
    train_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to train without updating dynamics parameter network",
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=7e-3, help="Learning rate"
    )
    train_group.add_argument(
        "--learning_rate_decay", type=float, default=0.8, help="Learning rate decay"
    )
    train_group.add_argument(
        "--burn_in", type=int, default=3, help="Number of burn-in steps"
    )
    train_group.add_argument(
        "--batch_operation",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Batch operation",
    )
    train_group.add_argument(
        "--sequence_operation",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Sequence operation",
    )
    train_group.add_argument(
        "--scheduler_step",
        type=int,
        default=20,
        help="Number of epochs between scheduler steps",
    )

    env_group = parser.add_argument_group("Environment settings")
    env_group.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    env_group.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Data type for tensors (float32 or float64)",
    )
    env_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_group.add_argument(
        "--name", type=str, default=current_time, help="Name of the experiment"
    )
    env_group.add_argument(
        "--evaluation_interval",
        type=int,
        default=10,
        help="Number of epochs between evaluations. Set to 0 to disable evaluation.",
    )

    args = parser.parse_args()

    return TrainingConfig(
        data_root_dir=args.data_root_dir,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        a_dim=args.a_dim,
        K=args.K,
        dynamics_parameter_network=args.dynamics_parameter_network,
        decoder_type=args.decoder_type,
        reconst_weight=args.reconst_weight,
        regularization_weight=args.regularization_weight,
        kalman_weight=args.kalman_weight,
        kl_weight=args.kl_weight,
        symmetrize_covariance=args.symmetrize_covariance,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        burn_in=args.burn_in,
        batch_operation=args.batch_operation,
        sequence_operation=args.sequence_operation,
        scheduler_step=args.scheduler_step,
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        project_name="Kalman-VAE",
        name=args.name,
        evaluation_interval=args.evaluation_interval,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
