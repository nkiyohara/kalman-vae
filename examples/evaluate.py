import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from bouncing_ball.dataloaders.bouncing_data import BouncingBallDataLoader
from config import EvaluationConfig
from evaluation import evaluate as evaluate_model
from kalman_vae import KalmanVariationalAutoencoder
from sample_control import SampleControl
from torch.utils.data import DataLoader


def setup_dataloader(root_dir: str, batch_size: int) -> DataLoader:
    dataloader_test = DataLoader(
        BouncingBallDataLoader(root_dir=f"{root_dir}/test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_first_collate_fn,
    )

    return dataloader_test


def setup_model_optimizer_scheduler(
    dataloader_test: DataLoader,
    checkpoint_file: str,
    config: EvaluationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[KalmanVariationalAutoencoder, optim.Optimizer, lr_scheduler.LRScheduler]:
    for batch in dataloader_test:
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

    optimizer = optim.Adam(kvae.parameters())
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    kvae.load_state_dict(checkpoint["model_state_dict"])
    kvae.eval()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return kvae, optimizer, scheduler


def sequence_first_collate_fn(batch: list) -> torch.Tensor:
    """
    Collate function for the DataLoader.
    """
    data = torch.Tensor(np.stack(batch, axis=0))
    data = data.permute(1, 0, 2, 3, 4)
    return data


def evaluate(config: EvaluationConfig) -> None:
    # Setup device and dtype
    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    # Setup DataLoaders
    dataloader_test = setup_dataloader(config.data_root_dir, config.num_evaluations)

    checkpoint_file = os.path.join(config.checkpoint_dir, f"state-{config.epoch}.pth")

    # Setup Model, Optimizer, Scheduler
    kvae, optimizer, scheduler = setup_model_optimizer_scheduler(
        dataloader_test, checkpoint_file, config, device, dtype
    )

    # Setup Sample Control
    sample_control_test = SampleControl(
        encoder="mean", decoder="mean", state_transition="mean", observation="mean"
    )

    evaluate_model(
        dataloader=dataloader_test,
        kvae=kvae,
        sample_control=sample_control_test,
        checkpoint_dir=config.checkpoint_dir,
        epoch=config.epoch,
        device=device,
        dtype=dtype,
        use_wandb=False,
    )


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate a Kalman Variational Autoencoder"
    )

    env_group = parser.add_argument_group("Evaluation Environment settings")
    env_group.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for evaluation"
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
        help="Directory for loading model checkpoints",
    )
    env_group.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Specific epoch of the model to be evaluated",
    )
    env_group.add_argument(
        "--num_evaluations",
        type=int,
        default=1,
        help="Total number of separate evaluations to be conducted",
    )

    data_group = parser.add_argument_group("Data settings")
    data_group.add_argument(
        "--data_root_dir",
        type=str,
        default="evaluation_data",
        help="Root directory containing evaluation data",
    )
    data_group.add_argument(
        "--batch_operation",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Aggregation operation over batches",
    )
    data_group.add_argument(
        "--sequence_operation",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Aggregation operation over sequences",
    )
    data_group.add_argument(
        "--symmetrize_covariance",
        type=bool,
        default=False,
        help="Flag to determine if covariance matrices should be symmetrized",
    )

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

    args = parser.parse_args()

    return EvaluationConfig(
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        epoch=args.epoch,
        num_evaluations=args.num_evaluations,
        data_root_dir=args.data_root_dir,
        batch_operation=args.batch_operation,
        sequence_operation=args.sequence_operation,
        symmetrize_covariance=args.symmetrize_covariance,
        z_dim=args.z_dim,
        a_dim=args.a_dim,
        K=args.K,
        dynamics_parameter_network=args.dynamics_parameter_network,
        decoder_type=args.decoder_type,
    )


if __name__ == "__main__":
    config = parse_args()
    evaluate(config)
