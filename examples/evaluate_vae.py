import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from moviepy.editor import ImageSequenceClip
from torch.utils.data import DataLoader

from kvae.bouncing_ball.dataloaders.bouncing_data import BouncingBallDataLoader
from kvae.config import EvaluationConfig
from kvae.kalman_vae import KalmanVariationalAutoencoder
from kvae.sample_control import SampleControl


def write_input_output_videos(
    data: torch.Tensor,
    kvae: KalmanVariationalAutoencoder,
    sample_control: SampleControl,
    input_filename: str,
    output_filename: str,
    channel: int = 0,
    fps: int = 10,
):
    kvae.eval()

    seq_length, batch_size, image_channels, *image_size = data.shape
    if sample_control.encoder == "mean":
        encoded_data = kvae.encoder(data.view(-1, image_channels, *image_size)).mean
    elif sample_control.encoder == "sample":
        encoded_data = kvae.encoder(data.view(-1, image_channels, *image_size)).sample
    else:
        raise ValueError(
            f"Invalid sample control for encoder: {sample_control.encoder}"
        )

    output_images = (
        kvae.decoder(encoded_data)
        .mean.view(seq_length, batch_size, image_channels, *image_size)
        .cpu()
        .float()
        .detach()
        .numpy()
    )

    idx = 0
    os.makedirs(os.path.dirname(input_filename), exist_ok=True)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    input_frame_paths = []
    output_frame_paths = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for step, (image) in enumerate((data)):
            image = (image > 0.5).cpu().float().detach().numpy()

            fig, ax = plt.subplots(figsize=(4, 4))
            black_grad = LinearSegmentedColormap.from_list(
                "black_grad", [(1, 1, 1), (0, 0, 0)], N=256
            )
            ax.imshow(
                image[idx][channel],
                vmin=0,
                vmax=1,
                aspect="equal",
                cmap=black_grad,
            )
            input_frame_path = os.path.join(tmpdirname, f"input_frame_{step:04d}.png")
            fig.savefig(input_frame_path, dpi=100)
            input_frame_paths.append(input_frame_path)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(
                output_images[step, idx, 0],
                vmin=0,
                vmax=1,
                aspect="equal",
                cmap=black_grad,
            )
            output_frame_path = os.path.join(tmpdirname, f"output_frame_{step:04d}.png")
            fig.savefig(output_frame_path, dpi=100)
            output_frame_paths.append(output_frame_path)
            plt.close(fig)

        input_video_clip = ImageSequenceClip(input_frame_paths, fps=fps)
        output_video_clip = ImageSequenceClip(output_frame_paths, fps=fps)
        input_video_clip.write_videofile(input_filename, codec="libx264")
        output_video_clip.write_videofile(output_filename, codec="libx264")


def write_videos(
    dataloader: torch.utils.data.DataLoader,
    kvae: KalmanVariationalAutoencoder,
    sample_control: SampleControl,
    checkpoint_dir: str,
    epoch: int,
    dtype: torch.dtype,
    device: torch.device,
    num_videos: int,
):
    batch = next(iter(dataloader))
    batch = (batch > 0.5).to(dtype=dtype, device=device)
    seq_length, batch_size, image_channels, *image_size = batch.shape

    video_directory = os.path.join(checkpoint_dir, "videos", "vae", f"epoch_{epoch}")

    video_count = 0
    for data_idx in range(batch_size):
        if video_count >= num_videos:
            break
        input_video_path = os.path.join(video_directory, f"input_idx_{data_idx}.mp4")
        output_video_path = os.path.join(video_directory, f"output_idx_{data_idx}.mp4")
        write_input_output_videos(
            data=batch[:, data_idx : data_idx + 1],
            kvae=kvae,
            input_filename=input_video_path,
            output_filename=output_video_path,
            channel=0,
            fps=10,
            sample_control=sample_control,
        )
        video_count += 1


def setup_dataloader(root_dir: str, batch_size: int) -> DataLoader:
    dataloader_test = DataLoader(
        BouncingBallDataLoader(root_dir=f"{root_dir}/test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sequence_first_collate_fn,
    )

    return dataloader_test


def setup_model(
    dataloader_test: DataLoader,
    checkpoint_file: str,
    config: EvaluationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> KalmanVariationalAutoencoder:
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
            init_transition_reg_weight=0.1,
            init_observation_reg_weight=0.1,
            init_noise_scale=0.1,
            learn_noise_covariance=True,
        )
        .to(dtype=dtype)
        .to(device)
    )

    checkpoint = torch.load(checkpoint_file, map_location=device)
    kvae.load_state_dict(checkpoint["model_state_dict"])
    kvae.eval()

    return kvae


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
    kvae = setup_model(dataloader_test, checkpoint_file, config, device, dtype)

    # Setup Sample Control
    sample_control_test = SampleControl(
        encoder="mean", decoder="mean", state_transition="mean", observation="mean"
    )

    write_videos(
        dataloader=dataloader_test,
        kvae=kvae,
        sample_control=sample_control_test,
        checkpoint_dir=config.checkpoint_dir,
        epoch=config.epoch,
        num_videos=config.num_videos,
        device=device,
        dtype=dtype,
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
        default=5,
        help="Total number of data used for evaluation",
    )
    env_group.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of videos to be generated for each evaluation. Must be less than or equal to num_evaluations",
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

    if args.num_videos > args.num_evaluations:
        raise ValueError(
            f"Number of videos to be generated ({args.num_videos}) must be less than or equal to num_evaluations ({args.num_evaluations})"
        )

    return EvaluationConfig(
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        epoch=args.epoch,
        num_evaluations=args.num_evaluations,
        num_videos=args.num_videos,
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
