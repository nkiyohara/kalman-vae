import os
from tempfile import TemporaryDirectory
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from moviepy.editor import ImageSequenceClip

import wandb
from kalman_vae import KalmanVariationalAutoencoder
from sample_control import SampleControl


def create_continuous_mask(seq_length, mask_length, batch_size, device, dtype):
    lst = [1.0] * seq_length
    start_index = (seq_length - mask_length) // 2
    for i in range(start_index, start_index + mask_length):
        lst[i] = 0.0
    return (
        torch.tensor(lst)
        .repeat(batch_size, 1)
        .transpose(0, 1)
        .to(device=device, dtype=dtype)
    )


def create_random_mask(seq_length, batch_size, mask_rate, device, dtype):
    mask = (torch.rand((seq_length, batch_size), device=device) >= mask_rate).to(
        device=device, dtype=dtype
    )
    mask[0] = 1
    mask[-1] = 1

    return mask


def evaluate(
    dataloader: torch.utils.data.DataLoader,
    kvae: KalmanVariationalAutoencoder,
    sample_control: SampleControl,
    dtype: torch.dtype,
    device: torch.device,
):
    random_masking = evaluate_random_masking(
        dataloader=dataloader,
        kvae=kvae,
        sample_control=sample_control,
        dtype=dtype,
        device=device,
    )
    continuous_masking, video_logs = evaluate_continuous_masking(
        dataloader=dataloader,
        kvae=kvae,
        sample_control=sample_control,
        dtype=dtype,
        device=device,
    )
    return random_masking, continuous_masking, video_logs


def evaluate_random_masking(
    dataloader: torch.utils.data.DataLoader,
    kvae: KalmanVariationalAutoencoder,
    sample_control: SampleControl,
    dtype: torch.dtype,
    device: torch.device,
):
    dropout_probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    filtering_incorrect_pixels = []
    smoothing_incorrect_pixels = []
    for dropout_probability in dropout_probabilities:
        batch = next(iter(dataloader))
        batch = (batch > 0.5).to(dtype=dtype, device=device)
        seq_length, batch_size, image_channels, *image_size = batch.shape
        mask = create_random_mask(
            seq_length=seq_length,
            batch_size=batch_size,
            mask_rate=dropout_probability,
            device=batch.device,
            dtype=batch.dtype,
        )
        _, info = kvae.elbo(
            xs=batch,
            observation_mask=mask,
            sample_control=sample_control,
        )
        filtered_images = kvae.decoder(info["filter_as"].view(-1, 2)).mean.view(
            seq_length, batch_size, image_channels, *image_size
        )
        filtering_incorrect_pixels.append(
            calculate_fraction_of_incorrect_pixels(batch, filtered_images)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )

        smoothed_images = kvae.decoder(info["as_resampled"].view(-1, 2)).mean.view(
            seq_length, batch_size, image_channels, *image_size
        )
        smoothing_incorrect_pixels.append(
            calculate_fraction_of_incorrect_pixels(batch, smoothed_images)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
    return pd.DataFrame(
        {
            "batch_id": 0,
            "dropout_probabilities": dropout_probabilities,
            "filtering_incorrect_pixels": filtering_incorrect_pixels,
            "smoothing_incorrect_pixels": smoothing_incorrect_pixels,
        }
    )


def evaluate_continuous_masking(
    dataloader: torch.utils.data.DataLoader,
    kvae: KalmanVariationalAutoencoder,
    sample_control: SampleControl,
    dtype: torch.dtype,
    device: torch.device,
    num_videos: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    batch = next(iter(dataloader))
    batch = (batch > 0.5).to(dtype=dtype, device=device)
    seq_length, batch_size, image_channels, *image_size = batch.shape

    mask_lengths = np.arange(2, seq_length - 4, 2).tolist()

    filtering_incorrect_pixels = []
    smoothing_incorrect_pixels = []
    video_logs = []

    video_count = 0

    for mask_length in mask_lengths:
        mask = create_continuous_mask(
            seq_length=seq_length,
            mask_length=mask_length,
            batch_size=batch_size,
            device=batch.device,
            dtype=batch.dtype,
        )
        _, info = kvae.elbo(
            xs=batch,
            observation_mask=mask,
            sample_control=sample_control,
        )
        filtered_images = kvae.decoder(info["filter_as"].view(-1, 2)).mean.view(
            seq_length, batch_size, image_channels, *image_size
        )
        filtering_incorrect_pixels.append(
            calculate_fraction_of_incorrect_pixels(batch, filtered_images)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        smoothed_images = kvae.decoder(info["as_resampled"].view(-1, 2)).mean.view(
            seq_length, batch_size, image_channels, *image_size
        )
        smoothing_incorrect_pixels.append(
            calculate_fraction_of_incorrect_pixels(batch, smoothed_images)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        for data_idx in range(batch_size):
            if video_count >= num_videos:
                break
            with TemporaryDirectory() as dname:
                video_path = os.path.join(dname, f"mask_length_{mask_length}.mp4")
                write_trajectory_video(
                    data=batch[:, data_idx : data_idx + 1],
                    kvae=kvae,
                    info=info,
                    observation_mask=mask,
                    filename=video_path,
                    channel=0,
                    fps=10,
                    sample_control=sample_control,
                )
                video = wandb.Video(
                    video_path,
                    f"idx_{data_idx}_mask_length_{mask_length}",
                    fps=10,
                    format="mp4",
                )
            video_count += 1
            log = {
                "batch_id": 0,
                "data_idx": data_idx,
                "mask_length": mask_length,
                "video": video,
            }
            video_logs.append(log)

    return (
        pd.DataFrame(
            {
                "batch_id": 0,
                "mask_lengths": mask_lengths,
                "filtering_incorrect_pixels": filtering_incorrect_pixels,
                "smoothing_incorrect_pixels": smoothing_incorrect_pixels,
            }
        ),
        video_logs,
    )


def calculate_fraction_of_incorrect_pixels(image, reconstructed_image):
    return (image != (reconstructed_image > 0.5)).float().mean()


def write_trajectory_video(
    data: torch.Tensor,
    kvae: KalmanVariationalAutoencoder,
    info: dict,
    observation_mask: torch.Tensor,
    sample_control: SampleControl,
    filename: str,
    channel: int = 0,
    fps: int = 10,
):
    kvae.eval()
    _, info = kvae.elbo(
        xs=data,
        observation_mask=observation_mask,
        sample_control=sample_control,
    )

    seq_length, batch_size, image_channels, *image_size = data.shape
    filtered_images = (
        kvae.decoder(info["filter_as"].view(-1, 2))
        .mean.view(seq_length, batch_size, image_channels, *image_size)
        .cpu()
        .float()
        .detach()
        .numpy()
    )
    smoothed_images = (
        kvae.decoder(info["as_resampled"].view(-1, 2))
        .mean.view(seq_length, batch_size, image_channels, *image_size)
        .cpu()
        .float()
        .detach()
        .numpy()
    )

    idx = 0
    cmap = plt.get_cmap("tab10")

    with TemporaryDirectory() as dname:
        png_files = []
        for step, (image) in enumerate((data)):
            fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
            fig.suptitle(f"$t = {step}$")

            image = (image > 0.5).cpu().float().detach().numpy()
            red_grad = LinearSegmentedColormap.from_list(
                "red_grad", [(1, 1, 1), (1, 0, 0)], N=256
            )
            black_grad = LinearSegmentedColormap.from_list(
                "black_grad", [(1, 1, 1), (0, 0, 0)], N=256
            )

            axes[0].imshow(
                image[idx][channel],
                vmin=0,
                vmax=1,
                cmap=red_grad,
                aspect="equal",
                alpha=0.5,
            )
            axes[0].imshow(
                filtered_images[step, idx, 0],
                vmin=0,
                vmax=1,
                cmap=black_grad,
                aspect="equal",
                alpha=0.5,
            )

            axes[1].imshow(
                image[idx][0], vmin=0, vmax=1, cmap=red_grad, aspect="equal", alpha=0.5
            )
            axes[1].imshow(
                smoothed_images[step, idx, 0],
                vmin=0,
                vmax=1,
                cmap=black_grad,
                aspect="equal",
                alpha=0.5,
            )

            axes[2].plot(
                info["as"][:, idx, 0].cpu().detach().numpy(),
                info["as"][:, idx, 1].cpu().detach().numpy(),
                ".-",
                color=cmap(0),
                label="Encoded",
            )

            axes[2].plot(
                info["filter_as"][:, idx, 0].cpu().detach().numpy(),
                info["filter_as"][:, idx, 1].cpu().detach().numpy(),
                ".-",
                color=cmap(1),
                label="Filtered",
            )

            axes[2].plot(
                info["as_resampled"][:, idx, 0].cpu().detach().numpy(),
                info["as_resampled"][:, idx, 1].cpu().detach().numpy(),
                ".-",
                color=cmap(2),
                label="Smoothed",
            )

            for key in ("as", "filter_as", "as_resampled"):
                axes[2].plot(
                    info[key][step, idx, 0].cpu().detach().numpy(),
                    info[key][step, idx, 1].cpu().detach().numpy(),
                    "o",
                    markersize=8,
                    color="red",
                    linestyle="none",
                    zorder=10,
                )

            axes[2].plot(
                (observation_mask.unsqueeze(-1) * info["as"])[:, idx, 0]
                .cpu()
                .detach()
                .numpy(),
                (observation_mask.unsqueeze(-1) * info["as"])[:, idx, 1]
                .cpu()
                .detach()
                .numpy(),
                "s",
                color="black",
                label="Observed",
            )

            axes[0].set_title("from filtered $\\mathbf{z}$")
            axes[1].set_title("from smoothed $\\mathbf{z}$")
            axes[2].set_title("$\\mathbf{a}$ space")
            axes[2].legend()
            axes[2].grid()

            plt.tight_layout()
            png_file = os.path.join(dname, f"{step}.png")
            plt.savefig(png_file)
            png_files.append(png_file)
            plt.close()
        clip = ImageSequenceClip(png_files, fps=fps)
        clip.write_videofile(filename, codec="libx264")
