from typing import Optional, Tuple

import torch

from config import Config


def compute_conv2d_output_size(input_size, kernel_size, stride, padding):
    h, w = input_size
    h_out = (h - kernel_size + 2 * padding) // stride + 1
    w_out = (w - kernel_size + 2 * padding) // stride + 1

    return h_out, w_out


def _validate_shape(value: torch.Tensor, expected_shape: Tuple[int, ...], name: str):
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, but got {value.shape}"
        )


def aggregate(
    value: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    config: Optional[Config] = None,
):
    _validate_shape(value, (sequence_length, batch_size), "value")
    if config is None:
        config = Config()
    if config.sequence_operation == "mean":
        value = value.mean(dim=0)
    elif config.sequence_operation == "sum":
        value = value.sum(dim=0)
    else:
        raise ValueError(f"Invalid sequence operation {config.sequence_operation}")
    if config.batch_operation == "mean":
        value = value.mean(dim=0)
    elif config.batch_operation == "sum":
        value = value.sum(dim=0)
    else:
        raise ValueError(f"Invalid batch operation {config.batch_operation}")
    return value
