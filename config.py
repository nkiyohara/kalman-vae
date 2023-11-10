from typing import Literal, NamedTuple


class Config(NamedTuple):
    batch_operation: Literal["mean", "sum"] = "mean"
    sequence_operation: Literal["mean", "sum"] = "mean"
