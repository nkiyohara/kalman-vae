from typing import Literal, NamedTuple


class SampleControl(NamedTuple):
    encoder: Literal["sample", "mean"] = "sample"
    decoder: Literal["sample", "mean"] = "mean"
    state_transition: Literal["sample", "mean"] = "sample"
    observation: Literal["sample", "mean"] = "sample"
