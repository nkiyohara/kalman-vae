from typing import NamedTuple
from typing_extensions import Literal


class SampleControl(NamedTuple):
    encoder: Literal["sample", "mean"] = "sample"
    decoder: Literal["sample", "mean"] = "sample"
    state_transition: Literal["sample", "mean"] = "sample"
    observation: Literal["sample", "mean"] = "sample"

    @classmethod
    def training_defaults(cls):
        return cls(encoder="sample", decoder="sample", state_transition="sample", observation="sample")

    @classmethod
    def evaluation_defaults(cls):
        return cls(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
