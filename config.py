from typing import Literal, NamedTuple


class Config(NamedTuple):
    device: str
    batch_size: int
    dtype: Literal["float", "double"]
    epochs: int
    learning_rate: float
    learning_rate_decay: float
    scheduler_step: int
    a_dim: int
    z_dim: int
    K: int
    decoder_type: Literal["bernoulli", "gaussian"]
    data_root_dir: str
    checkpoint_dir: str
    project_name: str
    reconst_weight: float
    regularization_weight: float
    symmetrize_covariance: bool
    burn_in: int
    batch_operation: Literal["mean", "sum"]
    sequence_operation: Literal["mean", "sum"]
