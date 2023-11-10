from typing import Literal, NamedTuple


class Config(NamedTuple):
    """Configuration for the experiment"""

    # Environment Settings - Parameters related to the experimental environment
    device: str  # Device to use for training
    dtype: Literal["float", "double"]  # Data type for tensors
    checkpoint_dir: str  # Directory to save checkpoints
    project_name: str  # Name of the project
    name: str  # Name of the experiment

    # Data Settings - Parameters related to data
    data_root_dir: str  # Root directory of the data
    batch_size: int  # Batch size
    batch_operation: Literal["mean", "sum"]  # Operation per batch
    sequence_operation: Literal["mean", "sum"]  # Operation per sequence

    # Model Settings - Parameters related to the configuration of the model
    a_dim: int  # Dimension of encoded space a
    z_dim: int  # Dimension of latent space z
    K: int  # Number of matrices for weighted average in observation and transition matrices
    decoder_type: Literal["bernoulli", "gaussian"]  # Type of decoder
    reconst_weight: float  # Weight for reconstruction loss
    regularization_weight: float  # Weight for regularization loss
    symmetrize_covariance: bool  # Whether to symmetrize the covariance matrix

    # Training Settings - Parameters related to training
    epochs: int  # Number of training epochs
    learning_rate: float  # Learning rate
    learning_rate_decay: float  # Learning rate decay
    scheduler_step: int  # Number of epochs between scheduler steps
    burn_in: int  # Number of burn-in steps
