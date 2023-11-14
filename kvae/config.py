from typing import Literal, NamedTuple


class TrainingConfig(NamedTuple):
    """
    Configuration for the training phase of the experiment.
    """

    # Environment Settings - Parameters related to the experimental environment
    device: str  # Device to use for training
    dtype: Literal["float", "double"]  # Data type for tensors
    checkpoint_dir: str  # Directory to save checkpoints
    project_name: str  # Name of the project
    name: str  # Name of the experiment
    evaluation_interval: int  # Number of epochs between evaluations

    # Data Settings - Parameters related to data
    data_root_dir: str  # Root directory of the data
    batch_size: int  # Batch size
    batch_operation: Literal["mean", "sum"]  # Operation per batch
    sequence_operation: Literal["mean", "sum"]  # Operation per sequence

    # Model Settings - Parameters related to the configuration of the model
    dynamics_parameter_network: Literal[
        "mlp", "lstm"
    ]  # Type of dynamics parameter network
    a_dim: int  # Dimension of encoded space a
    z_dim: int  # Dimension of latent space z
    K: int  # Number of matrices for weighted average in observation and transition matrices
    decoder_type: Literal["bernoulli", "gaussian"]  # Type of decoder
    reconst_weight: float  # Weight for reconstruction loss
    regularization_weight: float  # Weight for regularization loss
    kalman_weight: float  # Weight for Kalman loss
    kl_weight: float  # Weight for KL loss (when using VAE)
    symmetrize_covariance: bool  # Whether to symmetrize the covariance matrix

    # Training Settings - Parameters related to training
    epochs: int  # Number of training epochs
    warmup_epochs: int  # Number of epochs to train without updating dynamics parameter network
    learning_rate: float  # Learning rate
    learning_rate_decay: float  # Learning rate decay
    scheduler_step: int  # Number of epochs between scheduler steps
    burn_in: int  # Number of burn-in steps


class EvaluationConfig(NamedTuple):
    """
    Configuration for the evaluation phase of the experiment.
    """

    # Evaluation Environment Settings - Parameters related to the setup for evaluation
    device: str  # Device to use for evaluation (e.g., 'cuda', 'cpu')
    dtype: Literal[
        "float", "double"
    ]  # Data type precision for tensors during evaluation
    checkpoint_dir: str  # Directory for loading model checkpoints
    epoch: int  # Specific epoch of the model to be evaluated
    num_evaluations: int  # Total number of separate evaluations to be conducted
    dynamics_parameter_network: Literal[
        "mlp", "lstm"
    ]  # Type of dynamics parameter network
    a_dim: int  # Dimension of encoded space a
    z_dim: int  # Dimension of latent space z
    K: int  # Number of matrices for weighted average in observation and transition matrices
    decoder_type: Literal["bernoulli", "gaussian"]  # Type of decoder

    # Data Settings - Parameters concerning the data used in evaluation
    data_root_dir: str  # Root directory containing evaluation data
    batch_operation: Literal[
        "mean", "sum"
    ]  # Aggregation operation over batches (e.g., for loss calculation)
    sequence_operation: Literal[
        "mean", "sum"
    ]  # Aggregation operation over sequences (e.g., in sequence modeling tasks)
    symmetrize_covariance: bool  # Flag to determine if covariance matrices should be s
