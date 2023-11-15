# Kalman Variational Autoencoder (K-VAE)

This repository contains the PyTorch implementation of the Kalman Variational Autoencoder (K-VAE), based on the paper "A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning" ([arXiv:1710.05741](https://arxiv.org/abs/1710.05741)). It is a framework for unsupervised learning using a disentangled recognition and nonlinear dynamics model.

## Getting Started

Follow these steps to set up the environment, install dependencies, and run the training and evaluation scripts.

### Prerequisites

- Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
- Python 3.11 is recommended.

### Clone the Repository

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/nkgvl/kalman-vae.git
cd kalman-vae
```

### Set Up Environment and Install Dependencies

Create a new Conda environment and install the required packages:

```bash
conda create --name kvae-env python=3.11
conda activate kvae-env

# Install dependencies from conda-forge
conda install -c conda-forge opencv pygame pymunk

# Install other specific dependencies
conda install matplotlib~=3.8.0 numpy~=1.26.0 pandas~=2.1.1 Pillow~=10.0.1 tqdm~=4.65.0 wandb~=0.15.12

# For PyTorch installation, refer to the official website to select the appropriate version and CUDA support
# Visit https://pytorch.org for instructions
```

### Install the K-VAE Package

Install the K-VAE package using pip:

```bash
pip install .
```

Modify and run the training script:

```bash
bash examples/run_training.sh
```

### Evaluation
After training, modify and run the evaluation script to assess performance:

```bash
bash examples/run_evaluation.sh --checkpoint_dir [YOUR_CHECKPOINT_DIR] --epoch [EPOCH_NUMBER]
```

Evaluation videos and performance tables will be saved in videos/ and tables/ directories under the specified checkpoint directory.

## Usage

After completing the setup, you can use the K-VAE model for your research and experiments. Feel free to modify the training and evaluation scripts to explore different configurations.

## Acknowledgments

- This implementation is inspired by the original paper ["A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning"](https://arxiv.org/abs/1710.05741) and its [original implementation](https://github.com/simonkamronn/kvae).
- The dataset generation uses code from [this repository](https://github.com/charlio23/bouncing-ball).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
