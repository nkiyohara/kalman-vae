#!/bin/bash

# Get current time and format it for experiment name
current_time=$(date +"%Y%m%d_%H%M%S")
experiment_name="experiment_$current_time"

# Create a directory with the experiment name inside the checkpoints directory
checkpoint_directory="checkpoints/$experiment_name"
mkdir -p "$checkpoint_directory"

# Run the Python script with all arguments explicitly set
python train.py \
    --batch_size 128 \
    --z_dim 4 \
    --a_dim 2 \
    --K 3 \
    --decoder_type "bernoulli" \
    --reconst_weight 1.0 \
    --regularization_weight 0.3 \
    --kalman_weight 1.0 \
    --kl_weight 0.0 \
    --symmetrize_covariance True \
    --epochs 80 \
    --warmup_epochs 10 \
    --learning_rate 0.007 \
    --learning_rate_decay 0.8 \
    --burn_in 3 \
    --batch_operation "mean" \
    --sequence_operation "mean" \
    --scheduler_step 20 \
    --device "cuda:1" \
    --dtype "float64" \
    --checkpoint_dir "$checkpoint_directory" \
    --name "$experiment_name" \
    --evaluation_interval 1\
