#!/bin/bash

# Get current time and format it for experiment name
current_time=$(date +"%Y%m%d_%H%M%S")
experiment_name="experiment_$current_time"

# Create a directory with the experiment name inside the checkpoints directory
checkpoint_directory="checkpoints/$experiment_name"
mkdir -p "$checkpoint_directory"

# Run the Python script with all arguments explicitly set
python train.py \
    --data_root_dir "../kvae/bouncing_ball/datasets/bouncing_ball" \
    --batch_size 128 \
    --z_dim 4 \
    --a_dim 2 \
    --K 3 \
    --dynamics_parameter_network "lstm" \
    --decoder_type "bernoulli" \
    --reconst_weight 1.0 \
    --regularization_weight 0.3 \
    --kalman_weight 1.0 \
    --kl_weight 0.0 \
    --initial_noise_scale 1.0 \
    --init_transition_reg_weight 0.9 \
    --init_observation_reg_weight 0.1 \
    --epochs 100 \
    --warmup_epochs 10 \
    --learning_rate 0.007 \
    --learning_rate_decay 0.8 \
    --burn_in 3 \
    --batch_operation "mean" \
    --sequence_operation "mean" \
    --scheduler_step 20 \
    --device "cuda:0" \
    --dtype "float64" \
    --checkpoint_dir "$checkpoint_directory" \
    --name "$experiment_name" \
    --evaluation_interval 0 \
