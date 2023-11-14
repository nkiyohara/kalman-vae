#!/bin/bash

# Usage function to display help for the script
usage() {
    echo "Usage: $0 --checkpoint_dir CHECKPOINT_DIRECTORY --epoch EPOCH"
    echo "  --checkpoint_dir CHECKPOINT_DIRECTORY    Specify the checkpoint directory"
    echo "  --epoch EPOCH                            Specify the epoch for evaluation"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --checkpoint_dir)
            checkpoint_directory="$2"
            shift # past argument
            shift # past value
            ;;
        --epoch)
            epoch="$2"
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            usage
            exit
            ;;
        *)
            # unknown option
            usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$checkpoint_directory" ] || [ -z "$epoch" ]; then
    echo "Error: Both checkpoint_dir and epoch arguments are required"
    usage
fi

# Create a directory with the provided checkpoint directory name
mkdir -p "$checkpoint_directory"

# Run the Python script with all arguments set from the argparse configuration
python evaluate.py \
    --device "cuda:0" \
    --dtype "float64" \
    --checkpoint_dir "$checkpoint_directory" \
    --epoch $epoch \
    --num_evaluations 5 \
    --data_root_dir "bouncing_ball/datasets/bouncing-ball" \
    --batch_operation "mean" \
    --sequence_operation "mean" \
    --symmetrize_covariance True \
    --z_dim 4 \
    --a_dim 2 \
    --K 3 \
    --decoder_type "bernoulli"
