#!/bin/bash

# Fair laptop benchmarking script using the same container as LUMI
# This script processes a subset of the same HDF5 data and extrapolates timing

set -e

# Configuration
CONTAINER_PATH="/LUMI-AI-Guide-Visualizations/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif"
SCRIPT_DIR="/LUMI-AI-Guide-Visualizations/9-Wandb-visualization/local"
WORKING_DIR="/LUMI-AI-Guide-Visualizations"
export WANDB_API_KEY=insert_api_key


echo "=================================================="
echo "FAIR LAPTOP BENCHMARK vs LUMI"
echo "=================================================="
echo "Using the same dataset, model, and hyperparameters as LUMI"
echo "Processing subset and extrapolating full training time"
echo ""
echo "Container: $(basename $CONTAINER_PATH)"
echo "Working directory: $WORKING_DIR"
echo ""

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "ERROR: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Check if HDF5 data exists
HDF5_PATH="$WORKING_DIR/resources/train_images.hdf5"
if [ ! -f "$HDF5_PATH" ]; then
    echo "ERROR: HDF5 dataset not found at $HDF5_PATH"
    echo "Please ensure the dataset is available"
    exit 1
fi

# Display system info
echo "System Information:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  HDF5 dataset size: $(ls -lh $HDF5_PATH | awk '{print $5}')"
echo ""

# Check if Wandb is configured
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Please set your Wandb API key:"
    echo "  export WANDB_API_KEY=your_key_here"
    echo ""
    echo "You can find your API key at: https://wandb.ai/authorize"
    echo ""
    read -p "Continue without Wandb logging? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set WANDB_API_KEY and try again."
        exit 1
    fi
    export WANDB_MODE=offline
    echo "Running in offline mode..."
fi

echo "Starting fair laptop benchmark..."
echo "This will:"
echo "  1. Load the exact same HDF5 dataset as LUMI (4.6GB)"
echo "  2. Use the same ViT-B-16 model and hyperparameters"
echo "  3. Process 1% of the data to measure timing"
echo "  4. Extrapolate total training time for the full dataset"
echo "  5. Log results to Wandb for comparison with LUMI runs"
echo ""

# Run in container with proper environment
echo "Executing in container..."
cd "$WORKING_DIR"

singularity exec \
--bind "$WORKING_DIR:$WORKING_DIR" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env WANDB_MODE="${WANDB_MODE:-online}" \
--env OMP_NUM_THREADS=$(nproc) \
--env MKL_NUM_THREADS=$(nproc) \
"$CONTAINER_PATH" \
bash -c '$WITH_CONDA && cd '"$WORKING_DIR"' && python3 '"$SCRIPT_DIR"'/wandb_laptop_benchmark.py'

echo ""
echo "=================================================="
echo "BENCHMARK COMPLETED"
echo "=================================================="
echo "Check your Wandb dashboard for results:"
echo "  Project: LUMI-Laptop-Benchmark"
echo ""
echo "The script measured timing on a subset and extrapolated"
echo "the time needed to train on the full dataset."
echo ""
echo "Compare these extrapolated times with actual LUMI runs"
echo "to see the performance advantage of distributed training."
