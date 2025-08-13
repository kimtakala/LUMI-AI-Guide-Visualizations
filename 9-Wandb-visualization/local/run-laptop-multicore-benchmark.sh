#!/bin/bash

set -e

# configuration
CONTAINER_PATH="/LUMI-AI-Guide-Visualizations/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif"
SCRIPT_DIR="/LUMI-AI-Guide-Visualizations/9-Wandb-visualization/local"
WORKING_DIR="/LUMI-AI-Guide-Visualizations"
export WANDB_API_KEY=insert_api_key

# Run in container with proper environment
echo "Executing multicore benchmark in container..."
cd "$WORKING_DIR"

singularity exec \
--bind "$WORKING_DIR:$WORKING_DIR" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env WANDB_MODE="${WANDB_MODE:-online}" \
--env OMP_NUM_THREADS=$(nproc) \
--env MKL_NUM_THREADS=$(nproc) \
"$CONTAINER_PATH" \
bash -c '$WITH_CONDA && cd '"$WORKING_DIR"' && python3 '"$SCRIPT_DIR"'/wandb_laptop_multicore_benchmark.py'

echo ""
echo "=================================================="
echo "MULTICORE BENCHMARK COMPLETED"
echo "=================================================="
echo "Check your Wandb dashboard for results:"
echo "  Project: LUMI-Laptop-Benchmark"
echo ""
echo "The script measured timing on a subset with multicore optimization"
echo "and extrapolated the time needed to train on the full dataset."
echo ""
echo "Compare these extrapolated times with actual LUMI runs"
echo "to see the performance advantage of distributed training."
