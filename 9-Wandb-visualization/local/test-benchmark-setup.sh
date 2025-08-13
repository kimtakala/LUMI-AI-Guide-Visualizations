#!/bin/bash

# Quick test to validate the benchmark setup without full run
# This tests container access, dataset loading, and basic functionality

set -e

CONTAINER_PATH="/home/takalaki/Projects/LUMI-AI-Guide-COPY/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif"
WORKING_DIR="/home/takalaki/Projects/LUMI-AI-Guide-COPY"
HDF5_PATH="$WORKING_DIR/resources/train_images.hdf5"

echo "Testing benchmark setup..."

# Check files exist
echo "Checking container: $(ls -lh $CONTAINER_PATH | awk '{print $5}')"
echo "Checking HDF5 dataset: $(ls -lh $HDF5_PATH | awk '{print $5}')"

# Test container and basic imports
echo ""
echo "Testing container and imports..."
cd "$WORKING_DIR"

# a simple test script
cat > /tmp/test_imports.py << 'EOF'
import sys
sys.path.append('/home/takalaki/Projects/LUMI-AI-Guide-COPY/resources')

print('Testing imports...')
import torch
import torchvision
import numpy as np
from hdf5_dataset import HDF5Dataset
print(f' PyTorch: {torch.__version__}')
print(f' Torchvision: {torchvision.__version__}')
print(f' NumPy: {np.__version__}')
print(' HDF5Dataset imported successfully')

print('')
print('Testing dataset loading...')
with HDF5Dataset('/home/takalaki/Projects/LUMI-AI-Guide-COPY/resources/train_images.hdf5') as dataset:
    print(f' Dataset loaded: {len(dataset):,} samples')
    sample_image, sample_label = dataset[0]
    print(f' Sample shape: {sample_image.size}, Label: {sample_label}')

print('')
print('Testing model loading...')
from torchvision.models import vit_b_16
model = vit_b_16(weights='DEFAULT')
total_params = sum(p.numel() for p in model.parameters())
print(f'ViT-B-16 loaded: {total_params:,} parameters')

print('')
print('All tests passed! Benchmark should work correctly.')
EOF

singularity exec \
--bind "$WORKING_DIR:$WORKING_DIR" \
--bind /tmp:/tmp \
"$CONTAINER_PATH" \
bash -c '$WITH_CONDA && python3 /tmp/test_imports.py'

echo ""
echo "Setup validation complete!"
echo ""
echo "To run the full benchmark:"
echo "  export WANDB_API_KEY=your_key"
echo "  ./run-laptop-benchmark.sh"
