#!/bin/bash
#SBATCH --job-name=train_tensorboard
#SBATCH --account=project_462000008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00                 # Adjust runtime as needed
#SBATCH --partition=small-g
#SBATCH --output=logs/%x_%j.out         # Stdout log file
#SBATCH --error=logs/%x_%j.err          # Stderr log file

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

export SIF=/projappl/project_462000008/takalaki/LUMI-AI-Guide/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

srun singularity exec --nv $SIF \
    bash -c '$WITH_CONDA && torchrun --nnodes=1 --nproc_per_node=8 tensorboard_ddp_visualtransformer.py'
