#!/bin/bash
#SBATCH --account=project_462000008  # project account to bill
#SBATCH --partition=small-g          # other options are dev-g and standard-g
#SBATCH --gpus-per-node=8            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=8          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=2:00:00                 # time limit

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../resources/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

# add path to additional packages in squasfs file
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
# bind squashfs file into container and run python script inside container 
singularity exec -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER python visualtransformer.py
