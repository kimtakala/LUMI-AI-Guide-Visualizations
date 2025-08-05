#!/bin/bash
#SBATCH --account=project_462000008  # project account to bill
#SBATCH --partition=dev-g            # other options are small-g and standard-g
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --gpus-per-node=8            # Number of GPUs per node
#SBATCH --ntasks-per-node=8          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 7 CPUs per GPU (8 total - 1 for system overhead)
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=30:00               # time limit
#SBATCH --output=slurm/16gpu-2node-run.sh-slurm-%j.out      # Tab-completion friendly output filename

# Load additional modules for stability
module load cray-pals

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# AMD RCCL specific settings for multi-node
export RCCL_SOCKET_IFNAME=hsn0
export RCCL_DEBUG=INFO
export RCCL_NCHANNELS=$SLURM_GPUS_PER_NODE
export RCCL_GROUPSIZE=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))

# Add NCCL timeout protection
export NCCL_SOCKET_TIMEOUT=30000
export NCCL_IB_TIMEOUT=50
export NCCL_TIMEOUT=1800000  # 30 minutes

# wandb authentication
export WANDB_API_KEY=00cb5e75213d7e0ec78d970c19fbd2808c9ab3db

# add path to additional packages in squasfs file
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export PYTHONPATH=$PYTHONPATH:../resources

# Let SLURM handle CPU binding automatically
srun singularity exec -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && cd ../resources && python ../9-Wandb-visualization/scripts/16gpu-2node-wandb.py"
