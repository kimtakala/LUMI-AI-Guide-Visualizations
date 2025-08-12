#!/bin/b#SBATCH --time=08:00                 # time limit (optimized for 5 epochs)sh
#SBATCH --account=project_462000008  # project account to bill
#SBATCH --partition=dev-g            # other options are dev-g and standard-g
#SBATCH --nodes=16                   # Number of nodes
#SBATCH --gpus-per-node=8            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=8          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=10:00                 # time limit (increased for stability)
#SBATCH --output=slurm/slurm-%j-%x.out     # Custom output filename

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

# AMD RCCL specific settings for multi-node communication
export RCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export RCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN  # Reduce debug verbosity for large scale
export RCCL_DEBUG=WARN

# 16-node specific optimizations (128 GPUs!)
export NCCL_TREE_THRESHOLD=0
export NCCL_RING_THRESHOLD=0
export NCCL_ALGO=RING
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=64    # Even more channels for 128 GPUs
export NCCL_MIN_NCHANNELS=64
export NCCL_BUFFSIZE=16777216   # 16MB buffer for massive scale

# Additional settings for inter-node communication stability
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=60       # Shortest timeout for fastest detection

# Optimized timeouts for 16-node setup (fastest of all)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=180         # 3 minutes for 128 GPUs

# Advanced settings for extreme scale (essential for 16+ nodes)
export NCCL_LL_THRESHOLD=0
export NCCL_CROSS_NIC=1

# Advanced settings for extreme scale
export NCCL_LL_THRESHOLD=0      # Force low-latency algorithm
export NCCL_CROSS_NIC=1        # Enable cross-NIC communication

# Set up the CPU bind masks for optimal performance
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# wandb authentication
export WANDB_API_KEY=00cb5e75213d7e0ec78d970c19fbd2808c9ab3db

# add path to additional packages in squasfs file
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export PYTHONPATH=$PYTHONPATH:../resources

# bind squashfs file into container and run python script inside container 
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && cd ../resources && python ../9-Wandb-visualization/scripts/wandb_ddp_visualtransformer_16nodes.py"
