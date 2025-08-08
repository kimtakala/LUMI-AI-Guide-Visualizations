#!/bin/bash
#SBATCH --account=project_462000008
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=30:00
#SBATCH --output=slurm/04gpu-1node-run.sh-slurm-%j.out

module load cray-pals
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

CONTAINER=../lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

export WANDB_API_KEY=00cb5e75213d7e0ec78d970c19fbd2808c9ab3db
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export PYTHONPATH=$PYTHONPATH:../resources

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

srun singularity exec -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python scripts/04gpu-1node-wandb.py"
