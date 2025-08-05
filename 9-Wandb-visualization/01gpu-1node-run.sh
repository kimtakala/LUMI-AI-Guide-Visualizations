#!/bin/bash
#SBATCH --account=project_462000008
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=30:00
#SBATCH --output=slurm/01gpu-1node-run.sh-slurm-%j.out

module load cray-pals
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

CONTAINER=../lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

export WANDB_API_KEY=00cb5e75213d7e0ec78d970c19fbd2808c9ab3db
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export PYTHONPATH=$PYTHONPATH:../resources

srun singularity exec -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER python scripts/01gpu-1node-wandb.py
