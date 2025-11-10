#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/Sif_fsdp2_h100_72B_%j.out 
#SBATCH --error=logs/err/Sif_fsdp2_h100_72B_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread 
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --account=sos@h100


## load module 
module purge
module load singularity

## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

export SINGULARITY_BINDPATH = "$WORK,$ALL_CCFRWORK,$DSDIR,$SCRATCH,$ALL_CCFRSCRATCH,$JOBSCRATCH"

## launch script on every task 
set -x
time srun singularity exec --nv --bind $SINGULARITY_BINDPATH \
    $SINGULARITY_ALLOWED_DIR/nemo2506_amd.sif \
    python fsdp2.py --test --model Qwen/Qwen2.5-72B-Instruct --fsdp-checkpointing 7/8 --bsz 2 --grad-acc 1 --compile
date
