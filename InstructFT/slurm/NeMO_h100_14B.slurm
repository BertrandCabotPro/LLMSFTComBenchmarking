#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/NeMO_h100_14B_%j.out 
#SBATCH --error=logs/err/NeMO_h100_14B_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread 
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --account=sos@h100

## load module 
module purge
module load arch/h100
module load nemo/2.4.0


## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
    

## launch script on every task 
set -x
time srun python -u nemo_megatron.py --model Qwen/Qwen2.5-14B-Instruct \
                             --devices 4 \
                             --num-nodes 16 \
                             --tp-size 4 \
                             --pp-size 4 \
                             --accumulate-grad-batches 8 \
                             --global-batch-size 128
date
