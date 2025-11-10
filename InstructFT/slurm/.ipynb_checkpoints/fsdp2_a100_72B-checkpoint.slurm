#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/fsdp2_a100_72B_%j.out 
#SBATCH --error=logs/err/fsdp2_a100_72B_%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --hint=nomultithread 
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --account=sos@a100


## load module 
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0


## launch script on every task 
set -x
time srun python -u fsdp2.py --test --model Qwen/Qwen2.5-72B-Instruct --fsdp-checkpointing 7/8 --bsz 2 --grad-acc 1 --compile
date
