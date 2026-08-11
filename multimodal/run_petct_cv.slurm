#!/bin/bash
#SBATCH --job-name=petctcv
#SBATCH --array=0-2
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/petctcv_%A_%a.out
#SBATCH --error=logs/petctcv_%A_%a.err


PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs
$PY train_petct_cv.py --seed ${SLURM_ARRAY_TASK_ID}
