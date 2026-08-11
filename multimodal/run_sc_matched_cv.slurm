#!/bin/bash
#SBATCH --job-name=scmatch
#SBATCH --array=0-4
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/scmatch_%A_%a.out
#SBATCH --error=logs/scmatch_%A_%a.err



PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs
$PY train_sc_matched_cv.py --seed ${SLURM_ARRAY_TASK_ID}
