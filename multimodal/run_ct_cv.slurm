#!/bin/bash
#SBATCH --job-name=ctcv
#SBATCH --array=0-2
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/ctcv_%A_%a.out
#SBATCH --error=logs/ctcv_%A_%a.err

# CT-only baseline on the PET cohort, 3 seeds x 5-fold CV.
# 3 array tasks = seeds 0,1,2. Each writes figures/petct/ct_only_s<seed>/.
PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs
$PY train_ct_cv.py --seed ${SLURM_ARRAY_TASK_ID}
