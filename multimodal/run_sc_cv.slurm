#!/bin/bash
#SBATCH --job-name=sccv
#SBATCH --array=0-2
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/sccv_%A_%a.out
#SBATCH --error=logs/sccv_%A_%a.err

# Secondary Capture arm: 3 seeds x 5-fold CV on the SAME 83 patients
# used by the CT-only and CT+PET arms.
PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs
$PY train_sc_cv.py --seed ${SLURM_ARRAY_TASK_ID}
