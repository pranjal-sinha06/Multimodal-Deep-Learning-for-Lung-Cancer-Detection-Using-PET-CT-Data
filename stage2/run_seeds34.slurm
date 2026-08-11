#!/bin/bash
#SBATCH --job-name=seeds34
#SBATCH --array=0-5
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/seeds34_%A_%a.out
#SBATCH --error=logs/seeds34_%A_%a.err

# Extends ALL THREE arms from 3 seeds to 5, matching the Stage 2 sweep protocol.
# Every arm is extended, not just one, so no arm gets extra seeds selectively.
#
# Writes to *_nested_s3 and *_nested_s4 only. Seeds 0, 1 and 2 are untouched.

PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs

SCRIPTS=(train_ct_cv.py train_petct_cv.py train_sc_cv.py \
         train_ct_cv.py train_petct_cv.py train_sc_cv.py)
SEEDS=(3 3 3 4 4 4)

S=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
D=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "running $S with seed $D"
$PY $S --seed $D
