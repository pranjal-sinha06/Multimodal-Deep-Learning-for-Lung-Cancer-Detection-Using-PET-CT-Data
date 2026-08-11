#!/bin/bash
#SBATCH --job-name=s2sweep
#SBATCH --array=0-24
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/s2sweep_%A_%a.out
#SBATCH --error=logs/s2sweep_%A_%a.err

# 5 runs x 5 seeds = 25 tasks.
#
# run3 trains via train_stage2_run3.py, which imports stage2_dataset_run3.py
# (stronger augmentation). That import is deliberate: do not rewire it.
#
# Each task writes to stage2_runs/<run>_s<seed>/, so the original
# stage2_runs/run1..run5 are never touched.
#
# Absolute python path: conda activation fails silently in batch shells.
PY=/home/ps306/.conda/envs/lungtrain/bin/python

mkdir -p logs

RUNS=(run1 run2 run3 run4 run5)
SCRIPTS=(train_stage2.py train_stage2_run2.py train_stage2_run3.py train_stage2_run4.py train_stage2_run5.py)

T=${SLURM_ARRAY_TASK_ID}
R=$((T / 5))          # 0..4  -> which run
S=$((T % 5))          # 0..4  -> which seed

RUN=${RUNS[$R]}
SCRIPT=${SCRIPTS[$R]}
TAG=${RUN}_s${S}

echo "=== task ${T}: ${SCRIPT} --tag ${TAG} --seed ${S} on $(hostname) ==="
$PY ${SCRIPT} --tag ${TAG} --seed ${S}
