#!/bin/bash
#SBATCH --job-name=s2sweepev
#SBATCH --array=0-24
#SBATCH --partition=gpu.A30
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/s2sweepev_%A_%a.out
#SBATCH --error=logs/s2sweepev_%A_%a.err

# Run AFTER sweep_train.slurm has finished.
#
# eval_stage2.py is already parameterised by --run, and reads
# stage2_runs/<run>/best.pth, writing figures/stage2/<run>/. Passing the swept
# tag works unmodified. This is what produces the PER-LESION numbers;
# metrics.csv only ever holds per-crop.
PY=/home/ps306/.conda/envs/lungtrain/bin/python

mkdir -p logs

RUNS=(run1 run2 run3 run4 run5)

T=${SLURM_ARRAY_TASK_ID}
R=$((T / 5))
S=$((T % 5))
TAG=${RUNS[$R]}_s${S}

if [ ! -f "stage2_runs/${TAG}/best.pth" ]; then
  echo "MISSING stage2_runs/${TAG}/best.pth -- did sweep_train.slurm finish?"
  exit 1
fi

echo "=== task ${T}: eval_stage2.py --run ${TAG} on $(hostname) ==="
$PY eval_stage2.py --run ${TAG}
