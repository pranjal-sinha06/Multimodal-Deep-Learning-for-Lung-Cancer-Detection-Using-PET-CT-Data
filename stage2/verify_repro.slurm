#!/bin/bash
#SBATCH --job-name=s2repro
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/s2repro_%j.out
#SBATCH --error=logs/s2repro_%j.err

# RUN THIS FIRST, BEFORE sweep_train.slurm.
#
# Trains seed 0 twice on a truncated loop (2 epochs, 10 train iters, 5 val
# iters) and checks the two metrics.csv files match exactly. Takes minutes,
# not hours, and proves determinism before 25 full jobs are committed.
#
# If this fails, the sweep is measuring plumbing noise, not seeds.
PY=/home/ps306/.conda/envs/lungtrain/bin/python

mkdir -p logs
rm -rf stage2_runs/repro_a stage2_runs/repro_b

FLAGS="--seed 0 --epochs 2 --max-train-iters 10 --max-val-iters 5"

echo "=== pass A ==="
$PY train_stage2.py --tag repro_a $FLAGS
echo "=== pass B ==="
$PY train_stage2.py --tag repro_b $FLAGS

echo
echo "=== comparing ==="
$PY - << 'EOF'
import sys, pandas as pd

a = pd.read_csv("stage2_runs/repro_a/metrics.csv")
b = pd.read_csv("stage2_runs/repro_b/metrics.csv")

# time_s is wall-clock and always differs. Everything else must match bitwise.
cols = [c for c in a.columns if c != "time_s"]
a, b = a[cols], b[cols]

if a.equals(b):
    print("\n  REPRODUCIBLE: both passes identical across", list(cols))
    sys.exit(0)

print("\n  NOT REPRODUCIBLE. Divergence:\n")
for c in cols:
    if not a[c].equals(b[c]):
        d = (a[c] - b[c]).abs().max() if a[c].dtype.kind == "f" else "n/a"
        print(f"    {c:<16} max|diff| = {d}")
print("""
  Likely causes, in order:
    1. A dataset still calls np.random.default_rng() with no argument.
    2. cudnn.benchmark is still True somewhere.
    3. An op has no deterministic CUDA kernel. Diagnose with:
         torch.use_deterministic_algorithms(True, warn_only=True)
       plus CUBLAS_WORKSPACE_CONFIG=:4096:8
""")
sys.exit(1)
EOF
