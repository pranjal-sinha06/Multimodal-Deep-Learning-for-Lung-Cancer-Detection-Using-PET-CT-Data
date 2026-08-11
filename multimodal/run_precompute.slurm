#!/bin/bash
#SBATCH --job-name=suvpre
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/suvpre_%j.out
#SBATCH --error=logs/suvpre_%j.err

# Builds suv_slices.npy + suv_index.csv. CPU only, run once.
PY=/home/ps306/.conda/envs/lungtrain/bin/python
mkdir -p logs
cd /sharedscratch/ps306/lung
$PY precompute_suv.py
$PY verify_suv_cache.py
