#!/bin/bash
#SBATCH --job-name=evals2
#SBATCH --partition=gpu.A30
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=evals2_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/python eval_stage2.py "$@"
