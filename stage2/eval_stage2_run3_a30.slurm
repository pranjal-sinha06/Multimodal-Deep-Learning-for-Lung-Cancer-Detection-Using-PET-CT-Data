#!/bin/bash
#SBATCH --job-name=s2eval
#SBATCH --partition=gpu.A30
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=s2eval_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/python eval_stage2_run3.py
