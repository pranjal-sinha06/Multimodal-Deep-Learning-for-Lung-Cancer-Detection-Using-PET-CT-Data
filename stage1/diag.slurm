#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=diag_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/python diag.py
