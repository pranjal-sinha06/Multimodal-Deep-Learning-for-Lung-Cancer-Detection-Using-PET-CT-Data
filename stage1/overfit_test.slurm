#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=overfit_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/python overfit_test.py
