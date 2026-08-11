#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=smoke_%j.out

/home/ps306/.conda/envs/lungtrain/bin/python smoke.py
