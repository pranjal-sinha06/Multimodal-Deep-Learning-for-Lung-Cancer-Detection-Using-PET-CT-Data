#!/bin/bash
#SBATCH --job-name=pipefrcnn
#SBATCH --partition=gpu.A30
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=pipefrcnn_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/python eval_pipeline_frcnn.py "$@"
