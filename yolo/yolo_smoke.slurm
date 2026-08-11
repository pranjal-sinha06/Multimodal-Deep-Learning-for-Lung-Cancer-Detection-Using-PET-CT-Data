#!/bin/bash
#SBATCH --job-name=yolosmoke
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=yolosmoke_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/yolo detect train \
    model=yolov8s.pt data=lung.yaml \
    epochs=2 imgsz=512 batch=16 fraction=0.05 \
    project=yolo_runs name=smoke exist_ok=True

