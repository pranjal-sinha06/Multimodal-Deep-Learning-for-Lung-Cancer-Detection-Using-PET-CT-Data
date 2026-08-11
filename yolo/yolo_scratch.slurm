#!/bin/bash
#SBATCH --job-name=yoloscratch
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=yoloscratch_%j.out

cd /sharedscratch/ps306/lung
/home/ps306/.conda/envs/lungtrain/bin/yolo detect train \
    model=yolov8s.yaml data=lung.yaml \
    epochs=100 imgsz=512 batch=16 workers=8 patience=20 \
    pretrained=False project=yolo_runs name=scratch exist_ok=True
