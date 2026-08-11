from ultralytics import YOLO

# from scratch = build from architecture yaml (random init), NOT yolov8s.pt
model = YOLO("yolov8s.yaml")

model.train(
    data="/sharedscratch/ps306/lung/lung.yaml",
    epochs=100,
    batch=16,
    imgsz=512,
    optimizer="Adam",
    lr0=0.001,   # paper is silent on lr; Adam-appropriate default (0.01 is SGD-tuned)
    seed=0,      # reproducibility
    device=0,
    project="/sharedscratch/ps306/lung/runs",
    name="yolo_scratch_run1",
)
