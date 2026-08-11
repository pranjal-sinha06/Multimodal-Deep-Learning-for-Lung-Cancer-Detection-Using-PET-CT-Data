import os, json, argparse
import numpy as np, pandas as pd, torch
from collections import defaultdict
from ultralytics import YOLO
from torchvision.models import resnet50
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the EXACT verified Stage 2 crop/window/normalise contract (no reimplementation)
from stage2_dataset import (window_channels, crop_box, load_hu_cached,
                            S2_MEAN, S2_STD, CROP_SIZE, MARGIN, ROOT)
from torchvision.transforms import v2

YOLO_CKPT = "/sharedscratch/ps306/lung/runs/detect/yolo_runs/full/weights/best.pt"
S2_CKPT = "stage2_runs/run1/best.pth"
STAGE1_MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
STAGE2_MANIFEST = os.path.join(ROOT, "stage2_crops_manifest.csv")
OUT = "pipeline_eval"
CLASSES = ["A", "B", "G"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.5)   # Stage 1 confidence threshold
    ap.add_argument("--max-patients", type=int, default=0)  # smoke: limit patients
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")

    mean = torch.tensor(S2_MEAN).view(3, 1, 1)
    std = torch.tensor(S2_STD).view(3, 1, 1)
    resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)

    # ground-truth subtype per patient, from Stage 2 crop manifest (test split)
    s2 = pd.read_csv(STAGE2_MANIFEST)
    s2t = s2[s2.split == "test"]
    true_sub = s2t.groupby("patient").subtype.first().to_dict()
    lab_idx = {"A": 0, "B": 1, "G": 2}

    # all test slices (positive + negative) for the honest cascade
    s1 = pd.read_csv(STAGE1_MANIFEST)
    test = s1[s1.split == "test"].reset_index(drop=True)
    patients = sorted(test.patient.unique())
    if args.max_patients:
        patients = patients[:args.max_patients]
        test = test[test.patient.isin(patients)].reset_index(drop=True)
    print(f"cascade over {len(patients)} test patients, {len(test)} slices, score>={args.score}")

    detector = YOLO(YOLO_CKPT)

    clf = resnet50(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 3)
    ck = torch.load(S2_CKPT, map_location=device)
    clf.load_state_dict(ck["model"]); clf.to(device).eval()

    # accumulate softmax probs per patient across all detected crops
    prob_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    n_boxes = defaultdict(int)

    for _, r in test.iterrows():
        hu = load_hu_cached(r.sop)
        yolo_img = (window_channels(hu) * 255).astype(np.uint8).transpose(1, 2, 0)  # native res, like to_yolo.py
        res = detector.predict(yolo_img, conf=args.score, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()  # pixel xyxy in HU-cache coordinates
        crops = []
        for b in boxes:
            crop = crop_box(hu, b.tolist(), margin=MARGIN, jitter=0.0)  # exact Stage 2 crop, no jitter
            if crop.size == 0:
                continue
            img = torch.from_numpy(window_channels(crop))  # 3,h,w, is_train=False path
            img = resize(img)
            img = (img - mean) / std
            crops.append(img)
        if not crops:
            continue
        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            probs = torch.softmax(clf(batch), dim=1).cpu().numpy()
        prob_sum[r.patient] += probs.sum(axis=0)
        n_boxes[r.patient] += len(probs)

    # patient-level aggregation: mean prob -> argmax
    detected = sorted(prob_sum.keys())
    n_total = len(patients)
    n_detected = len(detected)
    print(f"\nStage 1 detection recall: {n_detected}/{n_total} patients = {n_detected/n_total:.3f}")
    print(f"(a patient is 'detected' if >=1 box scored >= {args.score} on any of its slices)")

    inv = {0: "A", 1: "B", 2: "G"}
    y_true, y_pred = [], []
    print("\nper-patient (patient, true, pred, correct):")
    for p in detected:
        mean_prob = prob_sum[p] / n_boxes[p]
        t = lab_idx[true_sub[p]]; q = int(mean_prob.argmax())
        y_true.append(t); y_pred.append(q)
        print(f"  {p}  true={inv[t]}  pred={inv[q]}  {'OK' if t==q else 'X'}")

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1c = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n[end-to-end, detected patients only]  n={n_detected}  acc {acc:.3f}  macro-F1 {mf1:.3f}")
    for i, c in enumerate(CLASSES):
        print(f"   {c}: F1 {f1c[i]:.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"End-to-end (YOLO->ResNet)\nmacro-F1 {mf1:.3f}, recall {n_detected}/{n_total}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cm_pipeline_lesion.png"), dpi=120)
    print(f"\nconfusion matrix -> {OUT}/cm_pipeline_lesion.png")

    # save a small summary
    with open(os.path.join(OUT, "pipeline_summary.txt"), "w") as f:
        f.write(f"score_threshold={args.score}\n")
        f.write(f"detection_recall={n_detected}/{n_total}={n_detected/n_total:.3f}\n")
        f.write(f"subtype_acc_detected={acc:.3f}\nsubtype_macroF1_detected={mf1:.3f}\n")
        for i, c in enumerate(CLASSES):
            f.write(f"F1_{c}={f1c[i]:.3f}\n")
    print(f"summary -> {OUT}/pipeline_summary.txt")

if __name__ == "__main__":
    main()
