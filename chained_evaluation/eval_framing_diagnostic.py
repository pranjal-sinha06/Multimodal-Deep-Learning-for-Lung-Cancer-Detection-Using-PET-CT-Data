import os, json, argparse
import numpy as np, pandas as pd, torch
from collections import defaultdict
from ultralytics import YOLO
from torchvision.models import resnet50
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the EXACT verified Stage 2 crop/window/normalise contract
from stage2_dataset import (window_channels, crop_box, load_hu_cached,
                            S2_MEAN, S2_STD, CROP_SIZE, MARGIN, ROOT)
from torchvision.transforms import v2

YOLO_CKPT = "/sharedscratch/ps306/lung/runs/detect/yolo_runs/full/weights/best.pt"
S2_CKPT = "stage2_runs/run1/best.pth"
STAGE1_MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
STAGE2_MANIFEST = os.path.join(ROOT, "stage2_crops_manifest.csv")
OUT = "pipeline_eval"
CLASSES = ["A", "B", "G"]
LAB = {"A": 0, "B": 1, "G": 2}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.5)
    return ap.parse_args()

def classify_crops(clf, device, resize, mean, std, hu, boxes):
    """Crop each box from hu via the exact Stage 2 contract, return summed softmax probs and count."""
    crops = []
    for b in boxes:
        crop = crop_box(hu, list(b), margin=MARGIN, jitter=0.0)
        if crop.size == 0:
            continue
        img = torch.from_numpy(window_channels(crop))
        img = resize(img)
        img = (img - mean) / std
        crops.append(img)
    if not crops:
        return np.zeros(3), 0
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        probs = torch.softmax(clf(batch), dim=1).cpu().numpy()
    return probs.sum(axis=0), len(probs)

def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1c = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n[{name}]  n={len(y_true)}  acc {acc:.3f}  macro-F1 {mf1:.3f}")
    for i, c in enumerate(CLASSES):
        print(f"   {c}: F1 {f1c[i]:.3f}")
    return acc, mf1, f1c

def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    mean = torch.tensor(S2_MEAN).view(3, 1, 1)
    std = torch.tensor(S2_STD).view(3, 1, 1)
    resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)

    s2 = pd.read_csv(STAGE2_MANIFEST)
    true_sub = s2[s2.split == "test"].groupby("patient").subtype.first().to_dict()

    s1 = pd.read_csv(STAGE1_MANIFEST)
    test = s1[s1.split == "test"].reset_index(drop=True)
    patients = sorted(test.patient.unique())
    print(f"cascade over {len(patients)} test patients, {len(test)} slices, score>={args.score}")

    detector = YOLO(YOLO_CKPT)
    clf = resnet50(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 3)
    ck = torch.load(S2_CKPT, map_location=device)
    clf.load_state_dict(ck["model"]); clf.to(device).eval()

    yolo_sum = defaultdict(lambda: np.zeros(3)); yolo_n = defaultdict(int)
    gt_sum = defaultdict(lambda: np.zeros(3)); gt_n = defaultdict(int)

    for _, r in test.iterrows():
        hu = load_hu_cached(r.sop)
        # --- YOLO branch (all slices) ---
        yolo_img = (window_channels(hu) * 255).astype(np.uint8).transpose(1, 2, 0)
        res = detector.predict(yolo_img, conf=args.score, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            s, n = classify_crops(clf, device, resize, mean, std, hu,
                                  res.boxes.xyxy.cpu().numpy())
            yolo_sum[r.patient] += s; yolo_n[r.patient] += n
        # --- GT branch (positive slices only, boxes from manifest) ---
        if r.role == "positive":
            gt_boxes = json.loads(r.boxes)
            if gt_boxes:
                s, n = classify_crops(clf, device, resize, mean, std, hu, gt_boxes)
                gt_sum[r.patient] += s; gt_n[r.patient] += n

    detected = sorted(yolo_n.keys())
    n_det, n_tot = len(detected), len(patients)
    print(f"\nStage 1 detection recall: {n_det}/{n_tot} = {n_det/n_tot:.3f}")

    # Both metrics computed on the SAME detected patient set
    y_true = [LAB[true_sub[p]] for p in detected]
    yolo_pred = [int((yolo_sum[p] / yolo_n[p]).argmax()) for p in detected]
    gt_pred = [int((gt_sum[p] / gt_n[p]).argmax()) for p in detected if gt_n[p] > 0]
    y_true_gt = [LAB[true_sub[p]] for p in detected if gt_n[p] > 0]

    print("\n=== FRAMING DIAGNOSTIC: same detected patients, box source varied ===")
    report("YOLO boxes (pipeline)", y_true, yolo_pred)
    report("GT boxes (same patients)", y_true_gt, gt_pred)

    with open(os.path.join(OUT, "framing_diagnostic.txt"), "w") as f:
        _, mf1_y, _ = f1_score(y_true, yolo_pred, average="macro", zero_division=0), \
                      f1_score(y_true, yolo_pred, average="macro", zero_division=0), None
        mf1_g = f1_score(y_true_gt, gt_pred, average="macro", zero_division=0)
        f.write(f"detected={n_det}/{n_tot}\n")
        f.write(f"yolo_box_macroF1={f1_score(y_true, yolo_pred, average='macro', zero_division=0):.3f}\n")
        f.write(f"gt_box_macroF1={mf1_g:.3f}\n")
    print(f"\nsummary -> {OUT}/framing_diagnostic.txt")
    print("\nINTERPRETATION:")
    print("  if GT-box F1 >> YOLO-box F1  -> the drop is localisation FRAMING (box mismatch)")
    print("  if GT-box F1 ~ YOLO-box F1   -> the 25 detected patients are just a hard subset")

if __name__ == "__main__":
    main()
