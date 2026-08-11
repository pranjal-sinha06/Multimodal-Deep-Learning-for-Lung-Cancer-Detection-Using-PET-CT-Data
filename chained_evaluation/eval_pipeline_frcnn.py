import os, json, argparse
import numpy as np, pandas as pd, torch
from collections import defaultdict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from stage2_dataset import (window_channels, crop_box, load_hu_cached,
                            S2_MEAN, S2_STD, CROP_SIZE, MARGIN, ROOT)
from torchvision.transforms import v2

FRCNN_CKPT = "/home/ps306/stage1_results/stage1_run3_best.pth"
S2_CKPT = "stage2_runs/run1/best.pth"
STAGE1_MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
STAGE2_MANIFEST = os.path.join(ROOT, "stage2_crops_manifest.csv")
OUT = "pipeline_eval"
CLASSES = ["A", "B", "G"]
LAB = {"A": 0, "B": 1, "G": 2}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--max-patients", type=int, default=0)
    return ap.parse_args()

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
    if args.max_patients:
        patients = patients[:args.max_patients]
        test = test[test.patient.isin(patients)].reset_index(drop=True)
    print(f"FRCNN cascade over {len(patients)} test patients, {len(test)} slices, score>={args.score}")

    detector = fasterrcnn_resnet50_fpn(weights=None)
    detector.roi_heads.box_predictor = FastRCNNPredictor(
        detector.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
    dck = torch.load(FRCNN_CKPT, map_location=device)
    detector.load_state_dict(dck["model"]); detector.to(device).eval()

    clf = resnet50(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 3)
    cck = torch.load(S2_CKPT, map_location=device)
    clf.load_state_dict(cck["model"]); clf.to(device).eval()

    prob_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    n_boxes = defaultdict(int)

    for _, r in test.iterrows():
        hu = load_hu_cached(r.sop)

        det_in = torch.from_numpy(window_channels(hu)).to(device)  # 3,H,W float [0,1]
        with torch.no_grad():
            pred = detector([det_in])[0]
        keep = pred["scores"] >= args.score
        boxes = pred["boxes"][keep].cpu().numpy()  # xyxy, HU-cache coordinates
        if len(boxes) == 0:
            continue
        crops = []
        for b in boxes:
            crop = crop_box(hu, b.tolist(), margin=MARGIN, jitter=0.0)  
            if crop.size == 0:
                continue
            img = torch.from_numpy(window_channels(crop))
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

    detected = sorted(prob_sum.keys())
    n_det, n_tot = len(detected), len(patients)
    print(f"\nStage 1 (FRCNN) detection recall: {n_det}/{n_tot} = {n_det/n_tot:.3f}")
    print("detected patients:", detected)

    y_true = [LAB[true_sub[p]] for p in detected]
    y_pred = [int((prob_sum[p] / n_boxes[p]).argmax()) for p in detected]

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1c = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n[FRCNN end-to-end, detected patients only]  n={n_det}  acc {acc:.3f}  macro-F1 {mf1:.3f}")
    for i, c in enumerate(CLASSES):
        print(f"   {c}: F1 {f1c[i]:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"FRCNN->ResNet\nmacro-F1 {mf1:.3f}, recall {n_det}/{n_tot}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cm_pipeline_frcnn.png"), dpi=120)
    print(f"\nconfusion matrix -> {OUT}/cm_pipeline_frcnn.png")

    with open(os.path.join(OUT, "pipeline_frcnn_summary.txt"), "w") as f:
        f.write(f"detector=FRCNN_run3\nscore_threshold={args.score}\n")
        f.write(f"detection_recall={n_det}/{n_tot}={n_det/n_tot:.3f}\n")
        f.write(f"subtype_acc_detected={acc:.3f}\nsubtype_macroF1_detected={mf1:.3f}\n")
        for i, c in enumerate(CLASSES):
            f.write(f"F1_{c}={f1c[i]:.3f}\n")
        f.write("detected_patients=" + ",".join(detected) + "\n")
    print(f"summary -> {OUT}/pipeline_frcnn_summary.txt")

if __name__ == "__main__":
    main()
