"""
make_detection_examples.py

Generates the qualitative Stage 1 figure for the dissertation
(fig:det_examples in subsec:detection_examples): CT slices with the
predicted boxes (from the best Faster R-CNN detector) drawn against the
radiologist reference boxes, chosen to show a good detection, a loosely
localised one, and a false positive.

Run this on the cluster (Hypatia), from /sharedscratch/ps306/lung, inside
the lungtrain env, on a GPU node (it does one FRCNN forward pass per slice).

    /home/ps306/.conda/envs/lungtrain/bin/python make_detection_examples.py

It reads the same manifests, HU cache and windowing the pipeline uses, so the
crops it draws are identical to what the detector actually saw. Nothing here
retrains or changes any result; it only loads the saved checkpoint and draws.

Output: figures/stage1/detection_examples.png  (the file the .tex expects).
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# The pipeline's own windowing and HU cache, imported so the input is
# byte-for-byte what the detector was trained and evaluated on. Stage 1 has
# no named HU loader; it reads one .npy per slice from CACHE keyed by SOP UID
# (stage1_dataset.py line 41), so we do exactly the same here.
from stage1_dataset import window_channels, CACHE


def load_hu(sop):
    """Load a slice's HU array from the cache, as the dataset does."""
    import os as _os
    return np.load(_os.path.join(CACHE, sop + ".npy")).astype(np.float32)


# ----------------------------------------------------------------------
# Config. These paths match the pipeline; override on the command line if
# any differ on your cluster (--ckpt, --manifest).
# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/stage1_results/stage1_run3_best.pth"),
        help="best Faster R-CNN checkpoint")
    ap.add_argument("--manifest", default="stage1_detection_manifest.csv",
        help="per-slice detection manifest with boxes and split")
    ap.add_argument("--split", default="val",
        help="which split to draw from (val is the leakage-free set)")
    ap.add_argument("--score-thresh", type=float, default=0.5,
        help="confidence threshold for a predicted box (same as eval)")
    ap.add_argument("--iou-hit", type=float, default=0.5,
        help="IoU at or above which a prediction counts as a hit")
    ap.add_argument("--iou-loose-hi", type=float, default=0.5,
        help="upper IoU bound for the 'loose' example (a hit but not tight)")
    ap.add_argument("--iou-loose-lo", type=float, default=0.2,
        help="lower IoU bound for the 'loose' example")
    ap.add_argument("--out", default="figures/stage1/detection_examples.png")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ----------------------------------------------------------------------
# Model: rebuild exactly as trained (2-class, COCO-pretrained architecture),
# then load the saved weights.
# ----------------------------------------------------------------------
def build_detector(ckpt_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def slice_image(sop):
    """Reproduce the 3-window input the detector sees, as a uint8 RGB image."""
    hu = load_hu(sop)                        # HU array for this slice
    chans = window_channels(hu, jitter=False)                     # [3,H,W] float in [0,1]
    rgb = (np.asarray(chans) * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(rgb)


@torch.no_grad()
def predict(model, sop, device, score_thresh):
    hu = load_hu(sop)
    chans = torch.as_tensor(np.asarray(window_channels(hu, jitter=False)), dtype=torch.float32)
    out = model([chans.to(device)])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep = scores >= score_thresh
    return boxes[keep], scores[keep]


# ----------------------------------------------------------------------
# Pick one slice for each of the three cases we want to illustrate.
# ----------------------------------------------------------------------
def categorise(gt_boxes, pred_boxes, args):
    """
    Return one of 'good', 'loose', 'false_positive', or None for this slice,
    given its ground-truth and (thresholded) predicted boxes.
    """
    if len(gt_boxes) == 0:
        # no tumour here: any prediction is a false positive
        return "false_positive" if len(pred_boxes) > 0 else None

    if len(pred_boxes) == 0:
        return None  # a miss; not one of the three cases we illustrate

    # best IoU between any prediction and any GT box on this slice
    best = max(iou(p, g) for p in pred_boxes for g in gt_boxes)
    if best >= args.iou_hit and best >= 0.7:
        return "good"
    if args.iou_loose_lo <= best < args.iou_loose_hi:
        return "loose"
    # a stray high-score box far from GT, alongside a GT box, is also an FP case
    stray = any(max(iou(p, g) for g in gt_boxes) < 0.1 for p in pred_boxes)
    if stray:
        return "false_positive"
    return None


def draw(img, gt_boxes, pred_boxes, title):
    """Draw reference boxes (green) and predictions (red) with a caption tag."""
    im = img.convert("RGB")
    d = ImageDraw.Draw(im)
    for g in gt_boxes:
        d.rectangle([g[0], g[1], g[2], g[3]], outline=(60, 200, 90), width=3)
    for p in pred_boxes:
        d.rectangle([p[0], p[1], p[2], p[3]], outline=(230, 60, 60), width=3)
    d.text((8, 8), title, fill=(255, 255, 0))
    return im


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)

    model = build_detector(args.ckpt, device)

    # Ground-truth boxes for a slice. The manifest has one row per slice with a
    # `boxes` column holding a JSON list of [x1,y1,x2,y2] (empty list on
    # negatives), exactly as stage1_dataset.py parses it (json.loads(r.boxes)).
    def gt_for(sop):
        r = df[df["sop"] == sop].iloc[0]
        parsed = json.loads(r["boxes"]) if isinstance(r["boxes"], str) else r["boxes"]
        return [list(map(float, b)) for b in parsed]

    picks = {"good": None, "loose": None, "false_positive": None}

    # Search positives first for the 'good'/'loose' cases and negatives for the
    # false positive, so we don't run a GPU forward pass over thousands of
    # empty slices. `role` is the manifest's own positive/negative flag.
    pos = df[df["role"] != "negative"]["sop"].drop_duplicates()
    neg = df[df["role"] == "negative"]["sop"].drop_duplicates()
    pos = pos.sample(frac=1.0, random_state=args.seed)
    neg = neg.sample(frac=1.0, random_state=args.seed)
    # positives first (fills good/loose, and can also yield a stray-box FP),
    # then negatives (fills false_positive if not already found).
    order = list(pos) + list(neg)

    for sop in order:
        if all(picks.values()):
            break
        gt = gt_for(sop)
        pred, _ = predict(model, sop, device, args.score_thresh)
        cat = categorise(np.array(gt) if gt else np.empty((0, 4)),
                         pred, args)
        if cat and picks[cat] is None:
            picks[cat] = (sop, gt, pred)

    # assemble a single side-by-side panel of whatever cases were found
    labels = {"good": "well localised",
              "loose": "loosely localised",
              "false_positive": "false positive"}
    panels = []
    for key in ["good", "loose", "false_positive"]:
        if picks[key] is None:
            print(f"WARNING: no '{key}' example found in {args.split}; "
                  f"panel will omit it. Try another split or loosen the "
                  f"IoU bounds.")
            continue
        sop, gt, pred = picks[key]
        img = slice_image(sop)
        panels.append(draw(img, gt, pred, labels[key]))

    if not panels:
        raise SystemExit("No example slices found; check manifest/split.")

    w = sum(p.width for p in panels) + 12 * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 12

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}  ({len(panels)} panels: "
          f"{', '.join(k for k in picks if picks[k])})")
    print("Legend: green = radiologist reference box, red = predicted box.")


if __name__ == "__main__":
    main()
