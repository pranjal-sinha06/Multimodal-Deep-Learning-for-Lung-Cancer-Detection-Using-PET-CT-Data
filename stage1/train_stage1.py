import os, json, csv, time, random, argparse, shutil
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import autocast, GradScaler
from stage1_dataset import LungDetectionDataset, collate_fn, ROOT

torch.multiprocessing.set_sharing_strategy("file_system")

MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
SEED = 0
BATCH = 4
LR, MOMENTUM, WEIGHT_DECAY = 0.001, 0.9, 5e-4
NUM_WORKERS = 4
SCORE_THR, IOU_THR = 0.5, 0.5
HOME_RESULTS = os.path.expanduser("~/stage1_results")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-iters", type=int, default=0)   # 0 = full epoch
    ap.add_argument("--max-val-images", type=int, default=0)    # 0 = full val
    return ap.parse_args()


def tp_fp_fn_image(pb, ps, gb):
    keep = ps >= SCORE_THR
    pb, ps = pb[keep], ps[keep]
    if len(gb) == 0:
        return 0, len(pb), 0
    if len(pb) == 0:
        return 0, 0, len(gb)
    ious = box_iou(pb, gb)
    matched, tp = set(), 0
    for i in torch.argsort(ps, descending=True).tolist():
        g = int(torch.argmax(ious[i]))
        if ious[i, g] >= IOU_THR and g not in matched:
            matched.add(g); tp += 1
    return tp, len(pb) - tp, len(gb) - len(matched)


@torch.no_grad()
def validate(model, loader, device, max_images):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", backend="pycocotools")
    TP = FP = FN = 0; seen = 0
    for imgs, targets in loader:
        preds = model([im.to(device) for im in imgs])
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, list(targets))
        for p, t in zip(preds, targets):
            tp, fp, fn = tp_fp_fn_image(p["boxes"], p["scores"], t["boxes"])
            TP += tp; FP += fp; FN += fn
        seen += len(imgs)
        if max_images and seen >= max_images:
            break
    res = metric.compute()
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    return {"map": float(res["map"]), "map_50": float(res["map_50"]),
            "precision": prec, "recall": rec, "tp": TP, "fp": FP, "fn": FN}


def main():
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run_dir = os.path.join(ROOT, "runs", f"stage1_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = os.path.join(run_dir, "last.pth")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    train_ds = LungDetectionDataset(MANIFEST, "train")
    val_ds = LungDetectionDataset(MANIFEST, "val")

    # positive-aware sampler: draw ~50% positive slices per batch
    POS_FRAC = 0.25
    is_pos = (train_ds.df.role.values == "positive")
    n_pos, n_neg = is_pos.sum(), (~is_pos).sum()
    weights = np.where(is_pos,POS_FRAC / n_pos, (1-POS_FRAC) / n_neg)
    g = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(train_ds), replacement=True, generator=g)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
    model.to(device)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    total_iters = 25 * len(train_loader)
    warmup_iters = min(1000, len(train_loader) - 1)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_iters),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_iters - warmup_iters, 1)),
    ], milestones=[warmup_iters])
    scaler = GradScaler("cuda")

    start_epoch, best_map, no_improve = 0, -1.0, 0
    rows = []
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1; best_map = ck["best_map"]; no_improve = ck["no_improve"]
        if os.path.exists(metrics_csv):
            rows = pd.read_csv(metrics_csv).to_dict("records")
        print(f"resumed at epoch {start_epoch}, best_map={best_map:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time()
        agg = {k: 0.0 for k in ["loss", "loss_classifier", "loss_box_reg",
                                "loss_objectness", "loss_rpn_box_reg"]}
        n = 0
        for it, (imgs, targets) in enumerate(train_loader):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast("cuda"):
                ld = model(imgs, targets)
                loss = sum(ld.values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            agg["loss"] += loss.item()
            for k in ld:
                agg[k] += ld[k].item()
            n += 1
            if args.max_train_iters and it + 1 >= args.max_train_iters:
                break
        for k in agg:
            agg[k] /= max(n, 1)

        val = validate(model, val_loader, device, args.max_val_images)
        dt = time.time() - t0
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **agg,
               "val_map": val["map"], "val_map_50": val["map_50"],
               "val_precision": val["precision"], "val_recall": val["recall"],
               "val_tp": val["tp"], "val_fp": val["fp"], "val_fn": val["fn"],
               "epoch_time_s": round(dt, 1)}
        rows.append(row)
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(rows)
        json.dump(rows, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
        print(f"epoch {epoch}: train_loss {agg['loss']:.4f} | val mAP {val['map']:.4f} "
              f"mAP50 {val['map_50']:.4f} | P {val['precision']:.3f} R {val['recall']:.3f} | "
              f"TP {val['tp']} FP {val['fp']} FN {val['fn']} | {dt:.0f}s")

        improved = val["map"] > best_map
        no_improve = 0 if improved else no_improve + 1
        if improved:
            best_map = val["map"]
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_map": best_map, "no_improve": no_improve}, last_ckpt)
        if improved:
            torch.save({"epoch": epoch, "model": model.state_dict(), "val": val},
                       os.path.join(run_dir, "best.pth"))
            print(f"  new best mAP {best_map:.4f} -> best.pth")
        if no_improve >= args.patience:
            print(f"early stopping at epoch {epoch} (no val mAP gain for {args.patience} epochs)")
            break

    os.makedirs(HOME_RESULTS, exist_ok=True)
    for f in ["best.pth", "metrics.csv", "metrics.json"]:
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HOME_RESULTS, f"stage1_{args.tag}_{f}"))
    print(f"done. best val mAP {best_map:.4f}. artifacts in {run_dir} and {HOME_RESULTS}")


if __name__ == "__main__":
    main()
