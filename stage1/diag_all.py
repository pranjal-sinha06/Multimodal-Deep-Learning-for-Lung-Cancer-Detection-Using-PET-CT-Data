import os, csv, torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from stage1_dataset import LungDetectionDataset, collate_fn, ROOT

torch.multiprocessing.set_sharing_strategy("file_system")
device = torch.device("cuda")
MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
N = 2000
RUNS = ["run1", "run2", "run3", "run4", "run5"]
OUT_CSV = "figures/stage1/diag_train_val.csv"


def build_model(ckpt):
    ck = torch.load(ckpt, map_location=device)
    m = fasterrcnn_resnet50_fpn(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), int(ck.get("epoch", -1))


def make_loader(split):
    ds = LungDetectionDataset(MANIFEST, split)
    ds.is_train = False; ds.geom = None          # evaluate train under val-identical (no-aug) conditions
    g = torch.Generator().manual_seed(0)         # fixed subset -> identical across every run
    idx = torch.randperm(len(ds), generator=g)[:N].tolist()
    return DataLoader(Subset(ds, idx), batch_size=4, collate_fn=collate_fn, num_workers=4)


@torch.no_grad()
def evaluate(model, loader):
    metric = MeanAveragePrecision(box_format="xyxy", backend="pycocotools")
    for imgs, targets in loader:
        preds = model([im.to(device) for im in imgs])
        metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], list(targets))
    r = metric.compute()
    return float(r["map"]), float(r["map_50"])


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# built once and reused: the subsets are deterministic, so every checkpoint sees identical data
train_loader = make_loader("train")
val_loader = make_loader("val")
print(f"fixed subsets (seed 0, no augmentation): "
      f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} slices")
print()

rows = []
for tag in RUNS:
    for which in ["best", "last"]:
        ckpt = f"runs/stage1_{tag}/{which}.pth"
        if not os.path.isfile(ckpt):
            print(f"  MISSING {ckpt} -- skipped")
            continue
        model, epoch = build_model(ckpt)
        tr_map, tr_map50 = evaluate(model, train_loader)
        va_map, va_map50 = evaluate(model, val_loader)
        row = dict(run=tag, which=which, epoch=epoch,
                   train_map=tr_map, train_map50=tr_map50,
                   val_map=va_map, val_map50=va_map50,
                   gap_map=tr_map - va_map, gap_map50=tr_map50 - va_map50)
        rows.append(row)
        print(f"{tag} {which:4s} (epoch {epoch:2d}): "
              f"train mAP50 {tr_map50:.3f} | val mAP50 {va_map50:.3f} | gap {row['gap_map50']:+.3f}"
              f"   [mAP50:95  train {tr_map:.3f} | val {va_map:.3f} | gap {row['gap_map']:+.3f}]")
        del model
        torch.cuda.empty_cache()

if rows:
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} evaluations)")
else:
    print("\nno checkpoints found -- nothing written")
print("DIAGNOSTIC DONE")
