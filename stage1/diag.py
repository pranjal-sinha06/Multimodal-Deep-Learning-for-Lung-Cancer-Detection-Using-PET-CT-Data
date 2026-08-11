import os, torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from stage1_dataset import LungDetectionDataset, collate_fn, ROOT

torch.multiprocessing.set_sharing_strategy("file_system")
device = torch.device("cuda")
MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
N = 2000

def build_model(ckpt):
    m = fasterrcnn_resnet50_fpn(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return m.to(device).eval()

def make_loader(split):
    ds = LungDetectionDataset(MANIFEST, split)
    ds.is_train = False; ds.geom = None          # evaluate train under val-identical (no-aug) conditions
    g = torch.Generator().manual_seed(0)
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

for tag, ckpt in [("epoch0 (best)", "runs/stage1_run5/best.pth"),
                  ("epoch10 (last)", "runs/stage1_run5/last.pth")]:
    model = build_model(ckpt)
    tr = evaluate(model, make_loader("train"))
    va = evaluate(model, make_loader("val"))
    print(f"{tag}: train mAP {tr[0]:.3f} mAP50 {tr[1]:.3f} | "
          f"val mAP {va[0]:.3f} mAP50 {va[1]:.3f} | gap(mAP50) {tr[1]-va[1]:+.3f}")
print("DIAGNOSTIC DONE")
