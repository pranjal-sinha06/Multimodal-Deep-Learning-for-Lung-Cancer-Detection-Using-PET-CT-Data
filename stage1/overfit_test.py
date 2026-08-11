import torch
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from stage1_dataset import LungDetectionDataset, collate_fn, ROOT
import os

device = torch.device("cuda")
ds = LungDetectionDataset(os.path.join(ROOT, "stage1_detection_manifest.csv"), "train")

# hand-pick 4 samples that include positives, so box loss is exercised
pos_idx = list(ds.df.index[ds.df.role == "positive"][:2])
neg_idx = list(ds.df.index[ds.df.role == "negative"][:2])
batch_idx = pos_idx + neg_idx
dl = DataLoader(Subset(ds, batch_idx), batch_size=4, collate_fn=collate_fn)
imgs, targets = next(iter(dl))
imgs = [im.to(device) for im in imgs]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
print("batch boxes per sample:", [t["boxes"].shape[0] for t in targets])

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
model.to(device).train()

opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                      lr=0.005, momentum=0.9, weight_decay=0.0005)

for step in range(51):
    loss_dict = model(imgs, targets)
    loss = sum(loss_dict.values())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  " +
              "  ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items()))

print("OVERFIT TEST DONE")
