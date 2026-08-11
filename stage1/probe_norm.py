import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from stage1_dataset import LungDetectionDataset, collate_fn, ROOT
from torch.utils.data import DataLoader
import os

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
t = model.transform
print("model.transform.image_mean:", t.image_mean)
print("model.transform.image_std :", t.image_std)
print("model.transform.min_size  :", t.min_size)
print("model.transform.max_size  :", t.max_size)

ds = LungDetectionDataset(os.path.join(ROOT, "stage1_detection_manifest.csv"), "val")
dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn, num_workers=0)
imgs, targets = next(iter(dl))
img = imgs[0]
print("\nour dataset output (already normalised):")
print("  shape:", tuple(img.shape), "range:", round(float(img.min()),2), "to", round(float(img.max()),2),
      "mean:", round(float(img.mean()),3))

# what the model's transform does to already-normalised input
images, _ = model.transform([img], None)
t_img = images.tensors[0]
print("\nafter model.transform (what the backbone actually sees):")
print("  shape:", tuple(t_img.shape), "range:", round(float(t_img.min()),2), "to", round(float(t_img.max()),2),
      "mean:", round(float(t_img.mean()),3))
