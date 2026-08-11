import os, glob, random
import numpy as np
from PIL import Image, ImageDraw
from stage1_dataset import ROOT

OUT = os.path.join(ROOT, "yolo")
# pick 6 positive samples (label files that are non-empty) from train
lbls = [f for f in glob.glob(os.path.join(OUT, "labels/train/*.txt")) if os.path.getsize(f) > 0]
random.seed(0); pick = random.sample(lbls, 6)

os.makedirs(os.path.join(OUT, "verify"), exist_ok=True)
for lf in pick:
    sop = os.path.splitext(os.path.basename(lf))[0]
    img = Image.open(os.path.join(OUT, "images/train", sop + ".jpg")).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    for line in open(lf):
        _, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H
        x2, y2 = (cx + bw/2) * W, (cy + bh/2) * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    img.save(os.path.join(OUT, "verify", sop + "_boxed.jpg"))
print(f"wrote 6 overlay images to {os.path.join(OUT, 'verify')}")
