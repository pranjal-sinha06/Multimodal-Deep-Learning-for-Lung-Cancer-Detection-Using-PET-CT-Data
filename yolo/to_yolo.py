import os, json, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from stage1_dataset import window_channels, ROOT, CACHE

MANIFEST = os.path.join(ROOT, "stage1_detection_manifest.csv")
OUT = os.path.join(ROOT, "yolo")
df = pd.read_csv(MANIFEST)

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

def convert(split):
    sub = df[df.split == split]
    n_pos = 0
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc=split):
        hu = np.load(os.path.join(CACHE, r.sop + ".npy")).astype(np.float32)
        img = (window_channels(hu) * 255).astype(np.uint8).transpose(1,2,0)          # H,W,3 in [0,1] -> uint8
        Image.fromarray(img).save(os.path.join(OUT, "images", split, r.sop + ".jpg"), quality=95)
        boxes = json.loads(r.boxes)
        lines = []
        H, W = hu.shape
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            n_pos += 1
        with open(os.path.join(OUT, "labels", split, r.sop + ".txt"), "w") as f:
            f.write("\n".join(lines))
    print(f"{split}: {len(sub)} images, {n_pos} with boxes")

for split in ["train", "val"]:
    convert(split)
print("conversion done ->", OUT)
