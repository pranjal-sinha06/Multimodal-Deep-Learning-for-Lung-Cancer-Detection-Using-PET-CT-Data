import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2
from stage1_dataset import window_channels, ROOT, CACHE # reuse verified helpers

def load_hu_cached(sop):
    return np.load(os.path.join(CACHE, sop + ".npy")).astype(np.float32)

MANIFEST = os.path.join(ROOT, "stage2_crops_manifest.csv")
STATS = json.load(open(os.path.join(ROOT, "norm_stats.json")))
S2_MEAN, S2_STD = STATS["stage2"]["mean"], STATS["stage2"]["std"]
CROP_SIZE = 256
MARGIN = 0.5

def crop_box(hu, box, margin=MARGIN, jitter=0.0, rng=None):
    H, W = hu.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if jitter and rng is not None:
        x1 += rng.uniform(-jitter, jitter) * bw; x2 += rng.uniform(-jitter, jitter) * bw
        y1 += rng.uniform(-jitter, jitter) * bh; y2 += rng.uniform(-jitter, jitter) * bh
    mx, my = bw * margin / 2, bh * margin / 2
    X1, Y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    X2, Y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    return hu[Y1:Y2, X1:X2]

class LungSubtypeDataset(Dataset):
    def __init__(self, split):
        df = pd.read_csv(MANIFEST)
        self.df = df[df.split == split].reset_index(drop=True)
        self.is_train = (split == "train")
        self.mean = torch.tensor(S2_MEAN).view(3, 1, 1)
        self.std = torch.tensor(S2_STD).view(3, 1, 1)
        self.resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)
        self.geom = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        ]) if self.is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hu = load_hu_cached(r.sop)
        rng = (np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
               if self.is_train else None)
        crop = crop_box(hu, json.loads(r.box), jitter=0.12 if self.is_train else 0.0, rng=rng)
        img = torch.from_numpy(window_channels(crop, jitter=self.is_train, rng=rng))  # 3,h,w
        img = self.resize(img)
        if self.geom is not None:
            img = self.geom(img)
        img = (img - self.mean) / self.std
        return img, int(r.label_idx)

if __name__ == "__main__":
    from collections import Counter
    for split in ["val", "train"]:
        ds = LungSubtypeDataset(split)
        dl = DataLoader(ds, batch_size=32, shuffle=(split == "train"), num_workers=4)
        imgs, labels = next(iter(dl))
        assert imgs.shape[1:] == (3, 256, 256) and imgs.dtype == torch.float32
        assert torch.isfinite(imgs).all()
        assert set(labels.tolist()) <= {0, 1, 2}
        print(f"[{split}] batch {tuple(imgs.shape)} range [{imgs.min():.2f},{imgs.max():.2f}] "
              f"labels {dict(Counter(labels.tolist()))}")
    
    ds = LungSubtypeDataset("train")
    print("train label counts:", dict(Counter(ds.df.label_idx.tolist())))

    from PIL import Image, ImageDraw
    outdir = os.path.join(ROOT, "stage2_verify"); os.makedirs(outdir, exist_ok=True)
    for k in range(6):
        r = ds.df.iloc[k * 500 % len(ds.df)]
        hu = load_hu_cached(r.sop)
        full = (window_channels(hu) * 255).astype(np.uint8).transpose(1, 2, 0)
        im = Image.fromarray(full); d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = json.loads(r.box)
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        im.save(os.path.join(outdir, f"{r.subtype}_{r.sop[-8:]}.jpg"))
    print("wrote 6 overlays to", outdir)
    print("STAGE 2 DATASET CHECKS PASSED")
