import os, json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

ROOT  = "/sharedscratch/ps306/lung"
CACHE = os.path.join(ROOT, "hu_cache")
STATS = json.load(open(os.path.join(ROOT, "norm_stats.json")))
WINDOWS, WIN_ORDER = STATS["windows"], STATS["win_order"]

TUMOUR_LABEL = 1   # 0 = background in torchvision detection

def window_channels(hu, jitter=False, rng=None):
    chans = []
    for name in WIN_ORDER:
        wl, ww = WINDOWS[name]
        if jitter and rng is not None:
            wl = wl + rng.uniform(-0.05, 0.05) * ww
            ww = ww * rng.uniform(0.9, 1.1)
        lo, hi = wl - ww/2, wl + ww/2
        chans.append(np.clip((hu - lo) / (hi - lo), 0, 1))
    return np.stack(chans, axis=0).astype(np.float32)   # 3 x H x W in [0,1]

class LungDetectionDataset(Dataset):
    def __init__(self, manifest_csv, split, cache_dir=CACHE):
        df = pd.read_csv(manifest_csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.cache_dir = cache_dir
        self.is_train = (split == "train")
        self.geom = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        ]) if self.is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hu = np.load(os.path.join(self.cache_dir, r.sop + ".npy")).astype(np.float32)
        rng = np.random.default_rng() if self.is_train else None
        img = torch.from_numpy(window_channels(hu, jitter=self.is_train, rng=rng))
        H, W = img.shape[1], img.shape[2]

        boxes = torch.tensor(json.loads(r.boxes), dtype=torch.float32).reshape(-1, 4)
        labels = torch.full((boxes.shape[0],), TUMOUR_LABEL, dtype=torch.int64)

        if self.geom is not None:
            if boxes.shape[0] > 0:
                bx = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))
                img_t, bx = self.geom(tv_tensors.Image(img), bx)
                img = img_t.as_subclass(torch.Tensor)
                boxes = bx.as_subclass(torch.Tensor)
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, labels = boxes[keep], labels[keep]
            else:
                img = self.geom(tv_tensors.Image(img)).as_subclass(torch.Tensor)
        return img, {"boxes": boxes, "labels": labels}

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    m = os.path.join(ROOT, "stage1_detection_manifest.csv")

    # windowing is [0,1] before normalisation
    ds0 = LungDetectionDataset(m, "val")
    hu0 = np.load(os.path.join(CACHE, ds0.df.sop.iloc[0] + ".npy")).astype(np.float32)
    w = window_channels(hu0)
    assert w.min() >= 0 and w.max() <= 1, (w.min(), w.max())
    print(f"windowing pre-normalise range: [{w.min():.3f}, {w.max():.3f}]")

    for split in ["val", "train"]:
        ds = LungDetectionDataset(m, split)
        dl = DataLoader(ds, batch_size=4, shuffle=(split == "train"),
                        collate_fn=collate_fn, num_workers=0)
        imgs, targets = next(iter(dl))
        assert len(imgs) == 4
        for img in imgs:
            assert img.shape == (3, 512, 512) and img.dtype == torch.float32
            assert torch.isfinite(img).all()
        for t in targets:
            b, l = t["boxes"], t["labels"]
            assert b.ndim == 2 and b.shape[1] == 4 and l.shape[0] == b.shape[0]
            if b.shape[0] > 0:
                assert torch.isfinite(b).all()
                assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all()
                assert (b[:, 2] <= 512).all() and (b[:, 3] <= 512).all()
                assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
        print(f"[{split}] batch ok: boxes per sample={[t['boxes'].shape[0] for t in targets]} "
              f"img range post-normalise=[{imgs[0].min():.2f}, {imgs[0].max():.2f}]")

    # explicitly exercise a positive slice (guarantees the box path ran)
    ds = LungDetectionDataset(m, "train")
    pos_i = int(ds.df.index[ds.df.role == "positive"][0])
    _, tgt = ds[pos_i]
    print(f"positive sample: {tgt['boxes'].shape[0]} box(es), labels={tgt['labels'].tolist()}")
    print("ALL DATASET CHECKS PASSED")
