"""
ct_cv_dataset.py  --  cohort-restricted, 2-class CT dataset for the PET-cohort CV.

Reuses the EXACT crop + window + normalise path from stage2_dataset.py, so the
CT-only arm here is directly comparable to Chapter 5. The only differences:
  - selects crops by an explicit patient list (a CV fold), not the split column
  - keeps only A and G, and remaps labels to a contiguous {A:0, G:1}
  - seeds the augmentation RNG (the fix that made Stage 2 reproducible)
"""
import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# reuse verified helpers — identical preprocessing to Stage 2
from stage2_dataset import (crop_box, load_hu_cached, MANIFEST,
                            S2_MEAN, S2_STD, CROP_SIZE)
from stage1_dataset import window_channels

# 2-class problem. Full names are used everywhere a label is displayed.
CLASS_NAMES = {0: "Adenocarcinoma", 1: "Squamous cell carcinoma"}
CLASS_SHORT = {0: "A", 1: "G"}
SUBTYPE_TO_LABEL = {"A": 0, "G": 1}          # remap: manifest has A=0,B=1,G=2


class CTCohortDataset(Dataset):
    def __init__(self, patients, is_train):
        df = pd.read_csv(MANIFEST)
        df = df[df.subtype.isin(["A", "G"]) & df.patient.isin(patients)].copy()
        df["y"] = df.subtype.map(SUBTYPE_TO_LABEL)
        assert set(df.y.unique()) <= {0, 1}, "label remap produced non-binary labels"
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.mean = torch.tensor(S2_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(S2_STD).view(3, 1, 1)
        self.resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)
        self.geom = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        ]) if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hu = load_hu_cached(r.sop)
        # seed numpy RNG from torch's per-worker generator -> reproducible AND
        # still varied per item/epoch (the Stage 2 fix)
        rng = (np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
               if self.is_train else None)
        crop = crop_box(hu, json.loads(r.box), jitter=0.12 if self.is_train else 0.0, rng=rng)
        img = torch.from_numpy(window_channels(crop, jitter=self.is_train, rng=rng))
        img = self.resize(img)
        if self.geom is not None:
            img = self.geom(img)
        img = (img - self.mean) / self.std
        return img, int(r.y)

    @property
    def class_counts(self):
        return self.df.y.value_counts().sort_index().to_dict()
