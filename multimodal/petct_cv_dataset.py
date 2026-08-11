"""
petct_cv_dataset.py  --  CT+PET dataset. Channel 2 is SUV instead of the wide window.

IDENTICAL to CTCohortDataset in every respect except channel 2. Same crops (both
arms call crop_window), same resize, same normalisation, same RNG seeding, same
83 patients, same A=0/G=1 remap. That single-channel swap IS the experiment.

The SUV channel is sampled over EXACTLY the same physical region as the CT crop,
including jitter, via crop_utils.suv_for_window. If it were sampled at the raw
box instead, the CT and PET channels would be spatially misaligned and any
result would be uninterpretable.

Reads only the precomputed cache from precompute_suv.py. No DICOM access.
"""
import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from stage2_dataset import (load_hu_cached, MANIFEST, S2_MEAN, S2_STD, CROP_SIZE)
from stage1_dataset import window_channels
from crop_utils import crop_window, suv_for_window

CLASS_NAMES = {0: "Adenocarcinoma", 1: "Squamous cell carcinoma"}
SUBTYPE_TO_LABEL = {"A": 0, "G": 1}

LUNG       = "/sharedscratch/ps306/lung"
SUV_SLICES = os.path.join(LUNG, "suv_slices.npy")
SUV_INDEX  = os.path.join(LUNG, "suv_index.csv")
COHORT_CSV = os.path.join(LUNG, "pet_cohort_83.csv")


class PETCTCohortDataset(Dataset):
    def __init__(self, patients, is_train):
        df = pd.read_csv(MANIFEST)
        df = df[df.subtype.isin(["A", "G"]) & df.patient.isin(patients)].copy()
        df["y"] = df.subtype.map(SUBTYPE_TO_LABEL)
        assert set(df.y.unique()) <= {0, 1}, "label remap produced non-binary labels"
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        if not os.path.exists(SUV_SLICES):
            raise FileNotFoundError(
                f"{SUV_SLICES} not found. Run precompute_suv.py first.")
        self.suv_ix = pd.read_csv(SUV_INDEX).set_index("sop")
        self._slices = None                       # memory-mapped lazily per worker

        self.mean = torch.tensor(S2_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(S2_STD).view(3, 1, 1)
        self.resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)
        self.geom = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        ]) if is_train else None

    @property
    def slices(self):
        # opened on first use so each DataLoader worker gets its own handle
        if self._slices is None:
            self._slices = np.load(SUV_SLICES, mmap_mode="r")
        return self._slices

    def __len__(self):
        return len(self.df)

    def _suv(self, sop, win):
        """SUV over the same window as the CT crop, [0,1]. Zeros if uncovered."""
        h, w = win[3] - win[1], win[2] - win[0]
        if sop not in self.suv_ix.index:
            return np.zeros((max(h, 1), max(w, 1)), np.float32)
        r = self.suv_ix.loc[sop]
        if not bool(r.covered):
            return np.zeros((max(h, 1), max(w, 1)), np.float32)
        sl = np.asarray(self.slices[int(r.row)], dtype=np.float32)
        pet = {"zs": np.array([r.ct_z]), "vol": sl[None, ...],
               "x0": float(r.pet_x0), "y0": float(r.pet_y0),
               "dr": float(r.pet_dr), "dc": float(r.pet_dc)}
        ctg = {"x0": float(r.ct_x0), "y0": float(r.ct_y0), "z": float(r.ct_z),
               "dr": float(r.ct_dr), "dc": float(r.ct_dc)}
        return suv_for_window(win, ctg, pet)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hu = load_hu_cached(r.sop)
        rng = (np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
               if self.is_train else None)
        box = json.loads(r.box)

        # ONE window, used for both modalities -> guaranteed alignment
        win = crop_window(hu.shape, box, jitter=0.12 if self.is_train else 0.0, rng=rng)
        X1, Y1, X2, Y2 = win
        crop = hu[Y1:Y2, X1:X2]

        ct3 = window_channels(crop, jitter=self.is_train, rng=rng)   # 3,h,w
        suv = self._suv(r.sop, win)                                   # h,w in [0,1]
        img = torch.from_numpy(np.stack([ct3[0], ct3[1], suv]).astype(np.float32))

        img = self.resize(img)
        if self.geom is not None:
            img = self.geom(img)
        return (img - self.mean) / self.std, int(r.y)

    @property
    def class_counts(self):
        return self.df.y.value_counts().sort_index().to_dict()
