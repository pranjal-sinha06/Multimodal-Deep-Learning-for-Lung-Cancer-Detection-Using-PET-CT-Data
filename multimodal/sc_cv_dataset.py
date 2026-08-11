"""
sc_cv_dataset.py  --  Secondary Capture dataset for the 2-class CV comparison.

The SC arm is the third representation in the comparison:
    CT-only  [lung window, mediastinal window, wide window]   AUC 0.779
    CT+PET   [lung window, mediastinal window, SUV]           AUC 0.740
    SC       [R, G, B] of the workstation-fused PET/CT render

All three use the same 83 patients, the same folds, the same seeds and the same
ResNet-50. Cropping goes through crop_utils.crop_window, the identical function
the CT and CT+PET arms use, so the geometry is the same and only the pixel
content differs.

ONE DELIBERATE DIFFERENCE, STATED RATHER THAN HIDDEN
  Normalisation cannot be shared. The CT arms normalise windowed HU in [0,1]
  using statistics computed over that distribution; SC is 8-bit colour in
  [0,255] with an entirely different distribution, so those statistics are
  meaningless here. ImageNet statistics are used instead, which is the standard
  choice for RGB input to an ImageNet-pretrained backbone.

  Normalisation is a property of the representation, not of the model, so this
  does not break the single-variable comparison. It does need saying in the
  methodology.
"""
import os, json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from stage2_dataset import CROP_SIZE
from crop_utils import crop_window

CLASS_NAMES = {0: "Adenocarcinoma", 1: "Squamous cell carcinoma"}
SUBTYPE_TO_LABEL = {"A": 0, "G": 1}

LUNG       = "/sharedscratch/ps306/lung"
SC_MAN     = os.path.join(LUNG, "sc_crops_manifest.csv")
SC_SLICES  = os.path.join(LUNG, "sc_slices.npy")
SC_INDEX   = os.path.join(LUNG, "sc_slice_index.csv")

# ImageNet statistics: SC is RGB and the backbone is ImageNet-pretrained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class SCCohortDataset(Dataset):
    def __init__(self, patients, is_train, manifest=SC_MAN):
        """manifest defaults to the full SC manifest, so existing callers are
        unaffected. The density-matched control passes its own manifest."""
        if not os.path.exists(SC_SLICES):
            raise FileNotFoundError(f"{SC_SLICES} not found. Run precompute_sc.py first.")
        df = pd.read_csv(manifest)
        df = df[df.subtype.isin(["A", "G"]) & df.patient.isin(patients)].copy()
        df["y"] = df.subtype.map(SUBTYPE_TO_LABEL)
        assert set(df.y.unique()) <= {0, 1}, "label remap produced non-binary labels"

        # keep only boxes whose slice actually made it into the cache
        idx = pd.read_csv(SC_INDEX).set_index("sop")
        df = df[df.sop.isin(idx.index)].copy()
        df["row"] = df.sop.map(idx.row)
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self._slices = None                       # memory-mapped per worker

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self.resize = v2.Resize((CROP_SIZE, CROP_SIZE), antialias=True)
        self.geom = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        ]) if is_train else None

    @property
    def slices(self):
        if self._slices is None:
            self._slices = np.load(SC_SLICES, mmap_mode="r")
        return self._slices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rng = (np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
               if self.is_train else None)
        box = json.loads(r.box)

        img = self.slices[int(r.row)]                       # (512, 512, 3) uint8
        X1, Y1, X2, Y2 = crop_window(img.shape[:2], box,
                                     jitter=0.12 if self.is_train else 0.0, rng=rng)
        crop = np.asarray(img[Y1:Y2, X1:X2], dtype=np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))                # HWC -> CHW

        t = torch.from_numpy(np.ascontiguousarray(crop))
        t = self.resize(t)
        if self.geom is not None:
            t = self.geom(t)
        return (t - self.mean) / self.std, int(r.y)

    @property
    def class_counts(self):
        return self.df.y.value_counts().sort_index().to_dict()
