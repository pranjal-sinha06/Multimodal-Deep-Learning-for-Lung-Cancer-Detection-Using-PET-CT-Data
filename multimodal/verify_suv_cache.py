"""
verify_suv_cache.py - check the cache reproduces the probe's numbers.

The SUV probe was run independently, from DICOMs, and returned:
    83 patients (A 63, G 20)
    SUV_max  AUC 0.522   median A 3.42   median G 3.57
    SUV_mean AUC 0.486   median A 1.04   median G 1.37

Run after precompute_suv.py:
    python verify_suv_cache.py
"""
import os, json
import numpy as np, pandas as pd

from crop_utils import crop_window, suv_for_window

LUNG       = "/sharedscratch/ps306/lung"
SUV_SLICES = os.path.join(LUNG, "suv_slices.npy")
SUV_INDEX  = os.path.join(LUNG, "suv_index.csv")
CROPS      = os.path.join(LUNG, "stage2_crops_manifest.csv")
COHORT     = os.path.join(LUNG, "pet_cohort_83.csv")
SUV_CLIP   = 20.0


PROBE = {"A": {"max": 3.42, "mean": 1.04}, "G": {"max": 3.57, "mean": 1.37}}


def main():
    ix = pd.read_csv(SUV_INDEX).set_index("sop")
    slices = np.load(SUV_SLICES, mmap_mode="r")
    crops = pd.read_csv(CROPS)
    coh = pd.read_csv(COHORT)
    crops = crops[crops.patient.isin(coh.patient) & crops.subtype.isin(["A", "G"])]

    rows = []
    for _, c in crops.iterrows():
        if c.sop not in ix.index:
            continue
        r = ix.loc[c.sop]
        if not bool(r.covered):
            continue
        box = json.loads(c.box)

        win = crop_window((512, 512), box, margin=0.0, jitter=0.0)
        pet = {"zs": np.array([r.ct_z]),
               "vol": np.asarray(slices[int(r.row)], dtype=np.float32)[None, ...],
               "x0": float(r.pet_x0), "y0": float(r.pet_y0),
               "dr": float(r.pet_dr), "dc": float(r.pet_dc)}
        ctg = {"x0": float(r.ct_x0), "y0": float(r.ct_y0), "z": float(r.ct_z),
               "dr": float(r.ct_dr), "dc": float(r.ct_dc)}

        patch = suv_for_window(win, ctg, pet, suv_clip=1000.0) * 1000.0
        if patch.size == 0 or patch.max() == 0:
            continue
        rows.append({"patient": c.patient, "subtype": c.subtype,
                     "suv_max": float(patch.max()), "suv_mean": float(patch.mean())})

    if not rows:
        print("NO SUV sampled. The cache is empty or misaligned. Do not train on it.")
        return
    df = pd.DataFrame(rows)
    pat = df.groupby(["patient", "subtype"]).agg(
        suv_max=("suv_max", "max"), suv_mean=("suv_mean", "mean")).reset_index()

    print(f"patients with SUV from cache: {len(pat)} "
          f"(A={sum(pat.subtype=='A')}, G={sum(pat.subtype=='G')})   probe had 83 (63/20)\n")
    print(f"{'':10}{'cache':>18}{'probe':>10}{'diff':>9}")
    ok = True
    for cls in ("A", "G"):
        for metric in ("max", "mean"):
            v = pat.loc[pat.subtype == cls, f"suv_{metric}"].median()
            p = PROBE[cls][metric]
            d = v - p
            flag = "" if abs(d) < 0.35 else "   <-- CHECK"
            if abs(d) >= 0.35:
                ok = False
            print(f"  {cls} SUV_{metric:<5}{v:>14.2f}{p:>10.2f}{d:>+9.2f}{flag}")

    try:
        from sklearn.metrics import roc_auc_score
        y = (pat.subtype == "G").astype(int)
        print(f"\n  AUC SUV_max  {roc_auc_score(y, pat.suv_max):.3f}   probe 0.522")
        print(f"  AUC SUV_mean {roc_auc_score(y, pat.suv_mean):.3f}   probe 0.486")
    except Exception:
        pass

    print("\n" + ("CACHE VERIFIED: matches the independently measured probe."
                  if ok else
                  "MISMATCH. Do not train on this cache. Report these numbers."))
    print("Small differences are expected: the probe sampled nearest-neighbour from"
          "\nthe full volume, this resamples onto the CT grid, and the cache is float16.")


if __name__ == "__main__":
    main()
