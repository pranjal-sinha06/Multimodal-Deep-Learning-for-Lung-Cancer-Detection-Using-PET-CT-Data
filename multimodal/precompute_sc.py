
# precompute_sc.py - cache the Secondary Capture RGB slices for training.
           

import os, sys, glob, argparse
import numpy as np, pandas as pd

try:
    import pydicom
except ImportError:
    sys.exit("pip install pydicom")

LUNG      = "/sharedscratch/ps306/lung"
SC_MAN    = os.path.join(LUNG, "sc_crops_manifest.csv")
CT_INDEX  = os.path.join(LUNG, "ct_sc_index.csv")
OUT_ARR   = os.path.join(LUNG, "sc_slices.npy")
OUT_IDX   = os.path.join(LUNG, "sc_slice_index.csv")
COHORT    = os.path.join(LUNG, "pet_cohort_83.csv")

MARKER = "Lung-PET-CT-Dx"
HW = (512, 512)


def to_relative(win_path):
    p = str(win_path).replace("\\", "/")
    i = p.find(MARKER + "/")
    return None if i < 0 else p[i + len(MARKER) + 1:]


def detect_root(sample_rel):
    cands = [os.path.join(LUNG, MARKER),
             os.path.join(LUNG, "manifest-1608669183333", MARKER),
             os.path.join(LUNG, "LPT", "manifest-1608669183333", MARKER)]
    cands += sorted(glob.glob(os.path.join(LUNG, "*", MARKER)))
    cands += sorted(glob.glob(os.path.join(LUNG, "*", "*", MARKER)))
    for c in dict.fromkeys(cands):
        if os.path.exists(os.path.join(c, sample_rel)):
            return c
    return None


def read_rgb(path):
    """Return an (H, W, 3) uint8 array from a Secondary Capture DICOM.
    """
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {arr.dtype}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N patients only")
    ap.add_argument("--cohort", default=COHORT,
                    help="restrict to these patients; pass '' for all in the manifest")
    args = ap.parse_args()

    for p in (SC_MAN, CT_INDEX):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    man = pd.read_csv(SC_MAN)
    if args.cohort and os.path.exists(args.cohort):
        keep = set(pd.read_csv(args.cohort).patient)
        man = man[man.patient.isin(keep)]
        print(f"restricted to {len(keep)} cohort patients "
              f"(from {os.path.basename(args.cohort)})")
    if args.limit:
        first = man.drop_duplicates("patient").patient.tolist()[:args.limit]
        man = man[man.patient.isin(first)]

    slices = man.drop_duplicates("sop")[["sop", "patient", "path"]].reset_index(drop=True)
    print(f"{slices.patient.nunique()} patients, {len(slices)} unique SC slices")
    gb = len(slices) * HW[0] * HW[1] * 3 / 1e9
    print(f"cache size will be about {gb:.2f} GB\n")

    sample = to_relative(slices.path.iloc[0])
    if sample is None:
        sys.exit(f"manifest paths lack '{MARKER}'. First path:\n  {slices.path.iloc[0]}")
    root = args.dicom_root or detect_root(sample)
    if root is None:
        sys.exit("Could not locate the DICOM collection.\n"
                 f"  needed a directory containing: {sample}\n"
                 "  pass it explicitly:  --dicom-root /path/to/Lung-PET-CT-Dx")
    print(f"DICOM root: {root}\n")

    arr = np.lib.format.open_memmap(OUT_ARR, mode="w+", dtype=np.uint8,
                                    shape=(len(slices), *HW, 3))
    recs, failed, row = [], [], 0
    per_patient = {}

    for i, r in slices.iterrows():
        rel = to_relative(r.path)
        f = os.path.join(root, rel) if rel else None
        if not f or not os.path.exists(f):
            failed.append((r.sop, "file not found")); continue
        try:
            img = read_rgb(f)
        except Exception as e:
            failed.append((r.sop, f"{type(e).__name__}: {e}")); continue

        h, w = img.shape[:2]
        if (h, w) != HW:                      
            canvas = np.zeros((*HW, 3), np.uint8)
            canvas[:min(h, HW[0]), :min(w, HW[1])] = img[:HW[0], :HW[1]]
            img = canvas
        arr[row] = img
        recs.append({"sop": r.sop, "patient": r.patient, "row": row,
                     "src_h": h, "src_w": w})
        st = per_patient.setdefault(r.patient, [255, 0])
        st[0] = min(st[0], int(img.min())); st[1] = max(st[1], int(img.max()))
        row += 1
        if row % 1000 == 0:
            print(f"  cached {row}/{len(slices)}")

    arr.flush(); del arr
    pd.DataFrame(recs).to_csv(OUT_IDX, index=False)

    print(f"\nwrote {OUT_ARR}  ({row} of {len(slices)} slices)")
    print(f"wrote {OUT_IDX}")
    if failed:
        print(f"\n{len(failed)} slices FAILED:")
        for s, e in failed[:10]:
            print(f"  {s[:40]}...: {e}")

    print("\n=== pixel ranges per patient (first 10) ===")
    for p, (lo, hi) in list(per_patient.items())[:10]:
        flag = "  <-- SUSPECT" if hi - lo < 30 else ""
        print(f"  {p:<20} min {lo:>3}  max {hi:>3}{flag}")
    odd = [p for p, (lo, hi) in per_patient.items() if hi - lo < 30]
    if odd:
        print(f"\n  {len(odd)} patient(s) with a near-constant image. Investigate "
              "before training.")
    else:
        print("\n  all patients show a normal 8-bit colour spread.")


if __name__ == "__main__":
    main()
