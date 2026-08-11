"""
precompute_suv.py - build the SUV cache once, so training reads no DICOMs.

"""
import os, sys, glob, argparse, datetime
import numpy as np, pandas as pd

try:
    import pydicom
except ImportError:
    sys.exit("pip install pydicom")

LUNG       = "/sharedscratch/ps306/lung"
PET_INDEX  = os.path.join(LUNG, "pet_index.csv")
CT_INDEX   = os.path.join(LUNG, "ct_sc_index.csv")
COHORT     = os.path.join(LUNG, "pet_cohort_83.csv")
CROPS      = os.path.join(LUNG, "stage2_crops_manifest.csv")
SUV_SLICES = os.path.join(LUNG, "suv_slices.npy")
SUV_INDEX  = os.path.join(LUNG, "suv_index.csv")

MARKER = "Lung-PET-CT-Dx"     
PET_HW = (200, 200)           
Z_TOL  = 10.0                 




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
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if os.path.exists(os.path.join(c, sample_rel)):
            return c
    return None




def suv_factor(ds):
    """Body-weight SUV factor for Units == BQML.
    SUV = RAW pixel * factor. RescaleSlope folded in exactly once."""
    units = str(getattr(ds, "Units", ""))
    if units != "BQML":
        raise ValueError(f"expected Units=BQML, got {units!r}")
    seq = ds.RadiopharmaceuticalInformationSequence[0]
    w_g = float(ds.PatientWeight) * 1000.0
    dose = float(seq.RadionuclideTotalDose)
    half = float(seq.RadionuclideHalfLife)
    def pt(x):
        return datetime.datetime.strptime(str(x).split(".")[0], "%H%M%S")
    scan = getattr(ds, "SeriesTime", None) or ds.AcquisitionTime
    dt = (pt(scan) - pt(seq.RadiopharmaceuticalStartTime)).seconds
    return float(getattr(ds, "RescaleSlope", 1.0)) * w_g / (dose * 0.5 ** (dt / half))


def pet_volume(rows, root):
    """One patient's PET series as an SUV volume, sorted by z."""
    rels = [to_relative(p) for p in rows.path]
    if any(r is None for r in rels):
        raise ValueError("path missing the collection marker")
    fac = suv_factor(pydicom.dcmread(os.path.join(root, rels[0]), force=True))
    sl = []
    for rel in rels:
        ds = pydicom.dcmread(os.path.join(root, rel), force=True)
        ipp = [float(v) for v in ds.ImagePositionPatient]
        ps = [float(v) for v in ds.PixelSpacing]
        sl.append({"z": ipp[2], "x0": ipp[0], "y0": ipp[1], "dr": ps[0], "dc": ps[1],
                   "img": ds.pixel_array.astype(np.float32) * fac})
    sl.sort(key=lambda s: s["z"])
    return (np.stack([s["img"] for s in sl]),
            np.array([s["z"] for s in sl], dtype=np.float64),
            sl[0])                       # geometry constant within a series


def ct_geometry(path, root):
    rel = to_relative(path)
    if rel is None:
        return None
    f = os.path.join(root, rel)
    if not os.path.exists(f):
        return None
    ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
    ipp = [float(v) for v in ds.ImagePositionPatient]
    ps = [float(v) for v in ds.PixelSpacing]
    return {"x0": ipp[0], "y0": ipp[1], "z": ipp[2], "dr": ps[0], "dc": ps[1]}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pet_ix = pd.read_csv(PET_INDEX, low_memory=False)
    ct_ix = pd.read_csv(CT_INDEX, low_memory=False)
    ct_ix = ct_ix[ct_ix.kind == "CT"] if "kind" in ct_ix.columns else ct_ix
    ct_ix = ct_ix.drop_duplicates("sop").set_index("sop")
    crops = pd.read_csv(CROPS)
    patients = pd.read_csv(COHORT).patient.tolist()
    if args.limit:
        patients = patients[:args.limit]

    sample = to_relative(pet_ix.path.iloc[0])
    if sample is None:
        sys.exit(f"pet_index paths lack '{MARKER}'. First path:\n  {pet_ix.path.iloc[0]}")
    root = args.dicom_root or detect_root(sample)
    if root is None:
        sys.exit("Could not locate the DICOM collection.\n"
                 f"  needed a directory containing: {sample}\n"
                 f"  searched under {LUNG}\n"
                 "  pass it explicitly:  --dicom-root /path/to/Lung-PET-CT-Dx")
    print(f"DICOM root: {root}")

    todo = crops[crops.patient.isin(patients)].drop_duplicates("sop")
    N = len(todo)
    print(f"{len(patients)} patients, {N} annotated slices to cache\n")

    arr = np.lib.format.open_memmap(SUV_SLICES, mode="w+",
                                    dtype=np.float16, shape=(N, *PET_HW))
    recs, row, failed = [], 0, []

    for i, p in enumerate(patients, 1):
        sops = todo[todo.patient == p]
        prow = pet_ix[pet_ix.patient == p]
        if prow.empty:
            failed.append((p, "no PET rows")); continue
        try:
            vol, zs, geo = pet_volume(prow, root)
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}")); continue

        n_cov, mx = 0, 0.0
        for _, c in sops.iterrows():
            if c.sop not in ct_ix.index:
                continue
            cg = ct_geometry(ct_ix.loc[c.sop, "path"], root)
            if cg is None:
                continue
            k = int(np.argmin(np.abs(zs - cg["z"])))
            covered = bool(abs(zs[k] - cg["z"]) <= Z_TOL)
            arr[row] = (vol[k] if covered else np.zeros(PET_HW, np.float32)).astype(np.float16)
            if covered:
                n_cov += 1
                mx = max(mx, float(vol[k].max()))
            recs.append({"sop": c.sop, "patient": p, "row": row, "covered": covered,
                         "ct_x0": cg["x0"], "ct_y0": cg["y0"], "ct_z": cg["z"],
                         "ct_dr": cg["dr"], "ct_dc": cg["dc"],
                         "pet_x0": geo["x0"], "pet_y0": geo["y0"],
                         "pet_dr": geo["dr"], "pet_dc": geo["dc"],
                         "pet_dz": float(abs(zs[k] - cg["z"]))})
            row += 1
        del vol
        print(f"  [{i}/{len(patients)}] {p}: {len(sops)} slices, {n_cov} PET-covered, "
              f"slice SUV max {mx:5.2f}")

    arr.flush(); del arr
    pd.DataFrame(recs).to_csv(SUV_INDEX, index=False)

    print(f"\nwrote {SUV_SLICES}  ({row} of {N} rows used, "
          f"{row * PET_HW[0] * PET_HW[1] * 2 / 1e6:.0f} MB)")
    print(f"wrote {SUV_INDEX}")
    if failed:
        print(f"\n{len(failed)} patients FAILED:")
        for p, e in failed[:10]:
            print(f"  {p}: {e}")
    d = pd.DataFrame(recs)
    if len(d):
        print(f"\nPET coverage: {d.covered.sum()}/{len(d)} slices "
              f"({100*d.covered.mean():.1f}%), median z offset "
              f"{d.loc[d.covered, 'pet_dz'].median():.2f} mm")
    print("\nSanity: tumour slice SUV maxima above should be roughly 3-20.")


if __name__ == "__main__":
    main()
