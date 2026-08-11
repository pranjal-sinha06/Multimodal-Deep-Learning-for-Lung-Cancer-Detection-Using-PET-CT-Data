"""
scanner_probe.py  --  can subtype be predicted from acquisition settings alone?

WHY
  The adenocarcinoma and squamous patients occupy contiguous identifier ranges
  (A0165 to A0265, G0033 to G0062), which is consistent with the two groups
  having been acquired in different periods or on different equipment. If the
  acquisition parameters differ systematically between the classes, a network
  can separate them by reading scanner signature rather than tumour morphology,
  and every classification result in this dissertation inherits that doubt.

  This is the same probe used to reject LIDC-IDRI as a negative source: rather
  than argue about whether a domain shift exists, measure whether the metadata
  alone is separable.

WHAT IT DOES
  Reads one header per annotated CT series, extracts the acquisition parameters
  that characterise a protocol, and asks two questions:

    1. Do the individual parameters differ between A and G?
       Mann-Whitney for continuous fields, contingency counts for categorical.

    2. Can a classifier predict subtype from the parameters alone?
       Leave-one-out logistic regression on the numeric fields, scored by AUC.
       This is the summary that matters: individually weak differences can
       still combine into a strong signature.

  Headers only, no pixel data. Nothing is modified.

READING THE RESULT
  AUC near 0.5   -> acquisition carries no subtype signal. The confound is
                    ruled out and this should be stated in the limitations as
                    a measured negative rather than an untested assumption.
  AUC above 0.7  -> acquisition alone separates the classes. Any image model
                    may be exploiting it, and the discussion must say so.

USAGE
  python scanner_probe.py
  python scanner_probe.py --cohort pet_cohort_83.csv     # the CV arms' patients
"""
import os, sys, glob, argparse
from collections import Counter

import numpy as np, pandas as pd

try:
    import pydicom
except ImportError:
    sys.exit("pip install pydicom")

LUNG   = "/sharedscratch/ps306/lung"
CROPS  = os.path.join(LUNG, "stage2_crops_manifest.csv")
CT_IDX = os.path.join(LUNG, "ct_sc_index.csv")
MARKER = "Lung-PET-CT-Dx"

NUMERIC = ["KVP", "SliceThickness", "Exposure", "XRayTubeCurrent",
           "SpacingBetweenSlices", "PixelSpacing0"]
CATEG   = ["ConvolutionKernel", "Manufacturer", "ManufacturerModelName",
           "ProtocolName", "StudyYear"]


def to_relative(p):
    s = str(p).replace("\\", "/")
    i = s.find(MARKER + "/")
    return None if i < 0 else s[i + len(MARKER) + 1:]


def detect_root(sample):
    c = [os.path.join(LUNG, MARKER),
         os.path.join(LUNG, "manifest-1608669183333", MARKER),
         os.path.join(LUNG, "LPT", "manifest-1608669183333", MARKER)]
    c += sorted(glob.glob(os.path.join(LUNG, "*", MARKER)))
    c += sorted(glob.glob(os.path.join(LUNG, "*", "*", MARKER)))
    for r in dict.fromkeys(c):
        if os.path.exists(os.path.join(r, sample)):
            return r
    return None


def header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    g = lambda t: getattr(ds, t, None)
    ps = g("PixelSpacing")
    date = str(g("StudyDate") or "")
    rec = {"KVP": g("KVP"), "SliceThickness": g("SliceThickness"),
           "Exposure": g("Exposure"), "XRayTubeCurrent": g("XRayTubeCurrent"),
           "SpacingBetweenSlices": g("SpacingBetweenSlices"),
           "PixelSpacing0": (float(ps[0]) if ps else None),
           "ConvolutionKernel": str(g("ConvolutionKernel") or ""),
           "Manufacturer": str(g("Manufacturer") or ""),
           "ManufacturerModelName": str(g("ManufacturerModelName") or ""),
           "ProtocolName": str(g("ProtocolName") or ""),
           "StudyYear": date[:4] if len(date) >= 4 else ""}
    for k in NUMERIC:
        try:
            rec[k] = float(rec[k]) if rec[k] is not None else np.nan
        except (TypeError, ValueError):
            rec[k] = np.nan
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None,
                    help="optional CSV with a patient column, to restrict the probe")
    ap.add_argument("--dicom-root", default=None)
    ap.add_argument("--permutations", type=int, default=200,
                    help="permutations for the null; 200 takes about a minute")
    args = ap.parse_args()

    crops = pd.read_csv(CROPS)
    ct = pd.read_csv(CT_IDX, low_memory=False)
    ct = ct[ct.kind == "CT"] if "kind" in ct.columns else ct

    crops = crops[crops.subtype.isin(["A", "G"])]
    if args.cohort and os.path.exists(args.cohort):
        keep = set(pd.read_csv(args.cohort).patient)
        crops = crops[crops.patient.isin(keep)]
        print(f"restricted to {len(keep)} patients from {os.path.basename(args.cohort)}")

    # the annotated CT series for each patient: exactly what the models see
    ann = ct[ct.sop.isin(set(crops.sop))].drop_duplicates("series")
    sub = crops.drop_duplicates("patient").set_index("patient").subtype
    print(f"{ann.patient.nunique()} patients, {len(ann)} annotated CT series\n")

    sample = to_relative(ann.path.iloc[0])
    root = args.dicom_root or detect_root(sample)
    if root is None:
        sys.exit("could not locate the DICOM collection; pass --dicom-root")
    print(f"DICOM root: {root}\n")

    rows = []
    for _, r in ann.iterrows():
        rel = to_relative(r.path)
        f = os.path.join(root, rel) if rel else None
        if not f or not os.path.exists(f):
            continue
        try:
            rec = header(f)
        except Exception:
            continue
        rec["patient"] = r.patient
        rec["subtype"] = sub.get(r.patient, "")
        rows.append(rec)

    df = pd.DataFrame(rows)
    df = df[df.subtype.isin(["A", "G"])]
    if df.empty:
        sys.exit("no headers read; check the paths")
    # one row per patient, so a patient with many series does not dominate
    df = df.groupby("patient").first().reset_index()
    print(f"read {len(df)} patient-level records "
          f"(A={sum(df.subtype=='A')}, G={sum(df.subtype=='G')})\n")

    A = df[df.subtype == "A"]; G = df[df.subtype == "G"]

    print("=" * 66)
    print("CONTINUOUS PARAMETERS")
    print("=" * 66)
    from scipy.stats import mannwhitneyu
    print(f"{'parameter':<22}{'A median':>12}{'G median':>12}{'p':>10}")
    for k in NUMERIC:
        a, g = A[k].dropna(), G[k].dropna()
        if len(a) < 3 or len(g) < 3:
            print(f"  {k:<20}{'(too few)':>34}")
            continue
        try:
            p = mannwhitneyu(a, g, alternative="two-sided")[1]
        except ValueError:
            p = float("nan")
        flag = "  <-- differs" if p < 0.05 else ""
        print(f"  {k:<20}{a.median():>12.2f}{g.median():>12.2f}{p:>10.3f}{flag}")

    print("\n" + "=" * 66)
    print("CATEGORICAL PARAMETERS")
    print("=" * 66)
    for k in CATEG:
        va, vg = Counter(A[k]), Counter(G[k])
        keys = [x for x in set(list(va) + list(vg)) if x]
        if not keys or len(keys) > 12:
            print(f"\n  {k}: {len(keys)} distinct values (not tabulated)")
            continue
        print(f"\n  {k}")
        print(f"    {'value':<28}{'A':>6}{'G':>6}")
        for x in sorted(keys, key=lambda z: -(va[z] + vg[z])):
            print(f"    {str(x)[:26]:<28}{va[x]:>6}{vg[x]:>6}")

    print("\n" + "=" * 66)
    print("CAN ACQUISITION ALONE PREDICT SUBTYPE?")
    print("=" * 66)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    X = df[NUMERIC].copy()
    X = X.loc[:, X.notna().sum() >= 0.8 * len(X)]         # drop mostly-missing fields
    X = X.fillna(X.median())
    used = list(X.columns)
    y = (df.subtype == "G").astype(int).values
    if X.shape[1] == 0 or len(set(y)) < 2:
        print("  not enough usable numeric fields")
        return
    print(f"  fields used: {', '.join(used)}")

    Xv = X.values

    def loo_auc(Xa, ya):
        pred = np.zeros(len(ya), float)
        for tr, te in LeaveOneOut().split(Xa):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"))
            m.fit(Xa[tr], ya[tr])
            pred[te] = m.predict_proba(Xa[te])[:, 1]
        return roc_auc_score(ya, pred)

    auc = loo_auc(Xv, y)

    # A fixed threshold would be wrong here. With this sample size the null
    # distribution of leave-one-out AUC has a standard deviation near 0.10, so
    # values above 0.6 arise by chance often. The observed value is therefore
    # compared against labels shuffled the same way.
    print(f"\n  observed leave-one-out AUC: {auc:.3f}")
    print(f"  building the permutation null ", end="", flush=True)
    nperm = max(20, args.permutations)   # an empty null is not usable
    rng = np.random.default_rng(0)
    null = []
    for i in range(nperm):
        yp = rng.permutation(y)
        null.append(loo_auc(Xv, yp))
        if (i + 1) % 50 == 0:
            print(".", end="", flush=True)
    null = np.array(null)
    p = float((np.abs(null - 0.5) >= abs(auc - 0.5)).mean())
    print(f" done ({nperm} permutations)")
    print(f"  null: mean {null.mean():.3f}, SD {null.std(ddof=1):.3f}, "
          f"95th pct of |AUC-0.5| = {np.percentile(np.abs(null-0.5), 95):.3f}")
    print(f"  permutation p-value: {p:.3f}")
    print(f"  ({sum(y==0)} adenocarcinoma, {sum(y==1)} squamous)")
    print()
    if p >= 0.05:
        print("  NO ACQUISITION SIGNATURE detected. Subtype is not predictable from")
        print("  scanner settings beyond chance, so the image models are not")
        print("  separating the classes by protocol. Report as a measured negative")
        print("  in the limitations rather than an untested assumption.")
    else:
        print("  ACQUISITION SIGNATURE PRESENT. Scanner settings alone separate the")
        print("  classes more than chance allows. Any image classifier may be")
        print("  exploiting this, and the discussion must say so explicitly.")
        print("  Consider reporting performance stratified by protocol.")

    tag = (os.path.splitext(os.path.basename(args.cohort))[0]
           if args.cohort else "full_cohort")
    out = f"scanner_probe_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
