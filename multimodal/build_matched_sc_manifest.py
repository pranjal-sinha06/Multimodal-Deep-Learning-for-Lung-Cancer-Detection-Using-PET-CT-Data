"""
build_matched_sc_manifest.py  --  control for annotation density.

WHY
  On the comparison cohort the Secondary Capture annotations are denser than the
  CT ones. If the SC arm scores higher, part of that could be more training
  boxes rather than a better representation, and the objection cannot be
  answered after the fact.

  This writes a second SC manifest subsampled so each patient contributes the
  same number of boxes as they do in the CT arm. Running the SC model on it
  isolates the representation from the annotation density.

DESIGN
  Matching is per patient, not global. A global match would preserve the total
  while distorting which patients dominate the training set.

  Patients whose SC boxes are already fewer than their CT boxes are kept whole;
  they cannot be matched upward and are reported separately.

  The subsample is drawn once with a fixed seed and written to disk, so every
  fold and every training seed sees the identical subset. Drawing it per run
  would add a second source of variance and confound the comparison it exists
  to clean up.

SAFETY
  Reads  sc_crops_manifest.csv, stage2_crops_manifest.csv, pet_cohort_83.csv
  Writes sc_crops_manifest_matched.csv   (new file; nothing is overwritten)

USAGE
  python build_matched_sc_manifest.py
  python build_matched_sc_manifest.py --seed 0
"""
import os, sys, argparse
import numpy as np, pandas as pd

LUNG    = "/sharedscratch/ps306/lung"
SC_MAN  = os.path.join(LUNG, "sc_crops_manifest.csv")
CT_MAN  = os.path.join(LUNG, "stage2_crops_manifest.csv")
COHORT  = os.path.join(LUNG, "pet_cohort_83.csv")
OUT     = os.path.join(LUNG, "sc_crops_manifest_matched.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed for the subsample; changing it changes the control")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    for p in (SC_MAN, CT_MAN, COHORT):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite an existing file: {args.out}\n"
                 f"delete it first if you intend to rebuild the control.")

    cohort = set(pd.read_csv(COHORT).patient)
    sc = pd.read_csv(SC_MAN)
    ct = pd.read_csv(CT_MAN)

    # both arms see A/G within the locked cohort, so match on exactly that
    sc = sc[sc.patient.isin(cohort) & sc.subtype.isin(["A", "G"])].copy()
    ct = ct[ct.patient.isin(cohort) & ct.subtype.isin(["A", "G"])]

    n_ct = ct.groupby("patient").size()
    n_sc = sc.groupby("patient").size()
    print(f"cohort patients: {len(cohort)}")
    print(f"  CT boxes {len(ct):>6}   across {n_ct.size} patients")
    print(f"  SC boxes {len(sc):>6}   across {n_sc.size} patients")
    print(f"  SC / CT box ratio: {len(sc)/len(ct):.2f}\n")

    rng = np.random.default_rng(args.seed)
    keep, short, exact = [], [], 0
    for p, g in sc.groupby("patient"):
        target = int(n_ct.get(p, 0))
        if target == 0:
            continue                       # patient absent from the CT arm
        if len(g) <= target:
            keep.append(g)
            if len(g) < target:
                short.append((p, len(g), target))
            else:
                exact += 1
        else:
            idx = rng.choice(len(g), size=target, replace=False)
            keep.append(g.iloc[np.sort(idx)])

    out = pd.concat(keep).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    n_out = out.groupby("patient").size()
    print(f"wrote {args.out}")
    print(f"  boxes {len(out)} (from {len(sc)}), slices {out.sop.nunique()}, "
          f"patients {out.patient.nunique()}")
    print(f"  target was {int(n_ct.reindex(n_out.index).sum())} "
          f"(the CT count on the same patients)")

    common = n_out.index.intersection(n_ct.index)
    diff = (n_out[common] - n_ct[common])
    print(f"\n  patients matched exactly     : {int((diff == 0).sum())}")
    print(f"  patients short of the target : {len(short)}")
    if short:
        print("    (SC has fewer boxes than CT for these; kept whole)")
        for p, have, want in short[:8]:
            print(f"      {p:<20} {have} of {want}")
    print(f"  largest remaining shortfall  : {int(diff.min()) if len(diff) else 0}")

    pat = out.drop_duplicates("patient")
    print("\n  class balance after matching:")
    print("   ", pat.subtype.value_counts().to_dict())
    print("\nNext: python train_sc_matched_cv.py --seed 0   "
          "(writes figures/petct/sc_matched_*)")


if __name__ == "__main__":
    main()
