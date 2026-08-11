"""
fix_sc_manifest.py  --  assign splits and restrict the SC manifest to the cohort.

WHY
  build_sc_manifest.py could not find pet_cohort_flags.csv on the cluster, so it
  fell back to an empty split column. It also kept 133 patients, one more than
  the 132 SC patients inside the 348 working cohort, because nothing constrained
  it to that cohort.

  Both are fixed here using stage2_crops_manifest.csv, which is already on the
  cluster and is the authority for both the cohort membership and the
  patient-level split used throughout the dissertation. Taking the split from
  the same file the CT arm uses guarantees the two arms are aligned: a patient
  in CT's test partition is in SC's test partition.

USAGE
  python fix_sc_manifest.py
"""
import os, sys
import pandas as pd

LUNG = "/sharedscratch/ps306/lung"
SC_IN   = os.path.join(LUNG, "sc_crops_manifest.csv")
CT_MAN  = os.path.join(LUNG, "stage2_crops_manifest.csv")
SC_OUT  = os.path.join(LUNG, "sc_crops_manifest.csv")       # overwritten in place


def main():
    for p in (SC_IN, CT_MAN):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    sc = pd.read_csv(SC_IN)
    ct = pd.read_csv(CT_MAN)

    # the CT manifest defines both the cohort and the split
    cohort = ct[["patient", "split"]].drop_duplicates("patient")
    print(f"cohort from {os.path.basename(CT_MAN)}: {len(cohort)} patients")

    before_p = sc.patient.nunique()
    before_b = len(sc)

    if "split" in sc.columns:
        sc = sc.drop(columns=["split"])
    sc = sc.merge(cohort, on="patient", how="inner")        # inner = drop non-cohort

    dropped_p = before_p - sc.patient.nunique()
    print(f"\ndropped {dropped_p} patient(s) outside the cohort, "
          f"{before_b - len(sc)} box(es)")

    # column order matching the CT manifest, so both are interchangeable
    sc = sc[["patient", "split", "sop", "path", "box", "subtype", "label_idx"]]
    sc.to_csv(SC_OUT, index=False)
    print(f"wrote {SC_OUT}\n")

    print("=== Secondary Capture cohort, split-aligned with the CT arm ===")
    pat = sc.drop_duplicates("patient")
    print(f"  boxes {len(sc)}   slices {sc.sop.nunique()}   patients {len(pat)}")
    print("\n  patients by subtype:")
    print(pat.subtype.value_counts().to_string())
    print("\n  patients by split:")
    print(pat.split.value_counts().to_string())
    print("\n  patients by split and subtype:")
    print(pd.crosstab(pat.split, pat.subtype).to_string())

    print("\n=== how this compares to the CT arm on the same patients ===")
    both = set(sc.patient) & set(ct.patient)
    ct_sub = ct[ct.patient.isin(both)]
    print(f"  patients in both arms : {len(both)}")
    print(f"  CT boxes on them      : {len(ct_sub)}  ({ct_sub.sop.nunique()} slices)")
    print(f"  SC boxes on them      : {len(sc)}  ({sc.sop.nunique()} slices)")

    test = pat[pat.split == "test"]
    print(f"\n  test partition: {len(test)} patients "
          f"({', '.join(f'{k}={v}' for k, v in test.subtype.value_counts().items())})")
    if len(test) < 15:
        print("  NOTE: a test set this small cannot support a stable per-class")
        print("  estimate. Cross-validation over the full SC cohort, as used for")
        print("  the PET arm, is the appropriate protocol here too.")


if __name__ == "__main__":
    main()
