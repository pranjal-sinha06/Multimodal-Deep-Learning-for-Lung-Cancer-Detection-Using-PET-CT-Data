"""
compare_arms.py  -  CT-only vs CT+PET vs Secondary Capture.

Three representations, the same 83 patients, the same folds, the same seeds and
the same ResNet-50. The only thing that varies is what the three input channels
contain:

    CT-only  [lung window, mediastinal window, wide window]
    CT+PET   [lung window, mediastinal window, SUV]
    SC       [R, G, B] of the workstation-fused PET/CT render


    python compare_arms.py            # nested runs
    python compare_arms.py fixed      # the --fixed-epochs runs
"""
import sys, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ARMS = [("ct_only", "CT-only"), ("ct_pet", "CT+PET"), ("sc", "Secondary Capture"),
        ("sc_matched", "SC density-matched")]


def patient_auc(csv):
    df = pd.read_csv(csv)
    g = df.groupby("patient").agg(y=("y", "first"), p1=("p1", "mean"))
    return roc_auc_score(g.y, g.p1) if g.y.nunique() > 1 else np.nan


def arm_aucs(arm, mode):
    out = {}
    for f in sorted(glob.glob(f"figures/petct/{arm}_{mode}_s*/fold_predictions.csv")):
        seed = f.split("_s")[-1].split("/")[0]
        out[seed] = patient_auc(f)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nested"
    print(f"mode: {mode}\n")

    data = {a: arm_aucs(a, mode) for a, _ in ARMS}
    present = [(a, lab) for a, lab in ARMS if data[a]]
    if not present:
        sys.exit(f"no runs found for mode '{mode}'. Use: python compare_arms.py [nested|fixed]")
    for a, lab in ARMS:
        if not data[a]:
            print(f"  ({lab} not run yet)")

    seeds = sorted(set.intersection(*[set(data[a]) for a, _ in present]))
    if not seeds:
        sys.exit("no seeds common to all arms present")

    print("=== patient-level AUC per seed ===")
    hdr = f"{'seed':<6}" + "".join(f"{lab:>21}" for _, lab in present)
    print(hdr)
    for s in seeds:
        print(f"{s:<6}" + "".join(f"{data[a][s]:>21.3f}" for a, _ in present))

    print("\n=== summary, against the CT-only baseline ===")
    base = np.array([data["ct_only"][s] for s in seeds]) if data["ct_only"] else None
    for a, lab in present:
        v = np.array([data[a][s] for s in seeds])
        m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))
        line = f"  {lab:<22} {m:.3f} +- {sd:.3f}"
        if base is not None and a != "ct_only":
            d = v - base
            dm, dsd = d.mean(), (d.std(ddof=1) if len(d) > 1 else float("nan"))
            verdict = ("no effect" if not np.isfinite(dsd) or dsd == 0 or abs(dm) < 2 * dsd
                       else "EFFECT (over 2 SD)")
            line += (f"   delta {dm:+.3f} (SD {dsd:.3f}, "
                     f"{abs(dm)/dsd:.2f} SD)  {verdict}" if np.isfinite(dsd) and dsd > 0
                     else f"   delta {dm:+.3f}")
            line += f"   [{sum(d < 0)}/{len(d)} seeds lower]"
        print(line)

    print("\n  Threshold: |delta| under about 2 SD of the seed-level differences is")
    print("  not separable from run-to-run variation (Section 4.3.5 protocol).")
    print("\n  Context on this dataset: Aksu et al. 2025 report CT 0.489, PET 0.465,")
    print("  early fusion 0.452, and 0.681 for their multi-stage intermediate fusion.")


if __name__ == "__main__":
    main()
