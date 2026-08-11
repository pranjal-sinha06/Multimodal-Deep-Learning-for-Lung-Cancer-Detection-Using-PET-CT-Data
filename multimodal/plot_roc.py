"""
plot_roc.py -ROC curves for every arm, from the predictions

"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

OUTDIR = "figures/petct"
ARMS = [
    ("ct_only",    "CT-only",            "#4C72B0"),
    ("ct_pet",     "CT+PET",             "#C44E52"),
    ("sc",         "Secondary Capture",  "#55A868"),
    ("sc_matched", "SC density-matched", "#8172B2"),
]
POSITIVE = "Squamous cell carcinoma"      # label 1; adenocarcinoma is the negative
GRID = np.linspace(0, 1, 201)


def load(arm, mode, level):
    """Per-seed (seed, y, score) for one arm, or an empty list if it never ran."""
    out = []
    for f in sorted(glob.glob(f"{OUTDIR}/{arm}_{mode}_s*/fold_predictions.csv")):
        df = pd.read_csv(f)
        if level == "patient":
            g = df.groupby("patient").agg(y=("y", "first"), p1=("p1", "mean"))
            y, s = g.y.values, g.p1.values
        else:
            y, s = df.y.values, df.p1.values
        if len(set(y)) > 1:
            out.append((f.split("_s")[-1].split("/")[0], y, s))
    return out


def mean_curve(runs):
    """Vertical average of the per-seed ROC curves, plus the AUC spread."""
    tprs, aucs = [], []
    for _, y, s in runs:
        fpr, tpr, _ = roc_curve(y, s)
        t = np.interp(GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(roc_auc_score(y, s))
    T = np.vstack(tprs)
    m = T.mean(0); m[-1] = 1.0
    sd = T.std(0, ddof=1) if len(T) > 1 else np.zeros_like(m)
    return m, sd, np.array(aucs)


def plot_mean(mode, level):
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, zorder=1,
            label="Chance (AUC = 0.500)")
    n = None
    for arm, label, colour in ARMS:
        runs = load(arm, mode, level)
        if not runs:
            continue
        m, sd, aucs = mean_curve(runs)
        n = len(runs[0][1])
        ax.fill_between(GRID, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                        color=colour, alpha=0.13, lw=0, zorder=2)
        ax.plot(GRID, m, color=colour, lw=2.2, zorder=3,
                label=f"{label}: AUC {aucs.mean():.3f} ± "
                      f"{aucs.std(ddof=1) if len(aucs) > 1 else 0:.3f} "
                      f"({len(aucs)} seeds)")
    if n is None:
        return False
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    lvl = "Patient level" if level == "patient" else "Crop level"
    ax.set_title(f"{lvl}: {POSITIVE} vs Adenocarcinoma, n = {n}\n"
                 f"pooled out-of-fold; mean of seeds, shaded ±1 SD", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    p = f"{OUTDIR}/roc_{mode}_{level}.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  wrote {p}")
    return True


def plot_seeds(mode, level):
    """Every seed drawn, so the averaging above can be checked against the raw spread."""
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="Chance")
    n = None
    for arm, label, colour in ARMS:
        runs = load(arm, mode, level)
        if not runs:
            continue
        aucs = []
        for _, y, s in runs:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=colour, alpha=0.4, lw=1.1)
            aucs.append(roc_auc_score(y, s))
            n = len(y)
        ax.plot([], [], color=colour, lw=2.2,
                label=f"{label} ({len(aucs)} seeds)")
    if n is None:
        return False
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Patient level, n = {n}: one line per seed", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    p = f"{OUTDIR}/roc_{mode}_{level}_seeds.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  wrote {p}")
    return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nested"
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"mode: {mode}")
    for arm, label, _ in ARMS:
        k = len(load(arm, mode, "patient"))
        print(f"  {label:<20} {k} seed(s)" + ("" if k else "   (not run)"))
    print()
    ok = plot_mean(mode, "patient")
    if not ok:
        sys.exit(f"no runs found for mode '{mode}'. Use: python plot_roc.py [nested|fixed]")
    plot_seeds(mode, "patient")
    plot_mean(mode, "crop")
    print("\nUse roc_{}_patient.png in the chapter.".format(mode))
    print("The _seeds version belongs in the appendix, alongside the crop-level")
    print("figure: crop curves are drawn from thousands of crops but only 83")
    print("patients, so they are tighter than the evidence supports.")


if __name__ == "__main__":
    main()
