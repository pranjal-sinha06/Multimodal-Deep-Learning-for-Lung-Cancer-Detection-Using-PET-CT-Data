"""
plot_metrics.py  --  per-arm, per-metric and confusion-matrix figures.

Reads the fold_predictions.csv files each arm already wrote (columns
patient, y, p0, p1) and produces three families of figure in a new folder,
figures/petct/metric_figures/. Nothing existing is touched.

All figures are PATIENT LEVEL: each patient's crop probabilities are averaged
and then thresholded once, matching the cm_patient convention used elsewhere.
Crop-level numbers look far tighter than the evidence supports, because the
~4,600 crops come from only 83 patients, so they are not plotted here.

Everything is computed per seed and then summarised across seeds, so the spread
that the seed protocol exists to measure is visible in every figure.

Metrics, with squamous cell carcinoma as the positive class where a positive
class is needed:
  accuracy                overall
  precision, recall, F1   reported for BOTH classes, since with 63 vs 20
                          patients a single averaged value hides the minority
                          class, which is exactly where the model struggles.

    python plot_metrics.py            # nested runs (default)
    python plot_metrics.py fixed      # the --fixed-epochs runs

Figures written:
  metrics_<arm>.png       one per arm: metrics on the x-axis, bar = seed mean,
                          dot = each seed
  by_metric_<metric>.png  one per metric: arms on the x-axis, same seed dots
  cm_<arm>.png            one per arm: seed-averaged confusion matrix, counts
                          shown as mean +- SD across seeds
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

OUTDIR = "figures/petct/metric_figures"
ARMS = [
    ("ct_only",    "CT-only",            "#4C72B0"),
    ("ct_pet",     "CT+PET",             "#C44E52"),
    ("sc",         "Secondary Capture",  "#55A868"),
    ("sc_matched", "SC density-matched", "#8172B2"),
]
CLASSES = ["Adenocarcinoma", "Squamous cell carcinoma"]   # index 0, 1
POS = 1                                                    # squamous is positive

# the metrics, in plot order; each entry is (key, label, function)
METRICS = [
    ("accuracy",       "Accuracy",        lambda y, p: accuracy_score(y, p)),
    ("precision_adeno","Precision (Adeno)",lambda y, p: precision_score(y, p, pos_label=0, zero_division=0)),
    ("recall_adeno",   "Recall (Adeno)",   lambda y, p: recall_score(y, p, pos_label=0, zero_division=0)),
    ("f1_adeno",       "F1 (Adeno)",       lambda y, p: f1_score(y, p, pos_label=0, zero_division=0)),
    ("precision_squam","Precision (Squamous)",lambda y, p: precision_score(y, p, pos_label=1, zero_division=0)),
    ("recall_squam",   "Recall (Squamous)",lambda y, p: recall_score(y, p, pos_label=1, zero_division=0)),
    ("f1_squam",       "F1 (Squamous)",    lambda y, p: f1_score(y, p, pos_label=1, zero_division=0)),
]


def patient_level(csv):
    """Return (y_true, y_pred) at patient level from one prediction file."""
    df = pd.read_csv(csv)
    g = df.groupby("patient").agg(y=("y", "first"), p1=("p1", "mean"))
    y = g.y.astype(int).values
    pred = (g.p1.values >= 0.5).astype(int)
    return y, pred


def arm_runs(arm, mode):
    """Per-seed (seed, y_true, y_pred) for one arm; empty if it never ran."""
    out = []
    for f in sorted(glob.glob(f"figures/petct/{arm}_{mode}_s*/fold_predictions.csv")):
        seed = f.split("_s")[-1].split("/")[0]
        y, pred = patient_level(f)
        out.append((seed, y, pred))
    return out


def metrics_table(mode):
    """Nested dict: arm -> metric_key -> array over seeds. Also the CMs."""
    table, cms, present = {}, {}, []
    for arm, label, _ in ARMS:
        runs = arm_runs(arm, mode)
        if not runs:
            continue
        present.append((arm, label))
        table[arm] = {k: [] for k, _, _ in METRICS}
        mats = []
        for _, y, pred in runs:
            for k, _, fn in METRICS:
                table[arm][k].append(fn(y, pred))
            mats.append(confusion_matrix(y, pred, labels=[0, 1]))
        table[arm] = {k: np.array(v) for k, v in table[arm].items()}
        cms[arm] = np.stack(mats)          # (seeds, 2, 2)
    return table, cms, present


def fig_per_arm(table, present):
    for arm, label in present:
        vals = table[arm]
        xs = np.arange(len(METRICS))
        means = [vals[k].mean() for k, _, _ in METRICS]
        fig, ax = plt.subplots(figsize=(9, 5))
        colour = dict(ARMS_c)[arm]
        ax.bar(xs, means, color=colour, alpha=0.55, width=0.6, zorder=2)
        for i, (k, _, _) in enumerate(METRICS):
            v = vals[k]
            ax.scatter(np.full_like(v, i, dtype=float), v, color="black",
                       s=22, zorder=3, alpha=0.8)
        for i, m in enumerate(means):
            ax.text(i, min(m + 0.03, 0.98), f"{m:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab, _ in METRICS], rotation=30, ha="right",
                           fontsize=8.5)
        ax.set_ylim(0, 1.05); ax.set_ylabel("score")
        ax.set_title(f"{label}: patient-level metrics over {len(vals['accuracy'])} seeds\n"
                     "bar = mean, dot = individual seed", fontsize=11)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout()
        p = f"{OUTDIR}/metrics_{arm}.png"
        plt.savefig(p, dpi=150); plt.close()
        print(f"  wrote {p}")


def fig_per_metric(table, present):
    for k, label, _ in METRICS:
        arms = [a for a, _ in present]
        labels = [l for _, l in present]
        means = [table[a][k].mean() for a in arms]
        xs = np.arange(len(arms))
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for i, a in enumerate(arms):
            ax.bar(i, means[i], color=dict(ARMS_c)[a], alpha=0.55, width=0.6, zorder=2)
            v = table[a][k]
            ax.scatter(np.full_like(v, i, dtype=float), v, color="black",
                       s=22, zorder=3, alpha=0.8)
            ax.text(i, min(means[i] + 0.03, 0.98), f"{means[i]:.2f}",
                    ha="center", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_ylabel(label)
        ax.set_title(f"{label}, patient level, by input representation\n"
                     "bar = mean over seeds, dot = individual seed", fontsize=11)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout()
        p = f"{OUTDIR}/by_metric_{k}.png"
        plt.savefig(p, dpi=150); plt.close()
        print(f"  wrote {p}")


def fig_confusion(cms, present):
    for arm, label in present:
        M = cms[arm]                        # (seeds, 2, 2)
        mean, sd = M.mean(0), M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros_like(M[0])
        fig, ax = plt.subplots(figsize=(5.6, 5))
        im = ax.imshow(mean, cmap="Blues")
        for i in range(2):
            for j in range(2):
                txt = f"{mean[i,j]:.1f}\n$\\pm$ {sd[i,j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                        color="white" if mean[i, j] > mean.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASSES, fontsize=9); ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{label}: patient-level confusion matrix\n"
                     f"mean $\\pm$ SD over {M.shape[0]} seeds", fontsize=11)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        p = f"{OUTDIR}/cm_{arm}.png"
        plt.savefig(p, dpi=150); plt.close()
        print(f"  wrote {p}")


ARMS_c = [(a, c) for a, _, c in ARMS]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nested"
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"mode: {mode}")
    table, cms, present = metrics_table(mode)
    if not present:
        sys.exit(f"no runs found for mode '{mode}'. Use: python plot_metrics.py [nested|fixed]")
    for arm, label, _ in ARMS:
        n = len(table[arm]["accuracy"]) if arm in table else 0
        print(f"  {label:<20} {n} seed(s)" + ("" if n else "   (not run)"))
    print()
    fig_per_arm(table, present)
    fig_per_metric(table, present)
    fig_confusion(cms, present)
    print(f"\nall figures in {OUTDIR}/")
    print("These are patient level. The crop-level equivalents are not plotted:")
    print("thousands of crops from 83 patients make them look falsely precise.")


if __name__ == "__main__":
    main()
