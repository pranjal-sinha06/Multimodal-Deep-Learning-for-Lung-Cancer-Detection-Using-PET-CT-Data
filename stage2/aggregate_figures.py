"""
Seed-aggregated Stage 2 figures, for the dissertation.

The sweep produces 5 of every figure per run (one per seed). Putting one
arbitrary seed in Chapter 5 reports a single draw, which is exactly what the
sweep exists to prevent, and contradicts a mean +- SD table.

This reads the scores_<split>.csv that eval_stage2.py already wrote for every
seed and emits ONE figure per run, showing the mean across seeds with its
spread. Same layout and palette as eval_stage2.py so the chapter stays visually
consistent.

Run AFTER sweep_eval.slurm.

    python aggregate_figures.py                    # all runs, val + test
    python aggregate_figures.py --runs run1 --split test
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
                            precision_recall_curve, average_precision_score

CLASSES     = ["A", "B", "G"]
CLASS_FULL  = ["Adenocarcinoma", "Small cell carcinoma", "Squamous cell carcinoma"]
CLASS_SHORT = ["Adenocarcinoma", "Small cell", "Squamous"]
COLORS      = ["#4C72B0", "#DD8452", "#C44E52"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["run1", "run2", "run3", "run4", "run5"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--fig-dir", default="figures/stage2")
    return ap.parse_args()


def load_seed(fig_dir, run, seed, split):
    p = os.path.join(fig_dir, f"{run}_s{seed}", f"scores_{split}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def to_lesion(df):
    """Identical aggregation to eval_stage2.py: mean softmax per patient -> argmax."""
    g = df.groupby("patient").agg(p0=("p0", "mean"), p1=("p1", "mean"),
                                  p2=("p2", "mean"), true=("label_idx", "first"))
    return g.true.values, g[["p0", "p1", "p2"]].values.argmax(1)


def agg_confmat(per_seed, title, path):
    """Mean of row-normalised confusion matrices, with SD and the TRUE n.

    Percentages are averaged across seeds rather than pooling raw counts:
    pooling would print n=250 for a class that really has 50 patients seen
    5 times, which overstates the denominator.
    """
    pcts, n_true = [], None
    for y, p in per_seed:
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        rt = cm.sum(axis=1, keepdims=True)
        pcts.append(np.divide(cm * 100.0, rt, out=np.zeros_like(cm, dtype=float), where=rt > 0))
        n_true = rt.flatten()
    P = np.stack(pcts)
    mean, sd = P.mean(0), P.std(0, ddof=1) if len(P) > 1 else np.zeros_like(P[0])

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    im = ax.imshow(mean, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_FULL, rotation=25, ha="right")
    # n goes in the tick label: at x=2.78 it collides with the colorbar
    ax.set_yticklabels([f"{c}\n(n={int(n_true[i])})" for i, c in enumerate(CLASS_FULL)],
                       fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\nmean of {len(P)} seeds (row-normalised: diagonal = recall)",
                 fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mean[i, j]:.1f}%\n±{sd[i, j]:.1f}", ha="center", va="center",
                    color="white" if mean[i, j] > 50 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, label="% of true class (seed mean)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def agg_bars(per_seed, metric_idx, metric, title, path):
    vals = []
    for y, p in per_seed:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        vals.append([pr, rc, f1][metric_idx])
    V = np.stack(vals)
    mean = V.mean(0); sd = V.std(0, ddof=1) if len(V) > 1 else np.zeros_like(mean)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(CLASS_SHORT, mean, color=COLORS, yerr=sd, capsize=5,
                  error_kw=dict(ecolor="#333", lw=1.2))
    for b, m, s in zip(bars, mean, sd):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.03, f"{m:.3f}\n±{s:.3f}",
                ha="center", fontsize=8)
    # every seed as a dot, so the reader sees the raw spread not just the bar
    for i in range(3):
        ax.scatter(np.full(len(V), i), V[:, i], s=14, color="#222", zorder=5, alpha=0.6)
    ax.set_ylim(0, 1.15); ax.set_ylabel(metric)
    ax.set_title(f"{title}\nmean ± SD over {len(V)} seeds; dots are individual seeds",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def overlay_pr(dfs, title, path):
    """Every seed's PR curve overlaid. Curves are not averaged: each seed has its
    own recall grid, so averaging would need interpolation and invent points."""
    plt.figure(figsize=(6.8, 5))
    for i, c in enumerate(CLASSES):
        aps = []
        for k, df in enumerate(dfs):
            y = (df.label_idx.values == i).astype(int)
            if y.sum() == 0:
                continue
            s = df[["p0", "p1", "p2"]].values[:, i]
            prec, rec, _ = precision_recall_curve(y, s)
            aps.append(average_precision_score(y, s))
            plt.plot(rec, prec, color=COLORS[i], alpha=0.45, lw=1,
                     label=None if k else "_")
        if aps:
            plt.plot([], [], color=COLORS[i], lw=2,
                     label=f"{CLASS_SHORT[i]} (AP={np.mean(aps):.3f}±{np.std(aps, ddof=1):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.ylim(0, 1.02)
    plt.title(f"{title}\n{len(dfs)} seeds overlaid", fontsize=11)
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    args = parse_args()
    made = 0
    for run in args.runs:
        for split in args.splits:
            dfs = [d for d in (load_seed(args.fig_dir, run, s, split) for s in args.seeds)
                   if d is not None]
            if len(dfs) < 2:
                print(f"  skip {run}/{split}: found {len(dfs)} seed(s), need >=2")
                continue
            out = os.path.join(args.fig_dir, f"agg_{run}")
            os.makedirs(out, exist_ok=True)

            crop   = [(d.label_idx.values, d.pred_crop.values) for d in dfs]
            lesion = [to_lesion(d) for d in dfs]

            for gran, per_seed in [("crop", crop), ("lesion", lesion)]:
                agg_confmat(per_seed, f"{run} {split} per-{gran}",
                            os.path.join(out, f"cm_{split}_{gran}.png"))
                for idx, m in [(0, "Precision"), (1, "Recall"), (2, "F1")]:
                    agg_bars(per_seed, idx, m,
                             f"{run} {split} per-{gran}: {m.lower()} by class",
                             os.path.join(out, f"{m.lower()}_{split}_{gran}.png"))
                made += 4
            overlay_pr(dfs, f"{run} {split} per-crop: one-vs-rest PR",
                       os.path.join(out, f"pr_curve_{split}_crop.png"))
            made += 1
            print(f"  {out}/  <- {len(dfs)} seeds, split={split}")
    print(f"\n{made} aggregated figures written.")
    print("Use figures/stage2/agg_<run>/ in the dissertation, not the per-seed dirs.")


if __name__ == "__main__":
    main()
