"""
Collects the Stage 2 seed sweep and answers one question: which ablation
deltas survive run-to-run noise?

TWO GRANULARITIES, from two sources:
  per-crop   <- stage2_runs/<tag>/metrics.csv   (what best.pth is selected on)
  per-lesion <- figures/stage2/<tag>/scores_val.csv  (written by eval_stage2.py;
                this is the 0.559/0.575 number the dissertation reports)

Run sweep_train.slurm, then sweep_eval.slurm, then this.

Runs are reproducible: the dataset RNG is seeded from torch's per-worker
generator and cudnn is deterministic. Verify with verify_repro.slurm before
trusting these numbers. What is measured here is therefore genuine seed
variance, and it is the noise floor against which ablation deltas are judged.
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

OUT = "figures/stage2/seed_sweep"
LABELS = {"run1": "run1\nbaseline", "run2": "run2\n+sampler",
          "run3": "run3\n+aug", "run4": "run4\n+wd", "run5": "run5\nfreeze"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["run1", "run2", "run3", "run4", "run5"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val is correct for model selection; test only for the final table")
    ap.add_argument("--runs-dir", default="stage2_runs")
    ap.add_argument("--fig-dir", default="figures/stage2")
    return ap.parse_args()


def per_lesion(scores_csv):
    """Reproduces eval_stage2.py exactly: mean softmax per patient -> argmax."""
    df = pd.read_csv(scores_csv)
    g = df.groupby("patient").agg(p0=("p0", "mean"), p1=("p1", "mean"),
                                  p2=("p2", "mean"), true=("label_idx", "first"))
    pred = g[["p0", "p1", "p2"]].values.argmax(1)
    y = g.true.values
    return {"lesion_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "lesion_acc": accuracy_score(y, pred),
            "n_patients": len(g)}


def load(args):
    rows, missing = [], []
    for run in args.runs:
        for s in args.seeds:
            tag = f"{run}_s{s}"
            rec = {"run": run, "seed": s, "tag": tag}

            m = os.path.join(args.runs_dir, tag, "metrics.csv")
            if os.path.exists(m):
                df = pd.read_csv(m)
                b = df.loc[df.val_macro_f1.idxmax()]
                rec.update(crop_macro_f1=b.val_macro_f1, best_epoch=int(b.epoch),
                           n_epochs=len(df))
            else:
                missing.append(m); continue

            sc = os.path.join(args.fig_dir, tag, f"scores_{args.split}.csv")
            if os.path.exists(sc):
                rec.update(per_lesion(sc))
            else:
                missing.append(sc)
            rows.append(rec)
    return pd.DataFrame(rows), missing


def strip_plot(df, metric, ylabel, fname, args):
    runs = [r for r in args.runs if r in set(df.run)]
    fig, ax = plt.subplots(figsize=(1.7 * len(runs) + 2.2, 4.6))
    base = df.loc[df.run == "run1", metric]
    if len(base):
        ax.axhline(base.mean(), ls="--", lw=1, color="#888", zorder=1)
        ax.axhspan(base.mean() - base.std(ddof=1), base.mean() + base.std(ddof=1),
                   color="#888", alpha=0.11, zorder=0)
    stats = []
    for i, run in enumerate(runs):
        v = df.loc[df.run == run, metric].dropna().values
        if not len(v):
            continue
        x = np.random.RandomState(0).normal(i, 0.05, len(v))
        ax.scatter(x, v, s=46, zorder=3, alpha=0.85, edgecolor="white", linewidth=0.6)
        m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        ax.hlines(m, i - 0.24, i + 0.24, lw=2.2, color="#222", zorder=4)
        stats.append((i, m, sd))
    # annotate only after every point is plotted, else get_ylim() shifts mid-loop
    for i, m, sd in stats:
        ax.text(i, 0.015, f"{m:.3f}\n±{sd:.3f}", ha="center", va="bottom",
                fontsize=8, color="#222", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([LABELS.get(r, r) for r in runs], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Stage 2 ablations across {len(args.seeds)} independent runs\n"
                 f"dashed = run1 mean, band = run1 ±1 SD", fontsize=10)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  wrote {p}")


def verdicts(df, metric, args):
    base = df.loc[df.run == "run1", metric].dropna()
    if len(base) < 2:
        return
    sd = base.std(ddof=1)
    print(f"\n=== verdicts on {metric} (noise floor: run1 SD = {sd:.4f}) ===")
    print(f"{'run':<6} {'mean':>8} {'delta':>8} {'|delta|/SD':>11}  verdict")
    for run in args.runs:
        v = df.loc[df.run == run, metric].dropna()
        if not len(v):
            continue
        d = v.mean() - base.mean()
        z = abs(d) / sd if sd > 0 else np.inf
        if run == "run1":
            verdict = "(baseline)"
        elif z < 1:
            verdict = "indistinguishable from noise"
        elif z < 2:
            verdict = "weak, not conclusive"
        else:
            verdict = "real effect"
        print(f"{run:<6} {v.mean():>8.4f} {d:>+8.4f} {z:>11.2f}  {verdict}")
    print("\n  Rule of thumb: |delta| under ~2 SD is not separable from run-to-run noise.")


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    df, missing = load(args)

    if missing:
        print(f"MISSING {len(missing)} file(s), first few:")
        for m in missing[:6]:
            print("   ", m)
        if any("scores_" in m for m in missing):
            print("  -> scores_*.csv come from sweep_eval.slurm. Has it run?")
        print()

    if len(df) < 2:
        print("Not enough completed runs to compare. Have the array jobs finished?")
        return

    print("=== per-run, per-seed ===")
    cols = [c for c in ["run", "seed", "best_epoch", "crop_macro_f1",
                        "lesion_macro_f1", "lesion_acc", "n_patients"] if c in df]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== spread within each run ===")
    for metric in ["crop_macro_f1", "lesion_macro_f1"]:
        if metric not in df:
            continue
        print(f"\n  {metric}")
        print(f"  {'run':<6} {'n':>3} {'mean':>8} {'SD':>8} {'min':>8} {'max':>8}")
        for run in args.runs:
            v = df.loc[df.run == run, metric].dropna().values
            if not len(v):
                continue
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"  {run:<6} {len(v):>3} {v.mean():>8.4f} {sd:>8.4f} "
                  f"{v.min():>8.4f} {v.max():>8.4f}")

    # filenames carry the split, else a --split test run silently overwrites val
    for metric, ylab, fn in [
            ("lesion_macro_f1", f"{args.split} per-lesion macro-F1",
             f"sweep_{args.split}_lesion.png"),
            ("crop_macro_f1", f"{args.split} per-crop macro-F1",
             f"sweep_{args.split}_crop.png")]:
        if metric in df and df[metric].notna().any():
            strip_plot(df, metric, ylab, fn, args)

    if "lesion_macro_f1" in df and df.lesion_macro_f1.notna().any():
        verdicts(df, "lesion_macro_f1", args)
    else:
        print("\nNo per-lesion scores yet. Verdicts on per-crop instead:")
        verdicts(df, "crop_macro_f1", args)

    csv_path = os.path.join(OUT, f"sweep_{args.split}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    print("\nThese are seed variance under a FIXED patient split. They do not")
    print("capture split uncertainty, which with 3 small cell test patients is the")
    print("larger source. Do not report them as confidence intervals on performance.")


if __name__ == "__main__":
    main()
