import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = "figures/stage1/diag_train_val.csv"
OUT = "figures/stage1"
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(CSV)
print(d.to_string(index=False))
print()

PANELS = [("best", "best epoch (selected checkpoint)"),
          ("last", "final epoch (end of training)")]

for metric, lab in [("map50", "mAP@0.5"), ("map", "mAP@0.5:0.95")]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (which, title) in zip(axes, PANELS):
        sub = d[d.which == which].sort_values("run")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub)); w = 0.38
        tr = sub[f"train_{metric}"].values
        va = sub[f"val_{metric}"].values
        ax.bar(x - w / 2, tr, w, label="train", color="#4C72B0")
        ax.bar(x + w / 2, va, w, label="validation", color="#DD8452")
        for i, (t, v) in enumerate(zip(tr, va)):
            ax.text(i, max(t, v) + 0.035, f"{t - v:+.3f}", ha="center",
                    fontsize=9, color="#C44E52", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(ep {e})" for r, e in zip(sub.run, sub.epoch)])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel(lab)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle(f"Stage 1 generalisation gap: train vs validation {lab}\n"
                 f"red = train \u2212 val   |   identical fixed subsets (N=2000, seed 0), augmentation off",
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUT, f"stage1_gap_{metric}.png")
    plt.savefig(p, dpi=140); plt.close()
    print("wrote", p)

# gap-only summary: does the gap widen from best to last?
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for which, c, m in [("best", "#4C72B0", "o"), ("last", "#C44E52", "s")]:
    sub = d[d.which == which].sort_values("run")
    if len(sub) == 0:
        continue
    ax.plot(sub.run, sub.gap_map50, marker=m, color=c, ms=8, label=f"{which} epoch")
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("train \u2212 val   (mAP@0.5)")
ax.set_title("Stage 1 generalisation gap across runs\n"
             "a gap invariant to the lever pulled indicates a data ceiling, not a tuning failure",
             fontsize=10)
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
p = os.path.join(OUT, "stage1_gap_summary.png")
plt.savefig(p, dpi=140); plt.close()
print("wrote", p)
