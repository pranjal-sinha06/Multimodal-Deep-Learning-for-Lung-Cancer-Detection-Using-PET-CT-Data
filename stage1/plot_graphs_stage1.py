import os, glob, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "stage1_plots"
os.makedirs(OUT, exist_ok=True)

# discover metrics.csv 
found = {}
for p in glob.glob(os.path.expanduser("~/stage1_results/stage1_*_metrics.csv")):
    tag = re.search(r"stage1_(\w+?)_metrics\.csv", os.path.basename(p)).group(1)
    found.setdefault(tag, p)
for p in glob.glob("runs/stage1_*/metrics.csv"):
    tag = re.search(r"stage1_(\w+)", p).group(1)
    found.setdefault(tag, p)   

found = {t: p for t, p in found.items() if "smoke" not in t}  
if not found:
    raise SystemExit("no metrics.csv found in ~/stage1_results/ or runs/stage1_*/")

print("found runs:", sorted(found))
best = {}

for tag in sorted(found):
    df = pd.read_csv(found[tag])

    # 1) loss curves: total + 4 components
    plt.figure(figsize=(7, 4.2))
    for col, lab in [("loss", "total"), ("loss_classifier", "cls"),
                     ("loss_box_reg", "box_reg"), ("loss_objectness", "obj"),
                     ("loss_rpn_box_reg", "rpn_box")]:
        if col in df:
            plt.plot(df.epoch, df[col], marker=".", label=lab)
    plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title(f"Stage 1 {tag} \u2014 training loss")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"stage1_{tag}_loss.png"), dpi=130); plt.close()

    # 2) mAP curves: mAP@0.5 and mAP@0.5:0.95, mark best epoch by val_map
    plt.figure(figsize=(7, 4.2))
    plt.plot(df.epoch, df.val_map_50, marker="o", label="val mAP@0.5", color="#2E75B6")
    plt.plot(df.epoch, df.val_map, marker="s", label="val mAP@0.5:0.95", color="#9C6500")
    bi = int(df.val_map.idxmax())
    plt.axvline(df.epoch[bi], ls="--", color="grey", alpha=0.6)
    plt.scatter([df.epoch[bi]], [df.val_map_50[bi]], s=90, facecolors="none", edgecolors="red", zorder=5)
    plt.xlabel("epoch"); plt.ylabel("mAP"); plt.title(f"Stage 1 {tag} \u2014 validation mAP")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"stage1_{tag}_map.png"), dpi=130); plt.close()

    best[tag] = (float(df.val_map_50[bi]), float(df.val_map[bi]), int(df.epoch[bi]))
    print(f"{tag}: best epoch {best[tag][2]}  mAP@0.5 {best[tag][0]:.3f}  mAP@0.5:0.95 {best[tag][1]:.3f}")

# 3) cross-run comparison bar
tags = sorted(best)
import numpy as np
x = np.arange(len(tags)); w = 0.38
plt.figure(figsize=(max(6, 1.2 * len(tags)), 4.2))
plt.bar(x - w/2, [best[t][0] for t in tags], w, label="mAP@0.5", color="#2E75B6")
plt.bar(x + w/2, [best[t][1] for t in tags], w, label="mAP@0.5:0.95", color="#9C6500")
plt.xticks(x, tags); plt.ylabel("best validation mAP"); plt.title("Stage 1 \u2014 best mAP by run")
plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "stage1_map_comparison.png"), dpi=130); plt.close()

print(f"\nwrote plots to {OUT}/ (per-run loss + mAP, plus stage1_map_comparison.png)")
