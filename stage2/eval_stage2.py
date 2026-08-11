import os, json, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import (f1_score, accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, precision_recall_curve, average_precision_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from stage2_dataset import LungSubtypeDataset

CLASSES     = ["A", "B", "G"]
CLASS_FULL  = ["Adenocarcinoma", "Small cell carcinoma", "Squamous cell carcinoma"]
CLASS_SHORT = ["Adenocarcinoma", "Small cell", "Squamous"]
COLORS      = ["#4C72B0", "#DD8452", "#C44E52"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag, e.g. run1")
    return ap.parse_args()


args = parse_args()
CKPT = f"stage2_runs/{args.run}/best.pth"
OUT  = f"figures/stage2/{args.run}"
os.makedirs(OUT, exist_ok=True)

device = torch.device("cuda")
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
model.to(device).eval()
print(f"loaded {CKPT}")


@torch.no_grad()
def infer(split):
    ds = LungSubtypeDataset(split)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)   # order matches ds.df
    probs = []
    for imgs, _ in dl:
        probs.append(torch.softmax(model(imgs.to(device)), 1).cpu())
    probs = torch.cat(probs).numpy()
    df = ds.df.copy()
    df["p0"], df["p1"], df["p2"] = probs[:, 0], probs[:, 1], probs[:, 2]
    df["pred_crop"] = probs.argmax(1)
    return df


def report(name, y, p):
    acc = accuracy_score(y, p); mf1 = f1_score(y, p, average="macro", zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    print(f"\n[{name}]  accuracy {acc:.3f}  macro-F1 {mf1:.3f}")
    for i, c in enumerate(CLASSES):
        print(f"   {c} ({CLASS_SHORT[i]}): precision {pr[i]:.3f}  recall {rc[i]:.3f}  F1 {f1[i]:.3f}")
    return acc, mf1, (pr, rc, f1)


def confmat_png(y, p, title, path):
    """Row-normalised percentage heatmap; each cell shows count and % of its true class."""
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    row_tot = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_tot, out=np.zeros_like(cm, dtype=float), where=row_tot > 0)

    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_FULL, rotation=25, ha="right")
    ax.set_yticklabels(CLASS_FULL)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\n(row-normalised: diagonal = per-class recall)", fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=10)
    # n per true class, so small denominators stay visible
    for i in range(3):
        ax.text(2.75, i, f"n={int(row_tot[i, 0])}", va="center", fontsize=8, color="#595959")
    fig.colorbar(im, fraction=0.046, label="% of true class")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def metric_bar(vals, metric, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_SHORT, vals, color=COLORS)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_ylabel(metric); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def pr_and_f1_curves(df, split):
    """One-vs-rest ranking curves from the saved softmax scores (per-crop only)."""
    y = df.label_idx.values
    P = df[["p0", "p1", "p2"]].values

    # precision-recall
    plt.figure(figsize=(6.5, 5))
    for i, c in enumerate(CLASSES):
        y_bin = (y == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, P[:, i])
        ap = average_precision_score(y_bin, P[:, i])
        plt.plot(rec, prec, color=COLORS[i],
                 label=f"{CLASS_SHORT[i]} (AP={ap:.3f}, n={int(y_bin.sum())})")
        plt.axhline(y_bin.mean(), color=COLORS[i], ls=":", alpha=0.4)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{split} per-crop: one-vs-rest PR curves\n"
              f"(dotted = class prevalence baseline)", fontsize=11)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"pr_curve_{split}_crop.png"), dpi=150); plt.close()

    # F1 vs threshold
    plt.figure(figsize=(6.5, 5))
    for i, c in enumerate(CLASSES):
        y_bin = (y == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_bin, P[:, i])
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1t = f1[:-1]                      # align with thresholds
        if len(thr) == 0:
            continue
        k = int(np.argmax(f1t))
        plt.plot(thr, f1t, color=COLORS[i],
                 label=f"{CLASS_SHORT[i]} (best F1={f1t[k]:.3f} @ {thr[k]:.3f})")
        plt.scatter([thr[k]], [f1t[k]], color=COLORS[i], s=45, zorder=5)
    plt.xlabel("Score threshold"); plt.ylabel("F1")
    plt.title(f"{split} per-crop: one-vs-rest F1 vs threshold", fontsize=11)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"f1_curve_{split}_crop.png"), dpi=150); plt.close()


for split in ["val", "test"]:
    df = infer(split)

    # ---- per-crop ----
    acc, mf1, (pr, rc, f1) = report(f"{split} per-crop", df.label_idx.values, df.pred_crop.values)
    confmat_png(df.label_idx.values, df.pred_crop.values,
                f"{args.run} {split} per-crop", os.path.join(OUT, f"cm_{split}_crop.png"))
    for vals, m in [(pr, "Precision"), (rc, "Recall"), (f1, "F1")]:
        metric_bar(vals, m, f"{args.run} {split} per-crop: {m.lower()} by class",
                   os.path.join(OUT, f"{m.lower()}_{split}_crop.png"))

    # ---- per-lesion: mean of probabilities per patient -> argmax ----
    g = df.groupby("patient").agg(p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
                                  true=("label_idx", "first"))
    pred_les = g[["p0", "p1", "p2"]].values.argmax(1)
    acc, mf1, (pr, rc, f1) = report(f"{split} per-lesion", g.true.values, pred_les)
    print(f"   ({len(g)} patients aggregated)")
    confmat_png(g.true.values, pred_les,
                f"{args.run} {split} per-lesion", os.path.join(OUT, f"cm_{split}_lesion.png"))
    for vals, m in [(pr, "Precision"), (rc, "Recall"), (f1, "F1")]:
        metric_bar(vals, m, f"{args.run} {split} per-lesion: {m.lower()} by class",
                   os.path.join(OUT, f"{m.lower()}_{split}_lesion.png"))

    # ---- ranking curves: per-crop only (per-lesion n is far too small) ----
    pr_and_f1_curves(df, split)

    # keep the scores so curves can be redrawn without re-running inference
    df[["patient", "sop", "label_idx", "p0", "p1", "p2", "pred_crop"]].to_csv(
        os.path.join(OUT, f"scores_{split}.csv"), index=False)

print(f"\nfigures + scores written to {OUT}")
