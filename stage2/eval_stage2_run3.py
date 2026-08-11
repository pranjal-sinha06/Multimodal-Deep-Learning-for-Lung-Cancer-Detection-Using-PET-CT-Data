import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from stage2_dataset import LungSubtypeDataset
CKPT = "stage2_runs/run3/best.pth"
OUT = "stage2_runs/run3"
CLASSES = ["A", "B", "G"]
device = torch.device("cuda")

model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
model.to(device).eval()

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
        print(f"   {c}: precision {pr[i]:.3f}  recall {rc[i]:.3f}  F1 {f1[i]:.3f}")
    return acc, mf1, (pr, rc, f1)

def confmat_png(y, p, title, path):
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

for split in ["val", "test"]:
    df = infer(split)

    # per-crop
    report(f"{split} per-crop", df.label_idx.values, df.pred_crop.values)
    confmat_png(df.label_idx.values, df.pred_crop.values,
                f"{split} per-crop", os.path.join(OUT, f"cm_{split}_crop.png"))

    # per-lesion: mean of probabilities per patient -> argmax
    g = df.groupby("patient").agg(p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
                                  true=("label_idx", "first"))
    pred_les = g[["p0", "p1", "p2"]].values.argmax(1)
    acc, mf1, (pr, rc, f1) = report(f"{split} per-lesion", g.true.values, pred_les)
    confmat_png(g.true.values, pred_les,
                f"{split} per-lesion", os.path.join(OUT, f"cm_{split}_lesion.png"))
    print(f"   ({len(g)} patients aggregated)")

    if split == "test":   # per-class F1 bar chart on the headline (test, per-lesion)
        x = np.arange(3); w = 0.25
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w, pr, w, label="precision"); ax.bar(x, rc, w, label="recall"); ax.bar(x + w, f1, w, label="F1")
        ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1)
        ax.set_title("Test per-lesion: per-class metrics"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "test_lesion_perclass.png"), dpi=150); plt.close()

print("\nplots written to", OUT)
