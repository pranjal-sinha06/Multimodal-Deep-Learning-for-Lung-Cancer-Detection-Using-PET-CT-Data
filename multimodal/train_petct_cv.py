"""
train_petct_cv.py  --  CT+PET arm on the PET cohort, 5-fold patient-level CV.

The CT half of the CT-vs-CT+PET experiment. Reproducible (seeded RNG +
deterministic cudnn, the Stage 2 recipe). Pools out-of-fold predictions so
every patient is evaluated once, then reports AUC + all the Stage 2 figures,
with FULL class names on the confusion matrix.

    python train_ct_cv.py --seed 0
    python train_ct_cv.py --seed 0 --folds 5 --epochs 40

Outputs (per seed) under figures/petct/ct_pet_s<seed>/:
    fold_predictions.csv      out-of-fold prob + true label, every patient's crops
    metrics.json             pooled AUC, accuracy, macro-F1, per-class P/R/F1
    cm_crop.png  cm_patient.png
    precision_crop.png recall_crop.png f1_crop.png  (+ _patient)
    pr_curve_crop.png
"""
import os, json, argparse, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, accuracy_score, f1_score,
                             precision_recall_curve, average_precision_score)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models

from petct_cv_dataset import PETCTCohortDataset, CLASS_NAMES, SUBTYPE_TO_LABEL
from stage2_dataset import MANIFEST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 2)     # 2 classes: A, G
    return m.to(DEVICE)


def pet_cohort_patients():
    """The locked 83 aligned-and-annotated A/G patients. Both the CT-only and
    CT+PET arms read this SAME file, so the comparison is on identical patients."""
    c = pd.read_csv("pet_cohort_83.csv")
    return c[["patient", "subtype"]].reset_index(drop=True)


def train_one_fold(tr_pat, va_pat, seed, epochs, patience, fixed_epochs=False):
    """Nested selection: the outer fold (va_pat) is NEVER used for training or
    checkpoint selection. An inner validation split is carved out of the
    training patients for early stopping, so the reported out-of-fold
    predictions are free of selection bias."""
    set_seed(seed)

    # inner split: hold out ~15% of TRAINING patients, stratified by subtype
    coh = pd.read_csv("pet_cohort_83.csv").set_index("patient").subtype
    tr_pat = list(tr_pat)
    strat = np.array([SUBTYPE_TO_LABEL[coh[p]] for p in tr_pat])
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    itr_i, iva_i = next(inner.split(np.zeros(len(tr_pat)), strat))
    inner_tr = [tr_pat[i] for i in itr_i]
    inner_va = [tr_pat[i] for i in iva_i]

    tr = PETCTCohortDataset(inner_tr, is_train=True)
    iva = PETCTCohortDataset(inner_va, is_train=False)
    va = PETCTCohortDataset(va_pat, is_train=False)          # outer fold, untouched

    cc = tr.class_counts; n = sum(cc.values())
    w = torch.tensor([n / (2 * cc.get(c, 1)) for c in (0, 1)], dtype=torch.float32, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(tr, batch_size=64, shuffle=True, num_workers=4, generator=g)
    dl_iva = DataLoader(iva, batch_size=64, shuffle=False, num_workers=4)

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    scaler = torch.cuda.amp.GradScaler()

    # fixed_epochs mode: train a set number of epochs and keep the final model.
    # No selection at all, so no selection bias of any kind. Used as a check
    # that the nested-selection result is not an artefact of a noisy inner set.
    best_f1, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if fixed_epochs:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            continue
        # selection on the INNER validation set, never the outer fold
        ys, ps = _infer(model, dl_iva)
        vf1 = f1_score(ys, ps.argmax(1), average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, wait = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, va


def _infer(model, dl):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.cuda.amp.autocast():
                p = torch.softmax(model(x.to(DEVICE)), 1)
            ys += y.tolist(); ps.append(p.cpu().numpy())
    return np.array(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--fixed-epochs", action="store_true",
                    help="train a fixed number of epochs, no checkpoint selection")
    args = ap.parse_args()

    tag = "fixed" if args.fixed_epochs else "nested"
    OUT = f"figures/petct/ct_pet_{tag}_s{args.seed}"
    os.makedirs(OUT, exist_ok=True)

    pat = pet_cohort_patients()
    print(f"cohort: {len(pat)} A/G patients "
          f"(A={sum(pat.subtype=='A')}, G={sum(pat.subtype=='G')})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y_strat = pat.subtype.map(SUBTYPE_TO_LABEL).values

    rows = []   # pooled out-of-fold, one row per (patient, crop)
    for k, (tr_i, va_i) in enumerate(skf.split(pat.patient, y_strat)):
        tr_pat = pat.patient.iloc[tr_i].tolist()
        va_pat = pat.patient.iloc[va_i].tolist()
        model, va_ds = train_one_fold(tr_pat, va_pat, args.seed, args.epochs,
                                      args.patience, args.fixed_epochs)
        dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=4)
        ys, ps = _infer(model, dl)
        for pt, yt, pr in zip(va_ds.df.patient.values, ys, ps):
            rows.append({"patient": pt, "y": int(yt), "p0": float(pr[0]), "p1": float(pr[1])})
        print(f"  fold {k+1}/{args.folds}: {len(va_pat)} val patients")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fold_predictions.csv"), index=False)
    _report(df, OUT)


# ---------- reporting: pooled crop-level and patient-level, full class names ----------
def _report(df, OUT):
    prob = df[["p0", "p1"]].values
    yc   = df.y.values
    pc   = prob.argmax(1)

    # patient level: mean prob over a patient's crops
    g = df.groupby("patient").agg(y=("y", "first"), p0=("p0", "mean"), p1=("p1", "mean"))
    yp = g.y.values
    pp = g[["p0", "p1"]].values.argmax(1)
    probp = g[["p0", "p1"]].values

    metrics = {}
    for tag, (yt, pd_, prob2) in {
            "crop":    (yc, pc, prob),
            "patient": (yp, pp, probp)}.items():
        auc = roc_auc_score(yt, prob2[:, 1]) if len(set(yt)) > 1 else float("nan")
        pr, rc, f1, _ = precision_recall_fscore_support(yt, pd_, labels=[0, 1], zero_division=0)
        metrics[tag] = {"n": int(len(yt)), "auc": float(auc),
                        "accuracy": float(accuracy_score(yt, pd_)),
                        "macro_f1": float(f1_score(yt, pd_, average="macro", zero_division=0)),
                        "per_class": {CLASS_NAMES[c]: {"precision": float(pr[c]),
                                     "recall": float(rc[c]), "f1": float(f1[c])} for c in (0, 1)}}
        _confmat(yt, pd_, f"CT+PET {tag}-level", os.path.join(OUT, f"cm_{tag}.png"))
        for idx, mname in [(0, "precision"), (1, "recall"), (2, "f1")]:
            _bars([pr, rc, f1][idx], mname, f"CT+PET {tag}: {mname} by class",
                  os.path.join(OUT, f"{mname}_{tag}.png"))
    _pr_curve(yc, prob[:, 1], os.path.join(OUT, "pr_curve_crop.png"))

    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("\n=== pooled out-of-fold results ===")
    for tag in ("crop", "patient"):
        m = metrics[tag]
        print(f"  {tag:<8} n={m['n']:<5} AUC={m['auc']:.3f}  acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
    print(f"\nwrote {OUT}/  (metrics.json + figures)")


def _confmat(y, p, title, path):
    cm = confusion_matrix(y, p, labels=[0, 1])
    rn = cm.sum(1, keepdims=True)
    pct = np.divide(cm * 100.0, rn, out=np.zeros_like(cm, float), where=rn > 0)
    names = [CLASS_NAMES[0], CLASS_NAMES[1]]              # FULL names, not A/G
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticklabels([f"{names[i]}\n(n={int(rn[i, 0])})" for i in (0, 1)])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\nrow-normalised (diagonal = recall)", fontsize=11)
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, f"{pct[i,j]:.1f}%\n({cm[i,j]})", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=10)
    fig.colorbar(im, fraction=0.046, label="% of true class")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _bars(vals, mname, title, path):
    names = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar(names, vals, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel(mname); ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _pr_curve(y, score, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for cls, col in [(1, "#C44E52")]:                    # G as positive (of interest)
        yy = (y == cls).astype(int)
        prec, rec, _ = precision_recall_curve(yy, score)
        ap = average_precision_score(yy, score)
        ax.plot(rec, prec, color=col, lw=2,
                label=f"{CLASS_NAMES[cls]} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("CT+PET crop-level: one-vs-rest PR", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    main()
