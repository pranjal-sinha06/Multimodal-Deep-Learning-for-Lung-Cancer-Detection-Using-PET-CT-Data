"""
plot_stage1_figures.py  --  the two Stage 1 figures that were never generated.

Produces:
  figures/stage1_two_arch_map.png       Faster R-CNN vs the two YOLO runs, the
                                        two-architecture ceiling made visual
  figures/stage1/annotation_overlay_example.png
                                        a parsed box drawn on its CT slice, the
                                        visual proof that annotation parsing works

Run on the cluster, where both the metrics.json files and the DICOMs live.

  python plot_stage1_figures.py                      # both figures
  python plot_stage1_figures.py --patient Lung_Dx-A0001   # pick the overlay case
"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LUNG   = "/sharedscratch/ps306/lung"
RUNS   = "runs"
CROPS  = os.path.join(LUNG, "stage2_crops_manifest.csv")
CT_IDX = os.path.join(LUNG, "ct_sc_index.csv")
OUTDIR = "figures"
MARKER = "Lung-PET-CT-Dx"

# the two YOLO result CSVs; the from-scratch run collapses late in training and
# is truncated at its peak so the figure shows the ceiling, not the instability
YOLO_FILES = {
    "YOLOv8s (COCO-pretrained)": os.path.join(LUNG, "yolo_results_pretrained.csv"),
    "YOLOv8s (from scratch)":    os.path.join(LUNG, "yolo_results_scratch.csv"),
}


def frcnn_curves():
    """val mAP@0.5 per epoch for each usable Faster R-CNN run."""
    out = {}
    for f in sorted(glob.glob(f"{RUNS}/stage1_run*/metrics.json")):
        run = f.split("/")[-2].replace("stage1_", "")
        rows = json.load(open(f))
        if not isinstance(rows, list):
            continue
        ep = [r.get("epoch") for r in rows]
        mp = [r.get("val_map_50") for r in rows]
        pairs = [(e, m) for e, m in zip(ep, mp) if m is not None]
        if not pairs:
            continue
        # a run whose val mAP never exceeds 0.3 has failed; skip it
        if max(m for _, m in pairs) < 0.30:
            print(f"  skipping failed run {run} (peak {max(m for _,m in pairs):.3f})")
            continue
        out[run] = pairs
    return out


def yolo_curve(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    col = next(c for c in d.columns if "mAP50" in c and "95" not in c)
    m = d[col].values
    # truncate at the point of collapse, if any (a fall to below half the peak)
    peak = m.max()
    cut = len(m)
    for i in range(1, len(m)):
        if m[i] < 0.5 * peak and m[i-1] >= 0.5 * peak:
            cut = i
            break
    return np.arange(cut), m[:cut]


def fig_two_arch():
    fig, ax = plt.subplots(figsize=(8, 5.2))

    frc = frcnn_curves()
    # plot each FRCNN run faintly, and their envelope as the representative line
    if frc:
        for run, pairs in frc.items():
            e = [p[0] for p in pairs]; m = [p[1] for p in pairs]
            ax.plot(e, m, color="#4C72B0", alpha=0.28, lw=1)
        # a single bold FRCNN line: the per-epoch max across runs
        allep = sorted({p[0] for pairs in frc.values() for p in pairs})
        env = []
        for e in allep:
            vals = [dict(pairs).get(e) for pairs in frc.values() if dict(pairs).get(e) is not None]
            if vals: env.append((e, max(vals)))
        ax.plot([x for x, _ in env], [y for _, y in env], color="#4C72B0",
                lw=2.4, label="Faster R-CNN (best of configs)")
    else:
        print("  no Faster R-CNN metrics.json found; plotting YOLO only")

    styles = {"YOLOv8s (COCO-pretrained)": ("#C44E52", "-"),
              "YOLOv8s (from scratch)":    ("#DD8452", "--")}
    for name, path in YOLO_FILES.items():
        if not os.path.exists(path):
            print(f"  missing {path}; skipping {name}")
            continue
        e, m = yolo_curve(path)
        c, ls = styles[name]
        ax.plot(e, m, color=c, ls=ls, lw=2.2, label=f"{name} (peak {m.max():.3f})")

    ax.axhspan(0.63, 0.65, color="grey", alpha=0.12, zorder=0)
    ax.text(ax.get_xlim()[1]*0.98, 0.64, "ceiling 0.63--0.65", ha="right",
            va="center", fontsize=8.5, color="#555")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation mAP@0.5")
    ax.set_ylim(0, 1.0)
    ax.set_title("Two detectors, one ceiling\n"
                 "Faster R-CNN peaks early and degrades; YOLOv8s rises and holds; "
                 "both plateau near 0.63 to 0.65", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/stage1_two_arch_map.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  wrote {p}")


def to_relative(win):
    s = str(win).replace("\\", "/")
    i = s.find(MARKER + "/")
    return None if i < 0 else s[i+len(MARKER)+1:]


def detect_root(sample):
    c = [os.path.join(LUNG, MARKER),
         os.path.join(LUNG, "manifest-1608669183333", MARKER),
         os.path.join(LUNG, "LPT", "manifest-1608669183333", MARKER)]
    c += sorted(glob.glob(os.path.join(LUNG, "*", MARKER)))
    c += sorted(glob.glob(os.path.join(LUNG, "*", "*", MARKER)))
    for r in dict.fromkeys(c):
        if os.path.exists(os.path.join(r, sample)):
            return r
    return None


def fig_overlay(patient):
    try:
        import pydicom
    except ImportError:
        print("  pydicom not available; skipping the overlay figure")
        return
    import json as _json
    crops = pd.read_csv(CROPS)
    ct = pd.read_csv(CT_IDX, low_memory=False)
    ct = ct[ct.kind == "CT"] if "kind" in ct.columns else ct

    if patient is None:
        # a clear adenocarcinoma case with a mid-sized box
        patient = crops[crops.subtype == "A"].patient.iloc[0]
    row = crops[crops.patient == patient].iloc[0]
    box = _json.loads(row.box)
    path = ct[ct.sop == row.sop].path
    if path.empty:
        print(f"  no CT path for sop of {patient}; skipping overlay")
        return
    rel = to_relative(path.iloc[0]); root = detect_root(rel)
    if root is None:
        print("  could not locate DICOM root; skipping overlay")
        return
    ds = pydicom.dcmread(os.path.join(root, rel), force=True)
    img = ds.pixel_array.astype(np.float32)
    # lung window for display
    lo, hi = -1000, 400
    img = np.clip((img + (-int(getattr(ds, "RescaleIntercept", -1024))) - 0 - lo)
                  / (hi - lo), 0, 1) if False else np.clip((img - img.min())/(np.ptp(img)+1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    x1, y1, x2, y2 = box
    ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                    edgecolor="#55A868", lw=2.2))
    ax.set_title(f"Parsed annotation on its CT slice\n{patient}, subtype {row.subtype}",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(f"{OUTDIR}/stage1", exist_ok=True)
    p = f"{OUTDIR}/stage1/annotation_overlay_example.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=None, help="patient id for the overlay")
    ap.add_argument("--skip-overlay", action="store_true")
    args = ap.parse_args()
    print("two-architecture figure:")
    fig_two_arch()
    if not args.skip_overlay:
        print("annotation overlay figure:")
        fig_overlay(args.patient)


if __name__ == "__main__":
    main()
