"""
build_sc_manifest.py  --  turn the Secondary Capture annotations into a manifest.

CONTEXT
  check_annotations.py established that of 31,562 annotation files, 20,484
  reference CT objects and 10,400 reference Secondary Capture (fused PET/CT)
  objects. The existing pipeline uses only CT annotations, so the SC ones have
  never been touched. This script extracts them into the same shape as
  stage2_crops_manifest.csv so they can be used directly.

WHAT IT HANDLES
  - Label case is inconsistent in the source ('A' 13065 vs 'a' 7911, and the
    same for G and B). Labels are upper-cased before use, otherwise the class
    set silently doubles.
  - A 'Q' label appears 10 times and is not a valid subtype. Such rows are
    dropped and counted, never remapped.
  - Patient identity is not derivable from the XML path, so it is joined
    through the SOP UID via ct_sc_index.csv.
  - Annotations resolving to neither CT nor SC are checked against pet_index,
    to establish whether PET objects are annotated too.

OUTPUT
  sc_crops_manifest.csv   patient, split, sop, path, box, subtype, label_idx
                          one row per box, matching the CT manifest's columns
  sc_annotation_audit.csv per-file record, including dropped rows and reasons

USAGE
  python build_sc_manifest.py
  python build_sc_manifest.py --ann-dir /path/to/Annotation
"""
import os, sys, glob, json, argparse, re
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd

LUNG = "/sharedscratch/ps306/lung"
DEFAULTS = {
    "ann_dir":  os.path.join(LUNG, "LPT/manifest-1608669183333/Annotation"),
    "ct_index": os.path.join(LUNG, "ct_sc_index.csv"),
    "pet_index": os.path.join(LUNG, "pet_index.csv"),
    "flags":    os.path.join(LUNG, "pet_cohort_flags.csv"),
    "out":      os.path.join(LUNG, "sc_crops_manifest.csv"),
    "audit":    os.path.join(LUNG, "sc_annotation_audit.csv"),
}

UID_RE = re.compile(r"\d+(?:\.\d+){4,}")
VALID_SUBTYPES = {"A", "B", "E", "G"}          # per the TCIA naming scheme
LABEL_IDX = {"A": 0, "B": 1, "G": 2}           # matches the existing CT manifest


def parse(path):
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return None, f"{type(e).__name__}"
    size = root.find("size")
    depth = (size.findtext("depth") or "").strip() if size is not None else ""
    boxes = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        boxes.append({"label_raw": (o.findtext("name") or "").strip(), "box": box})
    uids = UID_RE.findall(os.path.basename(path))
    uids += UID_RE.findall(root.findtext("filename") or "")
    uids += UID_RE.findall(root.findtext("path") or "")
    return {"uids": uids, "boxes": boxes, "depth": depth}, None


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_','-')}", default=v)
    args = ap.parse_args()

    for k in ("ann_dir", "ct_index"):
        if not os.path.exists(getattr(args, k)):
            sys.exit(f"missing {k}: {getattr(args, k)}")

    ct = pd.read_csv(args.ct_index, low_memory=False)
    ct["sop"] = ct.sop.astype(str)
    sc_map = ct[ct.kind == "SC"].drop_duplicates("sop").set_index("sop")
    ct_sops = set(ct.loc[ct.kind == "CT", "sop"])
    pet_sops = set()
    if os.path.exists(args.pet_index):
        pet_sops = set(pd.read_csv(args.pet_index, low_memory=False).sop.astype(str))

    xmls = glob.glob(os.path.join(args.ann_dir, "**", "*.xml"), recursive=True)
    print(f"{len(xmls)} annotation files\n")

    rows, audit = [], []
    tally = Counter()
    bad_labels = Counter()

    for i, f in enumerate(xmls, 1):
        rec, err = parse(f)
        if rec is None:
            tally["parse_error"] += 1
            audit.append({"file": os.path.basename(f), "outcome": "parse_error",
                          "detail": err})
            continue

        sop = next((u for u in rec["uids"] if u in sc_map.index), None)
        if sop is None:
            if any(u in ct_sops for u in rec["uids"]):
                tally["CT (skipped, already used)"] += 1
                outcome = "ct"
            elif any(u in pet_sops for u in rec["uids"]):
                tally["PET"] += 1
                outcome = "pet"
            else:
                tally["unresolved"] += 1
                outcome = "unresolved"
            audit.append({"file": os.path.basename(f), "outcome": outcome, "detail": ""})
            continue

        meta = sc_map.loc[sop]
        kept = 0
        for b in rec["boxes"]:
            lab = b["label_raw"].upper()          # 'a' and 'A' are the same class
            if lab not in VALID_SUBTYPES:
                bad_labels[b["label_raw"]] += 1
                continue
            x1, y1, x2, y2 = b["box"]
            if x2 <= x1 or y2 <= y1:
                tally["degenerate_box"] += 1
                continue
            rows.append({"patient": meta.patient, "sop": sop, "path": meta.path,
                         "box": json.dumps([x1, y1, x2, y2]), "subtype": lab})
            kept += 1
        tally["SC (kept)"] += 1
        audit.append({"file": os.path.basename(f), "outcome": "sc",
                      "detail": f"{kept} box(es), depth={rec['depth']}"})
        if i % 5000 == 0:
            print(f"  parsed {i}/{len(xmls)}")

    print("\n=== annotation files by outcome ===")
    for k, n in tally.most_common():
        print(f"  {k:<28} {n}")
    if bad_labels:
        print("\n  invalid labels dropped (not remapped):")
        for k, n in bad_labels.most_common():
            print(f"    {k!r:<10} {n}")

    if not rows:
        sys.exit("\nNo SC boxes extracted. Check --ann-dir and --ct-index.")

    df = pd.DataFrame(rows)

    # subtype from the patient id is the authority; the XML label is a check
    df["subtype_from_id"] = df.patient.str.extract(r"Lung_Dx-([A-Za-z])")[0].str.upper()
    mism = int((df.subtype != df.subtype_from_id).sum())
    print(f"\n  boxes whose XML label disagrees with the patient id: {mism} "
          f"of {len(df)} ({100*mism/len(df):.1f}%)")
    if mism:
        print("    using the patient id, since histopathology is a patient-level fact")
    df["subtype"] = df.subtype_from_id
    df = df.drop(columns=["subtype_from_id"])

    # attach the split used everywhere else, so comparisons stay aligned
    if os.path.exists(args.flags):
        fl = pd.read_csv(args.flags)[["patient", "split"]]
        df = df.merge(fl, on="patient", how="left")
        missing = int(df.split.isna().sum())
        if missing:
            print(f"  {missing} boxes from patients outside the 348 cohort "
                  f"(no split assigned)")
    else:
        df["split"] = ""

    df = df[df.subtype.isin(LABEL_IDX)].copy()
    df["label_idx"] = df.subtype.map(LABEL_IDX)
    df = df[["patient", "split", "sop", "path", "box", "subtype", "label_idx"]]
    df.to_csv(args.out, index=False)
    pd.DataFrame(audit).to_csv(args.audit, index=False)

    print(f"\nwrote {args.out}")
    print(f"wrote {args.audit}")

    print("\n=== the recovered Secondary Capture cohort ===")
    print(f"  boxes    : {len(df)}")
    print(f"  slices   : {df.sop.nunique()}")
    print(f"  patients : {df.patient.nunique()}")
    print("\n  by subtype (patients):")
    print(df.drop_duplicates('patient').subtype.value_counts().to_string())
    if df.split.notna().any():
        print("\n  by split (patients):")
        print(df.drop_duplicates('patient').split.value_counts(dropna=False).to_string())
    ag = df[df.subtype.isin(["A", "G"])].drop_duplicates("patient")
    print(f"\n  A/G patients: {len(ag)}  "
          f"(A={sum(ag.subtype=='A')}, G={sum(ag.subtype=='G')})")
    if "split" in ag:
        print(f"  of which in test split: {int((ag.split=='test').sum())}")
    print("\n  boxes per patient: median "
          f"{df.groupby('patient').size().median():.0f}")


if __name__ == "__main__":
    main()
