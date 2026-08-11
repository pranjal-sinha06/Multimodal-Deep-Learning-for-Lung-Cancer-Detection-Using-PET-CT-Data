"""
check_annotations.py  --  do the XML annotations reference CT or Secondary Capture?

WHY THIS EXISTS
  The draft states the Secondary Capture (fused PET/CT) objects carry 10,227
  annotated slices. A join of the crops manifest against the DICOM index says
  otherwise: all 11,297 annotated SOPs are CT objects and none is SC. Those two
  claims cannot both be true, and the difference decides whether an SC model can
  do detection at all.

  The crops manifest is downstream of a pipeline that may have kept only CT
  rows, so it cannot settle the question. Only the raw XML files can.

WHAT IT DOES
  Walks the Annotation folder, parses every PASCAL VOC XML, works out which
  DICOM object each one refers to, and reports how many resolve to CT, how many
  to SC, and how many to neither. Read only; nothing is modified.

USAGE
  python check_annotations.py
  python check_annotations.py --ann-dir /path/to/Annotation
  python check_annotations.py --ct-index /path/to/ct_sc_index.csv

WHAT THE ANSWER MEANS
  all CT, none SC   -> SC carries no boxes. Detection on SC is impossible and
                       the draft's 10,227 claim must be removed.
  some SC           -> SC supports detection. The route reopens on ~123 A/G
                       patients and is worth serious consideration.
  many unresolved   -> the matching strategy is wrong, not the conclusion.
                       The script prints samples so it can be fixed.
"""
import os, sys, glob, argparse, re
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd


CANDIDATE_ANN_DIRS = [
    "/sharedscratch/ps306/lung/LPT/manifest-1608669183333/Annotation",
    "/sharedscratch/ps306/lung/manifest-1608669183333/Annotation",
    "/sharedscratch/ps306/lung/Annotation",
    r"E:\Dissertation - Code\LPT\manifest-1608669183333\Annotation",
]
CANDIDATE_CT_INDEX = [
    "/sharedscratch/ps306/lung/ct_sc_index.csv",
    "ct_sc_index.csv",
    r"E:\Dissertation - Code\LPT\manifest-1608669183333\ct_sc_index.csv",
]

UID_RE = re.compile(r"\d+(?:\.\d+){4,}")     # a DICOM UID looks like 1.3.6.1.4...


def find_first(paths, what):
    for p in paths:
        if os.path.exists(p):
            return p
    sys.exit(f"Could not locate {what}. Tried:\n  " + "\n  ".join(paths) +
             f"\nPass it explicitly with the matching --flag.")


def parse_xml(path):
    """Pull out everything that might identify the referenced image."""
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    get = lambda tag: (root.findtext(tag) or "").strip()
    size = root.find("size")
    w = h = d = ""
    if size is not None:
        w = (size.findtext("width") or "").strip()
        h = (size.findtext("height") or "").strip()
        d = (size.findtext("depth") or "").strip()
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        box = None
        if bb is not None:
            try:
                box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError):
                box = None
        objs.append({"name": (o.findtext("name") or "").strip(), "box": box})
    return {"folder": get("folder"), "filename": get("filename"), "path": get("path"),
            "width": w, "height": h, "depth": d, "objects": objs}


def candidate_uids(rec, xml_path):
    """Every UID-looking string that could identify the referenced DICOM.

    Different releases put the SOP UID in different places, so all of them are
    tried rather than assuming one convention.
    """
    out = []
    for s in (rec.get("filename", ""), rec.get("path", ""),
              os.path.basename(xml_path), rec.get("folder", "")):
        if s:
            out.extend(UID_RE.findall(str(s)))
    # also the XML's own stem, with any extension stripped
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    if UID_RE.fullmatch(stem):
        out.append(stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--ct-index", default=None)
    ap.add_argument("--limit", type=int, default=0, help="parse only N files (trial)")
    args = ap.parse_args()

    ann_dir = args.ann_dir or find_first(CANDIDATE_ANN_DIRS, "the Annotation folder")
    ct_path = args.ct_index or find_first(CANDIDATE_CT_INDEX, "ct_sc_index.csv")
    print(f"Annotation folder : {ann_dir}")
    print(f"DICOM index       : {ct_path}\n")

    xmls = glob.glob(os.path.join(ann_dir, "**", "*.xml"), recursive=True)
    if not xmls:
        # some releases ship the files without an extension
        xmls = [f for f in glob.glob(os.path.join(ann_dir, "**", "*"), recursive=True)
                if os.path.isfile(f)]
        print(f"(no *.xml found; falling back to all files: {len(xmls)})")
    if args.limit:
        xmls = xmls[:args.limit]
    print(f"annotation files found: {len(xmls)}\n")
    if not xmls:
        sys.exit("Nothing to parse. Check --ann-dir.")

    ct = pd.read_csv(ct_path, low_memory=False)
    ct_sops = set(ct.loc[ct.kind == "CT", "sop"].astype(str)) if "kind" in ct.columns \
              else set(ct.sop.astype(str))
    sc_sops = set(ct.loc[ct.kind == "SC", "sop"].astype(str)) if "kind" in ct.columns \
              else set()
    print(f"index: {len(ct_sops)} CT objects, {len(sc_sops)} SC objects\n")

    verdict = Counter()
    depths = Counter()
    sizes = Counter()
    labels = Counter()
    n_boxes = 0
    unresolved_samples = []
    sc_hits = []
    patients = set()

    for i, f in enumerate(xmls, 1):
        rec = parse_xml(f)
        if "error" in rec:
            verdict["parse_error"] += 1
            if len(unresolved_samples) < 5:
                unresolved_samples.append((f, rec["error"]))
            continue

        depths[rec["depth"] or "(absent)"] += 1
        sizes[f"{rec['width']}x{rec['height']}"] += 1
        for o in rec["objects"]:
            if o["name"]:
                labels[o["name"]] += 1
            if o["box"]:
                n_boxes += 1

        # patient id from the path, if present
        m = re.search(r"(Lung_Dx-[A-Za-z]\d+)", f.replace("\\", "/"))
        if m:
            patients.add(m.group(1))

        uids = candidate_uids(rec, f)
        hit_ct = any(u in ct_sops for u in uids)
        hit_sc = any(u in sc_sops for u in uids)
        if hit_ct and hit_sc:
            verdict["BOTH (ambiguous)"] += 1
        elif hit_ct:
            verdict["CT"] += 1
        elif hit_sc:
            verdict["SC"] += 1
            if len(sc_hits) < 10:
                sc_hits.append((f, rec["filename"], uids[:2]))
        else:
            verdict["unresolved"] += 1
            if len(unresolved_samples) < 5:
                unresolved_samples.append(
                    (f, f"filename={rec['filename']!r} uids_found={uids[:2]}"))
        if i % 5000 == 0:
            print(f"  parsed {i}/{len(xmls)}")

    # ---------- report ----------
    print("\n" + "=" * 62)
    print("WHICH DICOM OBJECT DOES EACH ANNOTATION REFERENCE?")
    print("=" * 62)
    total = sum(verdict.values())
    for k, n in verdict.most_common():
        print(f"  {k:<20} {n:>7}  ({100*n/total:.1f}%)")

    print(f"\n  total boxes across all files : {n_boxes}")
    print(f"  patients represented         : {len(patients)}")

    print("\n  image size recorded in the XML:")
    for k, n in sizes.most_common(5):
        print(f"    {k:<14} {n}")
    print("\n  colour depth recorded in the XML (1 = greyscale/CT, 3 = RGB/SC):")
    for k, n in depths.most_common(5):
        print(f"    depth={k:<10} {n}")
    print("\n  object labels:")
    for k, n in labels.most_common(8):
        print(f"    {k:<14} {n}")

    if sc_hits:
        print("\n  *** ANNOTATIONS RESOLVING TO SECONDARY CAPTURE ***")
        for f, fn, u in sc_hits:
            print(f"    {os.path.basename(f)}  filename={fn!r}")
    if unresolved_samples:
        print("\n  unresolved / error samples (for fixing the match, if needed):")
        for f, why in unresolved_samples:
            print(f"    {os.path.basename(f)}: {why}")

    print("\n" + "=" * 62)
    if verdict["SC"] > 0:
        print("SC IS ANNOTATED. Detection on Secondary Capture is possible.")
        print("Report the SC count above; the route reopens.")
    elif verdict["unresolved"] > 0.2 * total:
        print("INCONCLUSIVE. Too many files did not resolve to any DICOM object.")
        print("Send the unresolved samples above and the matching can be fixed.")
    else:
        print("NO SC ANNOTATIONS. Every resolved annotation is on a CT object.")
        print("Detection on Secondary Capture is not possible, and the draft's")
        print("claim of 10,227 annotated SC slices should be removed.")
    print("=" * 62)


if __name__ == "__main__":
    main()
