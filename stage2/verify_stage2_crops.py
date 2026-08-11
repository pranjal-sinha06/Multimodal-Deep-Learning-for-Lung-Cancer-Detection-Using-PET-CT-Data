import os, json, numpy as np
from PIL import Image
from stage1_dataset import window_channels, ROOT, CACHE
from stage2_dataset import load_hu_cached, crop_box, LungSubtypeDataset

ds = LungSubtypeDataset("train")
outdir = os.path.join(ROOT, "stage2_verify2"); os.makedirs(outdir, exist_ok=True)

for k in range(6):
    r = ds.df.iloc[k * 500 % len(ds.df)]
    box = json.loads(r.box)
    hu = load_hu_cached(r.sop)

    #  Verification : coordinate match
    print(f"\nsample {k}: subtype={r.subtype}  sop=...{r.sop[-8:]}")
    print(f"  manifest box (drawn on overlay): {box}")
    print(f"  box within image bounds {hu.shape}: "
          f"{0 <= box[0] < box[2] <= hu.shape[1] and 0 <= box[1] < box[3] <= hu.shape[0]}")

    #  Verification : the actual crop the model receives
    crop = crop_box(hu, box)                                   # margin crop, no jitter (eval path)
    crop_img = (window_channels(crop) * 255).astype(np.uint8).transpose(1, 2, 0)
    crop_pil = Image.fromarray(crop_img).resize((256, 256))    # what the model sees
    full = (window_channels(hu) * 255).astype(np.uint8).transpose(1, 2, 0)
    full_pil = Image.fromarray(full)
    # side-by-side: full slice (512) | crop (256), on one canvas
    canvas = Image.new("RGB", (512 + 256 + 10, 512), (0, 0, 0))
    canvas.paste(full_pil, (0, 0))
    canvas.paste(crop_pil, (522, 0))
    canvas.save(os.path.join(outdir, f"{k}_{r.subtype}_slice_and_crop.jpg"))
    print(f"  crop pixel size before resize: {crop.shape}  -> saved side-by-side view")

print(f"\nwrote 6 side-by-side (full slice | model crop) images to {outdir}")
print("Verification A: check each printed box is within bounds and x1<x2, y1<y2 (valid box).")
print("Verification B: in each image, the right panel (crop) should be centered on a distinct structure.")
