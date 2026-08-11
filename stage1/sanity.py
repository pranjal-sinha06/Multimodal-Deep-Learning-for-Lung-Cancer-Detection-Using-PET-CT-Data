import os, json, numpy as np, pandas as pd

ROOT  = "/sharedscratch/ps306/lung"
CACHE = os.path.join(ROOT, "hu_cache")

s1 = pd.read_csv(os.path.join(ROOT, "stage1_detection_manifest.csv"))
s2 = pd.read_csv(os.path.join(ROOT, "stage2_crops_manifest.csv"))
stats = json.load(open(os.path.join(ROOT, "norm_stats.json")))

print("stage1 rows:", len(s1), "| stage2 rows:", len(s2))
print("stage1 columns:", list(s1.columns))
print("split counts:", s1.split.value_counts().to_dict())
print("role counts :", s1.role.value_counts().to_dict())

# every SOP in the manifest must have a cached .npy
sample = s1.sop.sample(500, random_state=0)
missing = [s for s in sample if not os.path.exists(os.path.join(CACHE, s + ".npy"))]
print("missing cache files (of 500 sampled):", len(missing))

# a loaded slice must be 512x512 and window into [0,1]
arr = np.load(os.path.join(CACHE, s1.sop.iloc[0] + ".npy")).astype(np.float32)
print("array shape:", arr.shape, "| HU range:", round(float(arr.min())), "to", round(float(arr.max())))

lo, hi = -700 - 1400/2, -700 + 1400/2          # lung window
lung = np.clip((arr - lo) / (hi - lo), 0, 1)
print("lung-window range:", round(float(lung.min()), 3), "to", round(float(lung.max()), 3))

# boxes parse, and stats present
print("first boxes parse:", json.loads(s1.boxes.iloc[0]))
print("stats keys:", list(stats.keys()))
