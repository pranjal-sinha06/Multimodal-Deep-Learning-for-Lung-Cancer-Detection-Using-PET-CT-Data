# Multimodal-Deep-Learning-for-Lung-Cancer-Detection-Using-PET-CT-Data

Two-stage pipeline: a Faster R-CNN detector finds tumours on CT slices, a ResNet-50
classifies each tumour into a histological subtype. Three PET fusion arms test whether
PET improves subtype classification.

MSc dissertation , University of St Andrews.

---

## 1. Get the data

Download **Lung-PET-CT-Dx** from The Cancer Imaging Archive:

https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/

Use the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
with `Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia` from this repo. Download **both** the
images and the `Annotation` folder.

About 130 GB. You also need about 35 GB for the HU cache and 20 GB for the YOLO images.

---

## 2. Install

```bash
conda create -n lungtrain python=3.12 -y
conda activate lungtrain
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install torchmetrics==1.9.0 ultralytics==8.4.89 pycocotools \
            pydicom numpy pandas scikit-learn matplotlib pillow jupyter
```

A CUDA GPU is required for training.

---

## 3. Configure

**`stage1_dataset.py`**, line 7. Set this to your working directory. Every other module
imports it.

```python
ROOT = "/sharedscratch/ps306/lung"
```

**Every `.slurm` file.** Change these four lines for your cluster:

```bash
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
/home/ps306/.conda/envs/lungtrain/bin/python <script>
```

Use the absolute path to Python. `conda activate` does not work in SLURM batch scripts.

If you are not on SLURM, run the Python file named inside each `.slurm` directly.

---

## 4. Preprocess

Open `preprocessing_pipeline.ipynb`. In the config cell set:

```python
DX_ROOT = r"...\manifest-1608669183333"    # your download
OUT_DIR = r"...\preprocessed"              # where output goes
```

Run all cells top to bottom. Takes about 20 hours.

Produces:

| Output | What it is |
|---|---|
| `stage1_detection_manifest.csv` | 70,382 slices with split and boxes |
| `stage2_crops_manifest.csv` | 11,407 tumour crops |
| `norm_stats.json` | Windows and normalisation constants |
| `hu_cache/` | One `.npy` per slice, about 35 GB |

Copy all four to your `ROOT` directory.

---

## 5. Check it worked

```bash
python sanity.py                  # expect "missing cache files: 0"
python probe_norm.py
python verify_stage2_crops.py     # all boxes must print True
sbatch smoke.slurm                # short end-to-end test
sbatch overfit_test.slurm         # loss must fall from step 0 to 50
```


---

## 6. Stage 1: detection

```bash
sbatch train_stage1.slurm --tag run1 --epochs 100 --patience 10
```

Output goes to `runs/stage1_run1/`. Add `--resume` to continue an interrupted run.

Repeat with `--tag run2`, `run3` and so on for further experiments.

Plots and diagnostics:

```bash
python plot_stage1.py --run runs/stage1_run1
sbatch diag.slurm            # train vs validation gap, one run
sbatch diag_all_a30.slurm    # all runs
python plot_diag.py
```

---

## 7. YOLOv8 comparison

```bash
python to_yolo.py          # converts manifests to YOLO format, about 2 hours
python verify_yolo.py      # check the converted labels look right
sbatch yolo_smoke.slurm
sbatch yolo_train.slurm     # pretrained
sbatch yolo_scratch.slurm   # from scratch
```

Output goes to `runs/detect/yolo_runs/`.

---

## 8. Stage 2: subtype classification

Baseline:

```bash
sbatch train_stage2.slurm
sbatch eval_stage2.slurm
```

Output goes to `stage2_runs/run1/` and `figures/stage2/run1/`.

Ablations, run each pair in order:

| Run | Train | Evaluate |
|---|---|---|
| run2 | `sbatch train_stage2_run2.slurm` | `sbatch eval_stage2_run2_a30.slurm` |
| run3 | `sbatch train_stage2_run3.slurm` | `sbatch eval_stage2_run3.slurm` |
| run4 | `sbatch train_stage2_run4_a30.slurm` | `sbatch eval_stage2_run4_a30.slurm` |
| run5 | `sbatch train_stage2_run5_a30.slurm` | `sbatch eval_stage2_run5_a30.slurm` |

---

## 9. Five-seed sweep

```bash
sbatch verify_repro.slurm     # must pass before continuing
sbatch sweep_train.slurm      # 25 jobs, about 3 hours each
sbatch sweep_eval.slurm       # 25 jobs, about 40 minutes each
python collect_seeds.py
python aggregate_figures.py
```

Output goes to `stage2_runs/<run>_s<seed>/` and `figures/stage2/agg_run<N>/`.

---

## 10. End-to-end evaluation

Runs the detector and classifier chained together.

```bash
sbatch eval_pipeline_a30.slurm                    # YOLO into Stage 2
sbatch eval_pipeline_frcnn_a30.slurm              # Faster R-CNN into Stage 2
sbatch eval_pipeline_perpatient_a30.slurm
sbatch eval_pipeline_frcnn_perpatient_a30.slurm
sbatch eval_framing_diagnostic_a30.slurm
```

---

## 11. PET and Secondary Capture fusion

Prepare the caches and manifests first:

```bash
sbatch run_precompute.slurm         # runs precompute_suv.py and precompute_sc.py
python verify_suv_cache.py
python build_sc_manifest.py
python fix_sc_manifest.py
python build_matched_sc_manifest.py
```

Then run the four arms, five seeds each:

```bash
for s in 0 1 2 3 4; do
  sbatch run_ct_cv.slurm         --seed $s
  sbatch run_petct_cv.slurm      --seed $s
  sbatch run_sc_cv.slurm         --seed $s
  sbatch run_sc_matched_cv.slurm --seed $s
done
```

| Arm | Script | Input |
|---|---|---|
| CT only | `train_ct_cv.py` | Three CT windows |
| CT + PET | `train_petct_cv.py` | Two CT windows plus SUV |
| Secondary Capture | `train_sc_cv.py` | Fused PET/CT rendering |
| SC density matched | `train_sc_matched_cv.py` | Same, with box counts matched to CT |

Output goes to `figures/petct/<arm>_nested_s<seed>/`.

Optional flags: `--folds 5`, `--epochs 40`, `--patience 10`, and `--fixed-epochs` to skip
checkpoint selection.

Compare the arms:

```bash
python compare_arms.py
python plot_metrics.py
python plot_roc.py
```

---

## 12. Confound check

```bash
python scanner_probe.py
python scanner_probe.py --cohort pet_cohort_83.csv
```

---

## 13. Figures

```bash
python plot_stage1_figures.py       # detection comparison
sbatch detection_examples.slurm     # example detections
python aggregate_figures.py         # Stage 2
python plot_metrics.py              # fusion arms
python plot_roc.py                  # ROC curves
```

---

## File reference

| File | Purpose |
|---|---|
| `preprocessing_pipeline.ipynb` | Builds manifests, normalisation stats, HU cache |
| `stage1_dataset.py` | Sets `ROOT`, `CACHE`, and the CT windowing. Imported everywhere. |
| `stage2_dataset.py` | Tumour crops |
| `stage2_dataset_run3.py` | Crops with run3 augmentation |
| `crop_utils.py` | Crop geometry and SUV sampling |
| `train_stage1.py` | Faster R-CNN training |
| `to_yolo.py`, `lung.yaml` | YOLO conversion and dataset file |
| `train_stage2*.py` | Classifier, baseline plus four ablations |
| `eval_stage2*.py` | Classifier evaluation |
| `eval_pipeline*.py` | Chained detector plus classifier |
| `precompute_suv.py` | PET to SUV, resampled to CT geometry |
| `precompute_sc.py` | Extracts Secondary Capture slices |
| `*_cv_dataset.py` | Input for each fusion arm |
| `train_*_cv.py` | Fusion arm training |
| `compare_arms.py` | Compares the four arms |
| `scanner_probe.py` | Acquisition confound test |
| `sanity.py`, `probe_norm.py`, `verify_*.py` | Pre-run checks |
| `plot_*.py`, `aggregate_figures.py` | Figures |

---

## Other notebooks

| File | Purpose |
|---|---|
| `eda_lung_pet_ct_dx.ipynb` | Dataset exploration |

---

