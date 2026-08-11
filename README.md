# Multimodal-Deep-Learning-for-Lung-Cancer-Detection-Using-PET-CT-Data

A two-stage deep learning pipeline for lung tumour detection and histological subtype
classification from CT, extended with three PET fusion arms.

MSc Artificial Intelligence dissertation (CS5099), University of St Andrews.
Pranjal Sinha, supervised by Dr. David Harris-Birtill.

---

**Stage 1** detects tumours on axial CT slices with a class-agnostic Faster R-CNN
(ResNet-50-FPN), cross-checked against YOLOv8s.
**Stage 2** classifies each tumour crop into adenocarcinoma, small cell, or squamous
cell carcinoma with a ResNet-50.
**The fusion arms** ask whether PET adds anything to subtype classification, comparing
CT alone against SUV channel substitution and against the fused PET/CT renderings
recovered from the collection's Secondary Capture objects.

Headline results, all patient-level and averaged over five seeds:

| Task | Result |
|---|---|
| Detection, Faster R-CNN | mAP@0.5 approx. 0.65 |
| Detection, YOLOv8s pretrained | mAP@0.5 0.628 |
| Detection, YOLOv8s from scratch | mAP@0.5 0.614 |
| Subtype, three-class per-lesion macro-F1 | 0.556 +/- 0.098 |
| Subtype, CT-only patient AUC (binary, 83 patients) | 0.778 +/- 0.057 |
| Subtype, CT+PET channel swap | 0.745 +/- 0.050 |
| Subtype, Secondary Capture fused | 0.801 +/- 0.023 |
| Subtype, Secondary Capture density-matched | 0.819 +/- 0.040 |

Two architectures converging on the same detection ceiling is evidence of a data
limit rather than a tuning failure. Published figures on this collection reach
mAP@0.5 0.94, which is attributable to slice-level rather than patient-level
splitting.

---

## Data

The dataset is not in this repository. Download it from The Cancer Imaging Archive:

**Lung-PET-CT-Dx**
https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/

355 patients, 1,295 series, 251,135 DICOM objects, with radiologist bounding boxes in
PASCAL VOC format and histology confirmed by pathology. Approximately 130 GB.

`Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia` in this repository is the download manifest.
Open it with the **NBIA Data Retriever** (https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide)
and select both the images and the annotation folder.

---

## Requirements

Training needs a CUDA GPU. Everything was run on the University of St Andrews cluster
Hypatia under SLURM, on an L40S or A30, but the scripts are ordinary PyTorch and will run
anywhere with enough memory.

```bash
conda create -n lungtrain python=3.12 -y
conda activate lungtrain
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install torchmetrics==1.9.0 ultralytics==8.4.89 pycocotools \
            pydicom numpy pandas scikit-learn matplotlib pillow jupyter
```

Approximate storage: 130 GB for the raw DICOM, 35 GB for the HU cache, 20 GB for the
YOLO image set.

### SLURM

Every `.slurm` file carries the St Andrews account, QOS and partition, and invokes Python
by absolute path. Change these three lines for your own cluster:

```bash
#SBATCH --partition=gpu.L40S
#SBATCH --account=compsci
#SBATCH --qos=standrews
...
/home/ps306/.conda/envs/lungtrain/bin/python <script>
```

The absolute path is deliberate. `conda activate` fails silently in non-interactive batch
shells and runs the job against the wrong interpreter without raising an error.

---

## Step by step

Set `ROOT` in `stage1_dataset.py` to your working directory before anything else.
Every module imports it from there.

### 1. Preprocess

Open `preprocessing_pipeline.ipynb`, set `DX_ROOT` and `OUT_DIR` in the config cell, and
run all cells in order. Roughly 20 hours, reading 200,000 DICOM headers and
writing the HU cache.

It produces four artefacts, which are the entire interface to model training. No model
code reads DICOM.

| Artefact | Contents |
|---|---|
| `stage1_detection_manifest.csv` | 70,382 slices with role, split and boxes |
| `stage2_crops_manifest.csv` | 11,407 tumour crops with subtype and split |
| `norm_stats.json` | HU windows, channel order, normalisation constants |
| `hu_cache/` | One float16 `.npy` per slice, keyed by SOP UID, approx. 35 GB |

Each step ends with a `check()` assertion that prints PASS or FAIL, so an error surfaces
where it happened. The split is patient-level, stratified by subtype, 70/20/10, seeded at
0. It yields 244 training, 70 validation and 34 test patients, and reproduces exactly.

Copy the manifests, `norm_stats.json` and `hu_cache/`.

### 2. Verify before training

```bash
python sanity.py                  # expect: missing cache files: 0
python probe_norm.py              # normalisation statistics
python verify_stage2_crops.py     # writes 6 overlays; all boxes must print True
sbatch smoke.slurm                # one truncated epoch end to end
sbatch overfit_test.slurm         # 50 steps on 4 slices; loss must fall visibly
```

The overfit test uses two positive and two negative slices deliberately. An all-negative
batch never exercises the box regression loss and so proves nothing.

### 3. Stage 1: detection

```bash
sbatch train_stage1.slurm --tag run1 --epochs 100 --patience 10
```

Writes to `runs/stage1_run1/`: `metrics.csv`, `metrics.json`, `best.pth`, `last.pth`.
Repeat with `--tag run2` and so on for the single-variable experiments. Resume an
interrupted run with `--resume`.

```bash
python plot_stage1.py --run runs/stage1_run1     # curves and best-epoch summary
sbatch diag.slurm                                # train against validation gap, one run
sbatch diag_all_a30.slurm                        # the same across all five
python plot_diag.py
```

### 4. YOLOv8 cross-check

```bash
python to_yolo.py          # 49,257 train and 13,003 val images, approx. 2 hours
python verify_yolo.py      # draws converted labels back on; inspect before training
sbatch yolo_smoke.slurm
sbatch yolo_train.slurm    # COCO-pretrained
sbatch yolo_scratch.slurm  # architecture only, random init
```

`to_yolo.py` imports `window_channels` from `stage1_dataset` rather than reimplementing
it, so both architectures see pixel-identical inputs. That is what makes the ceiling
argument hold.

`train_yolo.py` is an alternative Python-API entry point that sets Adam at 1e-3. It was
**not** used for the reported results, which came from the SLURM scripts above with the
framework default optimiser and learning rate.

### 5. Stage 2: subtype classification

```bash
sbatch train_stage2.slurm            # run1, the baseline
sbatch eval_stage2.slurm             # per-crop and per-lesion metrics and figures
```

Ablations, one file each so every experiment is independently re-runnable and its diff
against the baseline directly inspectable:

| Run | Change from baseline |
|---|---|
| run2 | Adds a class-aware sampler on top of the weighted loss |
| run3 | Stronger geometric and intensity augmentation |
| run4 | Weight decay |
| run5 | Frozen early backbone layers |

```bash
sbatch train_stage2_run2.slurm && sbatch eval_stage2_run2_a30.slurm
sbatch train_stage2_run3.slurm && sbatch eval_stage2_run3.slurm
# and so on for run4, run5
```

### 6. Seed protocol

Single runs are not representative. Every configuration is retrained under five seeds.

```bash
sbatch verify_repro.slurm     # MUST pass before the sweep
sbatch sweep_train.slurm      # 25 tasks, approx. 3 hours each
sbatch sweep_eval.slurm       # 25 tasks, approx. 40 minutes each
python collect_seeds.py
python aggregate_figures.py
```

`verify_repro.slurm` trains seed 0 twice on a truncated loop and compares the metrics
field by field, ignoring wall-clock time. It is a gate, not a formality: a sweep run on a
non-deterministic pipeline measures plumbing noise alongside the effect under study and
cannot separate them.

Per-lesion standard deviation across seeds is 0.098 against 0.014 per crop, a factor of
seven explained entirely by the denominators, 34 patients against 1,279 crops. Treat a
difference as an effect only if it exceeds roughly twice the baseline standard deviation.

### 7. End-to-end chained evaluation

Training is decoupled; inference is chained. Stage 1's predicted boxes become Stage 2's
input, then predictions are aggregated per lesion.

```bash
sbatch eval_pipeline_a30.slurm                   # YOLO detector into Stage 2
sbatch eval_pipeline_frcnn_a30.slurm             # Faster R-CNN into Stage 2
sbatch eval_pipeline_perpatient_a30.slurm        # per-patient aggregation
sbatch eval_pipeline_frcnn_perpatient_a30.slurm
sbatch eval_framing_diagnostic_a30.slurm         # separates localisation from classification cost
```

Reporting both the ground-truth-crop ceiling and the chained figure is the point: the gap
between them is the detector's cost.

### 8. Multimodal fusion

```bash
sbatch run_precompute.slurm      # runs precompute_suv.py and precompute_sc.py
python verify_suv_cache.py
python build_sc_manifest.py
python fix_sc_manifest.py
python build_matched_sc_manifest.py
```

`precompute_suv.py` converts PET to body-weight SUV and resamples onto the annotated CT
geometry through physical coordinates. `precompute_sc.py` recovers the 10,300 annotations
that reference Secondary Capture objects rather than CT images. These were silently
discarded by the original pipeline, which filtered on the CT SOP class alone, so roughly a
third of the available annotations had gone unused.
`build_matched_sc_manifest.py` builds the density-matched control, since the fused
renderings carry 1.40 times as many boxes per patient as the CT arm.

Then the four arms, five seeds each:

```bash
for s in 0 1 2 3 4; do
  sbatch run_ct_cv.slurm         --seed $s
  sbatch run_petct_cv.slurm      --seed $s
  sbatch run_sc_cv.slurm         --seed $s
  sbatch run_sc_matched_cv.slurm --seed $s
done

python compare_arms.py
python plot_metrics.py
python plot_roc.py
```

All four arms run on the same 83 frame-matched patients under the same folds and seeds.
The trainers are identical line for line apart from the dataset import and, for the
matched control, the manifest, so no configuration difference can confound the comparison.
Verify with `diff train_ct_cv.py train_sc_cv.py`.

Each arm writes to `figures/petct/<arm>_nested_s<seed>/`.

Two design points worth knowing before reading the results. Selection is nested: the
outer fold is used for neither training nor checkpoint choice, with a 15 per cent inner
split carved out of the training patients for early stopping. An earlier version selected
on the outer fold and reported it, inflating AUC from 0.779 to 0.807 and shrinking the
standard deviation by a factor of 3.3. Add `--fixed-epochs` to train a set number of
epochs with no selection at all, as a cross-check. Separately, both modalities are cropped
through one window function returning pixel coordinates, so SUV lands on exactly the CT
pixel grid including jitter. An earlier version sampled SUV at the raw box and rescaled,
which misaligned the channels while leaving every metric plausible.

### 9. Confound probe

```bash
python scanner_probe.py                              # full adenocarcinoma and squamous cohort
python scanner_probe.py --cohort pet_cohort_83.csv   # the 83 used by the fusion arms
```

Leave-one-out logistic regression predicting subtype from CT acquisition parameters alone,
scored against a 200-fold label-permutation null. Both cohorts clear: p = 0.180 and
p = 0.835. A fixed threshold would have been unsafe, since the leave-one-out score has a
standard deviation near 0.10 under the null at this sample size.

### 10. Figures

```bash
python plot_stage1_figures.py       # two-architecture ceiling figure
sbatch detection_examples.slurm     # qualitative detection examples
python aggregate_figures.py         # Stage 2, seed-averaged
python plot_metrics.py              # multimodal, per-arm and per-metric
python plot_roc.py                  # nested cross-validation ROC
```

---

## Repository layout

```
preprocessing_pipeline.ipynb     Build manifests, normalisation stats, HU cache
stage1_dataset.py                Source of truth: ROOT, CACHE, window_channels
stage2_dataset.py                Tumour crops
crop_utils.py                    Shared crop geometry and SUV sampling

train_stage1.py                  Faster R-CNN
to_yolo.py  lung.yaml            YOLOv8 conversion and dataset definition
train_stage2*.py                 Baseline plus four ablations
eval_stage2*.py                  Per-crop and per-lesion evaluation
eval_pipeline*.py                Chained two-stage evaluation

precompute_suv.py                PET to SUV, resampled onto CT geometry
precompute_sc.py                 Secondary Capture recovery
*_cv_dataset.py                  The three input representations
train_*_cv.py                    The four fusion arms

*.slurm                          Submission scripts
plot_*.py  aggregate_figures.py  Figures
verify_*.py  sanity.py  probe_*  Verification gates
scanner_probe.py                 Acquisition-confound test
```

---

## Reproducibility

Stage 2 and the fusion arms are fully deterministic: the augmentation generator is seeded
from PyTorch's per-worker generator, cuDNN autotuning is disabled, and deterministic
algorithm selection is requested. `verify_repro.slurm` confirms it by running the same
seed twice and comparing.

**Stage 1 is not.** Its dataset constructs an unseeded NumPy generator and its trainer
enables the cuDNN autotuner, so its augmentation draws and convolution algorithm selection
vary between runs. This does not affect the Stage 1 conclusion, which rests on a gap of
approximately 0.23 reproduced across five configurations and two architectures, but that
conclusion is supported by replication rather than by determinism, and the two are
different arguments.

Determinism holds for a fixed hardware and library stack, not across GPU models or CUDA
versions.

---

## Known limitations

Large cell carcinoma is dropped at five patients. All PET work is binary adenocarcinoma
against squamous, forced by frame agreement leaving three small cell patients and none in
the test split. The fusion conclusions apply to the 83-patient subset rather than the full
cohort. No registration is performed anywhere; PET and CT are related by resampling
through physical coordinates, which is valid only for the frame-matched patients. No
PET-only arm was built.

---

## Licence and citation

The code is released for academic use. The Lung-PET-CT-Dx collection is governed by the
TCIA Data Usage Policy; cite the collection and TCIA if you use it.
