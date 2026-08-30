# Quantum-Enhanced Feature Learning for Multispectral Geoscience Data

**Author:** Dhruv Kulshrestha (25MAG0018)
**Supervisor:** Dr. Vanitha M.
**Institution:** VIT Vellore — PAAML698, Dissertation I (Fall 2026–27)

---

## Idea

This project studies hybrid quantum-classical deep learning for **multispectral satellite image classification**, using **EuroSAT** RGB + NIR (near-infrared) patches. A classical CNN backbone extracts spatial features, which are then passed through small **variational quantum circuits (VQCs)** before classification. Six experiments (EXP-A through EXP-F) compare classical baselines against several quantum-hybrid architectures, all trained on the same dataset, split, and training protocol so results are directly comparable.

---

## Dataset

- **Source:** EuroSAT multispectral (Sentinel-2), Kaggle dataset slug `apollo2506/eurosat-dataset`.[`LINK`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **Format:** 64×64 GeoTIFF patches, 13 Sentinel-2 bands per patch.
- **Bands used:**
  - RGB: B02, B03, B04
  - NIR: B08
- **Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
- **Preprocessing:** pixel values divided by a reflectance scale of 10,000, then z-score normalised per channel using training-split statistics.
- **Split:** stratified 70 / 15 / 15 (train / val / test), split seed 42 → 18,900 train / 4,050 val / 4,050 test images.

All six experiments (A–F) are trained on this same dataset and split, so results across the classical and quantum arms are directly comparable.

---

## Experiments

| ID | Description |
|----|-------------|
| **EXP-A** | Classical baseline, RGB only (`ClassicalRGBClassifier`) |
| **EXP-B** | Classical baseline, RGB+NIR (`ClassicalRGBNIRClassifier`) |
| **EXP-C** | Uniform VQC baseline: RGB+NIR combined into a single 8-qubit VQC (`UniformVQCClassifier`) |
| **EXP-D** | **Proposed architecture** — two independent 4-qubit VQCs, one for RGB and one for NIR, fused before the classification head (`DualVQCHybridClassifier`) |
| **EXP-E** | Ablation: RGB-only single-branch VQC (`RGBOnlyVQCClassifier`) |
| **EXP-F** | Ablation: NIR-only single-branch VQC (`NIROnlyVQCClassifier`) |

### Model architectures (as implemented)

- **VQC circuit** (shared building block, `VariationalQuantumProcessor`): `AngleEmbedding` (rotation Y) → `StronglyEntanglingLayers` → `expval(PauliZ)` per wire. 2 layers. Trained with `diff_method="adjoint"`; `parameter_shift` is used only for Hessian analysis.
- **LinearProjector:** `Linear(backbone_dim=64 → n_qubits)` + `Tanh`, scaled by π, to map backbone features into valid rotation angles.
- **EXP-C (UniformVQCClassifier):** RGB+NIR (4-channel input) → `ReducedResNetBackbone` → `LinearProjector(64→8)` → 8-qubit VQC → `Linear(8→128)` → ReLU → `Linear(128→10)`. 48 VQC parameters (2×8×3).
- **EXP-D (DualVQCHybridClassifier, proposed):** separate RGB and NIR backbones and 4-qubit VQCs, concatenated (4+4=8) before `Linear(8→128) → Linear(128→10)`. 48 VQC parameters total (2×(2×4×3)), matched to EXP-C for a fair comparison.
- **EXP-E / EXP-F:** single-branch versions of the RGB or NIR half of EXP-D (24 VQC parameters each), used to isolate the contribution of each spectral branch.

### Training settings

- Image size 64×64, batch size 128, 30 epochs, Adam (lr 0.001, betas (0.9, 0.999)), `CosineAnnealingLR` (T_max = max_epochs).
- 3 seeds per experiment: 42, 2021, 7.
- PennyLane backend: `lightning.qubit` / `lightning.gpu`.
- Backbone output dimension: 64. VQC init range: 0.1. Classifier hidden size: 128.

---

## Post-Processing

A post-processing step merges results from all six experiments and produces the final figures, tables, and exports used in the paper:

**Figures:** training curves, confusion matrices, reliability diagrams, Hessian landscape plots, OOD robustness plot, ablation bar chart, risk-coverage plot.

**Tables:** per-run results, ensemble summary, statistical significance, OOD robustness, Hessian traces, ensemble bootstrap CI, risk-coverage, per-class F1, efficiency.

**Exports:** recomputed Hessians for EXP-C and EXP-D, merged metrics, prediction cache, and a publishability audit.

---

## Checkpoints

Trained model checkpoints are available here:
[Google Drive — checkpoints](https://drive.google.com/drive/folders/14zbbiVHdZfGD_aMrWDZCUdnzqSXfOivH?usp=sharing)

---

## Environment

- **Platform:** Kaggle, GPU accelerator (T4 ×2 for training, T4 for post-processing).
- **Core library:** PyTorch + PennyLane (`lightning.qubit` / `lightning.gpu` backend).
- **Dataset attachment:** `apollo2506/eurosat-dataset` via Kaggle "Add Data".

## Reproducibility

- All hyperparameters are serialised to `exports/configuration.json` with an MD5 hash embedded for verification.
- Each experiment is run across 3 fixed seeds (42, 2021, 7).
- Training is resumable: a `completed.json` file is written once all seeds for an experiment finish, and subsequent runs skip retraining.