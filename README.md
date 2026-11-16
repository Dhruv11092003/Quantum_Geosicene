# Quantum-Enhanced Feature Learning for Multispectral Geoscience Data

**Author:** Dhruv Kulshrestha (25MAG0018)
**Supervisor:** Dr. Vanitha M.
**Institution:** VIT Vellore — PAAML698, Dissertation I (Fall 2026–27)

## Idea

This project studies hybrid quantum-classical deep learning for multispectral satellite image classification, using EuroSAT RGB + NIR (near-infrared) patches. A classical CNN backbone extracts spatial features, which are then passed through small variational quantum circuits (VQCs) before classification.

Eight experiments (EXP-A to EXP-H) are trained on the same dataset, split and training protocol, so their results are directly comparable:

- **EXP-A, EXP-B:** classical baselines.
- **EXP-C to EXP-F:** quantum-hybrid architectures.
- **EXP-G, EXP-H:** matched classical controls for EXP-C and EXP-D, in which the VQC is replaced by a classical layer of the same width.

## Dataset

- **Source:** EuroSAT multispectral (Sentinel-2), Kaggle dataset slug [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset), folder `EuroSATallBands`.
- **Format:** 64×64 GeoTIFF patches, 13 Sentinel-2 bands per patch.
- **Bands used:**
  - RGB: B02, B03, B04
  - NIR: B08
- **Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
- **Size:** 27,597 images. This copy differs from the canonical 27,000-image release: SeaLake has 3,597 images instead of 3,000; all other classes match.
- **Preprocessing:**
  - Pixel values are divided by a reflectance scale of 10,000.
  - Each channel is then z-score normalised using statistics from the training split (5,000 training images).
- **Augmentation (training only):** random horizontal flip (p = 0.5) and random rotation (±15°), applied identically to RGB and NIR.
- **Split:** stratified 70 / 15 / 15 (train / val / test), split seed 42, giving 19,317 train / 4,140 val / 4,140 test images.
  - The split indices are saved and verified by hash before every evaluation (test md5 `5d5ff68d7e75b127`).

All eight experiments use this same dataset and split.

## Experiments

| ID | Description | Parameters |
|---|---|---|
| EXP-A | Classical baseline, RGB only (`ClassicalRGBClassifier`) | 11,173,962 |
| EXP-B | Classical baseline, RGB+NIR (`ClassicalRGBNIRClassifier`) | 11,174,538 |
| EXP-C | Uniform VQC: RGB+NIR combined into a single 8-qubit VQC (`UniformVQCClassifier`) | 11,205,378 |
| EXP-D | Dual VQC: two independent 4-qubit VQCs, one for RGB and one for NIR, fused before the classification head (`DualVQCHybridClassifier`) | 22,405,442 |
| EXP-E | Ablation: RGB-only single-branch VQC (`RGBOnlyVQCClassifier`) | 11,204,006 |
| EXP-F | Ablation: NIR-only single-branch VQC (`NIROnlyVQCClassifier`) | 11,202,854 |
| EXP-G | Classical control for EXP-C: identical to EXP-C, with the 8-qubit VQC replaced by `tanh(Linear(8→8))` (`UniformClassicalBottleneck`) | 11,205,402 |
| EXP-H | Classical control for EXP-D: identical to EXP-D, with each 4-qubit VQC replaced by `tanh(Linear(4→4))` (`DualClassicalBottleneck`) | 22,405,434 |

EXP-G and EXP-H were added after the first round of peer review. They isolate the contribution of the VQC from the effect of the narrow feature bottleneck it imposes. Each keeps the bottleneck width, output range (−1 to 1), head and training protocol of its quantum counterpart.

## Model architectures (as implemented)

- **Backbone (`ReducedResNetBackbone`):** ResNet-18 with a 3×3, stride-1 first convolution and no initial max-pooling (adapted to 64×64 inputs). It outputs 512 pooled features. The number of input channels matches the bands each backbone receives.
- **EXP-A / EXP-B:** backbone → Linear(512 → 10).
- **Projector (models with a VQC or classical substitute):**
  1. Linear(512 → 64) → BatchNorm → ReLU (backbone_dim = 64)
  2. `LinearProjector`: Linear(64 → n_qubits) → Tanh, scaled by π, which maps the features to valid rotation angles.
- **VQC circuit (shared building block, `VariationalQuantumProcessor`):** AngleEmbedding (rotation Y) → StronglyEntanglingLayers (2 layers) → expval(PauliZ) on each wire.
  - Trained with `diff_method="adjoint"`.
  - VQC weights initialised in the range 0.1.
- **EXP-C (`UniformVQCClassifier`):** RGB+NIR (4-channel input) → backbone → projector (→ 8) → 8-qubit VQC → Linear(8 → 128) → ReLU → Linear(128 → 10).
  - 48 VQC parameters (2 × 8 × 3).
- **EXP-D (`DualVQCHybridClassifier`):** separate RGB and NIR backbones, each with a projector (→ 4) and a 4-qubit VQC. The outputs are concatenated (4 + 4 = 8) and passed to Linear(8 → 128) → ReLU → Linear(128 → 10).
  - 48 VQC parameters in total (2 × (2 × 4 × 3)), matched to EXP-C.
- **EXP-E / EXP-F:** single-branch versions of the RGB or NIR half of EXP-D, with head Linear(4 → 128) → ReLU → Linear(128 → 10). Each has 24 VQC parameters.
- **EXP-G / EXP-H:** copies of EXP-C / EXP-D in which only the VQC is replaced:
  - EXP-G: one `tanh(Linear(8 → 8))` layer (72 parameters).
  - EXP-H: two `tanh(Linear(4 → 4))` layers (20 parameters each).

## Training settings

- **Image size and batching:** 64×64 images, batch size 128, 30 epochs.
- **Optimiser:** AdamW, lr 0.001.
  - Weight decay is 1e-5 for classical parameters and 0 for VQC parameters and their classical substitutes.
- **Schedule:** CosineAnnealingLR (T_max = 30).
- **Gradients:** clipped to norm 1.0. Loss is CrossEntropyLoss.
- **Early stopping:** on validation loss (patience 7, min delta 0.001). It never triggered: all selected checkpoints are from epochs 28–30.
- **Seeds:** 3 per experiment (42, 2021, 7).
- **Precision:** EXP-A and EXP-B use automatic mixed precision. All models with a VQC or classical substitute are trained in 32-bit precision.
- **PennyLane backend for training:** lightning.qubit / lightning.gpu.
- **Classifier hidden size:** 128.

## Post-processing

A post-processing step (`postprocess-hes.ipynb`) merges the results of all eight experiments and produces the final figures, tables and exports used in the paper.

**Evaluation**
- Ensemble and per-seed metrics: OA, macro-F1, Cohen's κ, ECE (15 bins), NLL, Brier.
- Bootstrap confidence intervals (10,000 resamples).
- Exact McNemar tests, with Bonferroni correction over 28 pairs.
- Threshold-free selective prediction (AUROC-err, AURC, E-AURC).
- Robustness to additive Gaussian noise.
- Comparison of each VQC model with its classical control (EXP-C vs EXP-G, EXP-D vs EXP-H).

**Hessian**
- The exact Hessian trace with respect to the 48 VQC parameters is computed by calling the circuit directly as a QNode with backpropagation (`default.qubit`, double precision). This is Cell 7.5b.
- The result is reported alongside its Gauss–Newton part and a Hutchinson estimate (30 probes), for all three seeds of EXP-C and EXP-D.
- Second derivatives taken through the `TorchLayer` wrapper with the adjoint method returned only the Gauss–Newton part in finite-difference checks, so they are not used.

**Outputs**
- **Figures:** training curves, confusion matrices, reliability diagrams, Hessian plots, OOD robustness plot, ablation bar chart, risk–coverage plot.
- **Tables:** per-run results, ensemble summary, statistical significance, OOD robustness, Hessian traces, ensemble bootstrap CI, risk–coverage, per-class F1, efficiency, selective prediction (Table 11), VQC vs classical control (Table 12).
- **Exports:** merged metrics, prediction cache, exact Hessians (`hessian_exact.json`) and the control comparison (`control_comparison.json`).

## Checkpoints

Trained model checkpoints are available here: [Google Drive — checkpoints](https://drive.google.com/drive/folders/14zbbiVHdZfGD_aMrWDZCUdnzqSXfOivH)

## Environment

- **Platform:** Kaggle, GPU accelerator (T4 ×2 for training, T4 for post-processing).
- **Core libraries:** PyTorch + PennyLane 0.40.0 (autoray < 0.8.0).
  - Training used the lightning.qubit / lightning.gpu backend.
  - The final post-processing run used `default.qubit`, because lightning binaries were not available in the Kaggle Python 3.13 image.
- **Dataset attachment:** `apollo2506/eurosat-dataset` via Kaggle "Add Data".

## Reproducibility

- All hyperparameters are serialised to `exports/configuration.json`, with an MD5 hash embedded for verification.
- Each experiment is run with 3 fixed seeds (42, 2021, 7).
- Training is resumable: a `completed.json` file is written once all seeds of an experiment finish, and later runs skip retraining.
