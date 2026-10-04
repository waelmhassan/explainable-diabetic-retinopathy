# Explainable Diabetic Retinopathy Classification

This repository contains a reproducible deep-learning workflow for classifying retinal fundus images into five diabetic-retinopathy severity levels: No_DR, Mild, Moderate, Severe, and Proliferative_DR. It compares four transfer-learning architectures within one shared pipeline and generates Grad-CAM, SHAP, and LIME explanations.

## Project goal

The project evaluates whether pretrained image classifiers can accurately classify ordered diabetic-retinopathy grades while accounting for class imbalance. The primary task is five-class supervised image classification. The secondary goal is to inspect model decisions using post-hoc explainability methods.

## Dataset

The experiments use the APTOS 2019 Blindness Detection dataset.

- Source: [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- Downloaded dataset size used by this project: 3,662 retinal fundus images
- Labels: `0=No_DR`, `1=Mild`, `2=Moderate`, `3=Severe`, `4=Proliferative_DR`
- Benchmark split: 2,800 training / 312 validation / 550 test images
- Split method: stratified and disjoint, seed `1988`

Before publishing the repository, record the exact dataset download date and confirm the dataset licence/competition terms from Kaggle. Do not upload patient images, dataset archives, checkpoints, or generated artifacts unless the dataset terms permit redistribution.

## Data preparation

All four baseline models use the same input processing and split:

- image size: `380 x 380`
- circle crop and CLAHE contrast enhancement
- Gaussian smoothing and Ben Graham colour normalization
- ImageNet normalization
- training augmentation: horizontal flip, rotation, small affine transforms, brightness, and contrast jitter

The original class distribution is imbalanced. The training configuration uses inverse-frequency class weights and focal loss (`gamma=2.0`); weighted sampling is disabled. Macro-F1 and quadratic weighted kappa (QWK) are reported alongside accuracy because disease classes are imbalanced and ordered.

## Algorithms

The project compares the following verified baseline models. VGG16 is intentionally excluded from the reported comparison because its notebook retained stale output.

| Model | Approximate parameters | Test accuracy | Macro-F1 | QWK | Recorded training time |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 8.0 M | 84.00% | 71.58% | 0.907 | 266.7 min |
| EfficientNet-B4 | 19.3 M | 82.18% | 71.05% | 0.896 | Not retained* |
| ViT-B/16 | 86.6 M | 82.00% | 69.51% | 0.886 | 172.5 min |
| ResNet-50 | 25.6 M | 79.27% | 67.95% | 0.894 | 78.9 min |

\* The EfficientNet-B4 baseline notebook reused a checkpoint, so its notebook history contains no epoch-time logs. The value is deliberately not estimated.

Training times are the sum of recorded `epoch_time` values in notebook output history. The values are hardware-dependent and should be compared only when runs use the same environment.

## Loss-function ablation

Each ablation uses the exact same backbone, split, preprocessing, seed, class weights, batch size, and optimizer settings as its baseline. The only intended change is `focal` loss to `cross_entropy`; `FORCE_RETRAIN=True` ensures that a prior checkpoint cannot be reused.

| Model | Cross-entropy epochs | Training time | Accuracy | Macro-F1 | QWK | Macro-F1 change vs. focal |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet-121 | 6 | 87.3 min | 79.64% | 66.94% | 0.881 | -4.64 pp |
| EfficientNet-B4 | 10 | 187.7 min | 79.64% | 66.88% | 0.870 | -4.18 pp |
| ResNet-50 | 10 | 132.9 min | 83.45% | 70.39% | 0.903 | +2.44 pp |
| ViT-B/16 | 6 | 103.5 min | 78.55% | 57.05% | 0.819 | -12.45 pp |

Focal loss improved macro-F1 for DenseNet-121, EfficientNet-B4, and ViT-B/16. ResNet-50 was the exception, demonstrating that the loss-function choice is architecture-dependent.

## Repository layout

```text
configs/base.yaml                         Shared configuration
src/dr_pipeline.py                        Data, model, training, evaluation, and XAI pipeline
project_densenet.ipynb                    DenseNet-121 baseline
project_efficientnet_b4.ipynb             EfficientNet-B4 baseline
project_resnet50.ipynb                    ResNet-50 baseline
project_vit_b16.ipynb                     ViT-B/16 baseline
project_*_ablation_test.ipynb             Cross-entropy ablation notebooks
project_vgg.ipynb                         VGG16 experiment; excluded from reported results
artifacts/                                Local checkpoints, predictions, reports, and figures
dataset/                                  Local APTOS dataset files; not for Git upload
```

## Installation

Python 3.11 is recommended.

### Windows

1. Run `setup_windows.bat` once.
2. Run `launch_windows.bat`.
3. Select the `Python (Medical Dipatie)` kernel in Jupyter.

### macOS or Linux

```bash
sh setup_unix.sh
sh launch_unix.sh
```

Alternatively, create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Reproducing a baseline

1. Place the APTOS dataset under `dataset/aptos2019` using the expected CSV and image-folder layout in `configs/base.yaml`.
2. Start Jupyter from the repository folder.
3. Open one of the four baseline notebooks.
4. In the configuration cell, set the relevant `PIPELINE_BACKBONE` and ensure `ALLOW_TRAINING=True`.
5. Set `FORCE_RETRAIN=True` when a fresh model is required. This is necessary when changing loss, epochs, image size, or other training conditions.
6. Run all cells sequentially. Test metrics and output tables are written under `artifacts/runs/<model_version>/`.

Key baseline configuration:

```python
SEED = 1988
cfg['training'].update(
    device='auto',
    num_workers=0,
    batch_size=4,
    epochs=15,
)
```

The shared configuration is in `configs/base.yaml`. Baselines use `loss_name: focal`, class weights enabled, and weighted sampling disabled.

## Reproducing an ablation

Open the relevant `project_*_ablation_test.ipynb` notebook. The ablation configuration keeps the baseline settings and changes only:

```python
FORCE_RETRAIN = True
cfg['training']['loss_name'] = 'cross_entropy'
cfg['training'].update(epochs=15, batch_size=4, num_workers=0, device='auto')
```

Early stopping may finish before 15 epochs. This is expected and is part of the configured training procedure.

## Explainability outputs

The pipeline supports:

- Grad-CAM for CNN activation maps
- SHAP for attribution analysis
- LIME for local superpixel explanations

These outputs assist in inspecting whether the model focuses on clinically plausible retinal regions. They are not clinical diagnostic tools and must be reviewed by qualified clinicians before any real-world use.

## Limitations

- Results use one fixed split and seed; repeated-seed experiments and confidence intervals would strengthen the conclusions.
- Training times depend on hardware and whether a checkpoint was reused.
- The dataset is class-imbalanced and adjacent grades can be clinically difficult to distinguish.
- This project is for academic research and education, not clinical deployment.

## Contributions

The following responsibilities map directly to the Assignment 1 deliverables. Git history should show meaningful activity from every group member.

| Team member | Contribution |
|---|---|

| Ala' Abu Hilal | 
- Dataset task: Analyzed split reproducibility, feature dimensions, and manifest SHA-256 consistency.   
- Algorithms task: Coded, configured, and trained the DenseNet-121 model architecture.
- Results-comparison task: Prepared the comparative metrics, training-time analysis, model-size, and parameter-count tables.
- Results-discussion task: Analyzed the impact of loss-function changes and loss formulation interactions on model stability. 
- Reporting task: Reviewed, edited, finalized the comprehensive technical report, and document layout. |

| Amal El Ahmad | 
- Dataset task: Conducted exact duplicate detection, SHA-256 pixel/byte hashing, and partition leakage analysis.
- Algorithms task: Coded and ran the VGG-16 baseline model, maintained shared preprocessing scripts, and managed checkpoint configurations.
- Results-comparison task: Benchmarked training loop durations, validation overhead, and per-epoch runtime efficiencies.
- Results-discussion task: Completed the controlled loss-function ablation analysis comparing focal loss with Cross-Entropy.
- Reporting task: Formatted dataset audit tables, class distribution figures, and Prepared the three-slide presentation. |

| Wael Mohamed | 
- Dataset task: Investigated data quality, empty manifest fields, and patient identity mapping.   
- Algorithms task: Coded, implemented, and ran the ResNet-50 CNN baseline, EfficientNet-B4 checkpoint reused collaboratively, and prepared the GitHub repository.
- Results-comparison task: Evaluated macro precision, recall, and accuracy benchmarks across all model execution environments.
- Results-discussion task: Interpreted model ranking shifts observed between focal loss and Cross-Entropy conditions.
- Reporting task: Formatted experiment logs, reproducibility guidelines, and GitHub codebase documentation. |

| Lama Al Ibaisi | 
- Dataset task: Evaluated class distribution, split counts, and stratification balance across the five DR grades.
- Algorithms task: Coded and executed the ViT-B/16 vision transformer pipeline, patch tokenization, and training.
- Results-comparison task: Analyzed Quadratic Weighted Kappa (QWK) and class-level performance breakdowns for ordinal grading.
- Results-discussion task: Explained observed performance rankings using dataset properties, class imbalance, and visual features.
- Reporting task: Prepared the remaining two-slide presentation, and key summary visualizations. |


## Use of AI assistance

AI assistance was used to help draft code, documentation, and explanations. All experiments, configurations, reported metrics, and final interpretations were reviewed by the project team. This disclosure should be retained or adapted to match the team’s actual use.
