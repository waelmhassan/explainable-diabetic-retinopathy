# Run the Windows notebook

**Updated portable startup:** follow `README_PORTABLE.md` and use the setup/launch scripts. The dataset and optional ZIP archives are now inside this project folder. References below to Downloads describe the original import, not the current startup requirement. The main notebook no longer installs packages during Run All; requirements are shared in `requirements.txt`.

## LIME addition

The main notebook now runs Grad-CAM, SHAP, and LIME in Section 7. In the active notebook kernel, run `%pip install lime==0.2.0.1 scikit-image==0.25.2` once, restart the kernel, reopen the updated notebook, and run from the beginning. Do not save an older open notebook over this updated file. The separate `project_demo_windows - Copy.ipynb` was left unchanged.

LIME uses the same selected audit images, 256 perturbations per image, batches of 4, and approximately 50 SLIC superpixels. Configure these under `xai` in `configs/base.yaml`. Red/blue maps and numerical arrays go to `artifacts/reports/figures/lime`; per-image checks go to `rq3_lime_seed1988_test.csv`. The summary table/chart includes all three methods; `rq_xai_all_pairs_seed1988_test.csv` adds all three paired comparisons. The old two-method pairwise table is retained for compatibility.

The LIME panel shows signed coefficients, while the shared border/masking audit uses positive evidence, matching the default SHAP positive mode. Perturbations operate on the preprocessed image with the same normalization as inference. Local R² measures the surrogate's fit to perturbed model outputs, not diagnostic validity. Two audit images and 256 perturbations are demonstration settings, not a robust comparative study. Failed explanations are recorded explicitly in a LIME status JSON and omitted from paired comparisons.

Open `project_demo_windows.ipynb` in this folder. The original Downloads notebook was preserved. This is a research demonstration of the existing saved model; the small XAI sample is for checking operation, not reproducing the full audit.

## First-time environment setup

Use an installed Python 3.11 interpreter. Python and Jupyter were not found on the inspected shell PATH; an existing Anaconda environment may still be available. In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --user --name medical-dipatie --display-name "Python (Medical Dipatie)"
.\.venv\Scripts\python.exe -m notebook project_demo_windows.ipynb
```

If `py` is unavailable, use the full path to your Python 3.11 interpreter for the first command. Environment activation is unnecessary. In an existing Jupyter installation, select **Python (Medical Dipatie)** as the kernel after registering it.

Dependencies have not been installed or runtime-tested during this review. The original pinned dependencies are retained; OpenCV (required by `import cv2`), Notebook, and ipykernel are added. The CPU PyTorch version pair is listed in https://pytorch.org/get-started/previous-versions/.

## Run order and dataset

Run cells in order. The first cell locates `src/dr_pipeline.py` and `configs/base.yaml` from the current folder or its parents. If needed, set `PROJECT_ROOT_OVERRIDE` to this folder's absolute path.

The dataset is still in the five supplied ZIP archives under `Downloads/XAI_medical/Project Files`. The first cell restores only `dataset/aptos2019`, skipping files already present at the expected size. Keep all five archives and allow approximately 8.3 GiB of free space plus environment/output space. Change `ARCHIVE_DIR` if the archives move. Restoration takes time and may trigger OneDrive synchronization. It does not download data or train the model.

All 3,662 image names referenced by the supplied CSVs were found across the archives: 2,930 training-folder images, 366 validation-folder images, and 366 test-folder images. The benchmark code pools and re-splits these using seed 1988; these source folder counts are not the final evaluation split counts. Archive payload integrity and image decoding have not been exhaustively tested.

The standalone Roboflow ZIP is not needed by the configured `aptos_only` workflow. The existing EfficientNet-B4 checkpoint and calibration are already copied into `artifacts/checkpoints`.

## Windows changes

- Replaced Colab setup with local project discovery and dataset restoration.
- Retained portable configuration-relative paths; data preparation regenerates manifests with local Windows paths. Archived Mac paths in run logs can be recovered by the pipeline's existing checkpoint filename search.
- Defaults to CPU, batch size 4, and `num_workers=0` to avoid Windows notebook multiprocessing issues. GPU and RAM could not be queried in this session, so no hardware performance claim is made.
- Enabled all three existing XAI safe modes, with two audit targets and two SHAP background images. Visual review can still explain two images and the single-case section explains one. CPU SHAP can remain slow.
- Kept the saved model's architecture, image size, preprocessing, seed, and signature-relevant training settings unchanged.
- Added a checkpoint compatibility check before the training cell. `ALLOW_TRAINING=False` makes a missing/unreadable/mismatched checkpoint an error instead of silently starting training. Keep cleanup and force-retrain disabled for the demo.
- Cleared old outputs, including historical figures and errors, so displayed results come from your run.

The underlying pipeline already supports CPU/CUDA/MPS and loads checkpoints with `map_location`. Its SHAP retry logic is MPS-specific; this copy uses CPU XAI directly. The supplied README reports an unsuccessful historical committee audit (missing Captum and SHAP memory exhaustion), so archived figures do not establish a successful fresh run.

## Verification performed

Notebook JSON and all Python cells were parsed, pipeline syntax checked, and original notebook SHA256 verified unchanged. No model execution, package installation, dataset extraction, or training was performed. After installing dependencies, run setup, data preparation, checkpoint reuse, and core evaluation before the XAI sections. If checkpoint reuse fails, resolve the reported issue before enabling training.
