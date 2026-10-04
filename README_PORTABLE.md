# Copy this project to another computer

The primary notebook is `project_demo_windows.ipynb`. Despite its retained filename, its startup now works from this folder on Windows, macOS, and Linux. CPU execution is the default; no NVIDIA GPU or CUDA toolkit is required.

## Copy the files

Copy the entire project folder, including `src`, `configs`, `dataset`, `artifacts` (especially checkpoints), `portable.py`, requirements, scripts, and the main notebook. Ensure OneDrive files are downloaded locally and the transfer finishes before launching.

Do **not** transfer `.venv`, `.lime-test-deps`, `__pycache__`, or `.ipynb_checkpoints`: environments and cached binaries must be recreated for the destination. The old notebooks named `Copy` are historical copies; launch scripts open only the updated main notebook.

The extracted APTOS dataset is already in `dataset/aptos2019`. `Project Files` contains the five original ZIPs as an optional recovery source; these may be omitted from the transfer if the extracted dataset is complete. The notebook restores missing data from this local archive directory, never from a user's Downloads folder. Reports and checkpoint references from the previous computer are automatically rebased to this copy when matching local files exist. Metrics and model weights are not changed by rebasing.

## Windows

Install 64-bit Python **3.11** with the Python launcher. Double-click `setup_windows.bat` once and wait for **Setup complete**. Then double-click `launch_windows.bat`.

If Python is supplied by Anaconda instead of the Windows launcher, activate a Python 3.11 environment in Anaconda Prompt, change into this project folder, and run `python portable.py setup`. Setup creates its own `.venv` inside the project. You can then use `launch_windows.bat` normally.

## macOS or Linux

Install Python **3.11** with venv support, open a terminal in this folder, and run:

```sh
sh setup_unix.sh
sh launch_unix.sh
```

The scripts are designed for Windows/Linux x86-64 and Apple Silicon macOS where the pinned packages provide compatible builds. Other processors, older Intel Macs, and unsupported operating-system versions may need different dependency versions; this is not a universal binary application. Cross-platform setup has not been executed on Mac or Linux here.

## In Jupyter

Select **Python (Medical Dipatie)**. Run the notebook from the beginning. Setup and package installation are separate from the notebook; Run All will not install packages. The notebook finds its root automatically when opened by the launcher. If using another Jupyter server, start it in this folder or set `PROJECT_ROOT_OVERRIDE` in the first code cell.

Grad-CAM, SHAP, and LIME use conservative CPU settings and the saved model. Training remains disabled in the notebook by default. Use `cfg["xai"]["lime_num_samples"]` before Section 7 to adjust LIME's perturbation count; `max_targets` controls the number of original images. CPU explanation runs can be slow.

First setup requires internet and sufficient space for dependencies. Subsequent runs can use the local dataset/model without downloading pretrained weights when the saved checkpoint is reused. If the folder moves after setup, remove **only `.venv`** in the new copy and rerun setup. Do not reuse an environment copied from another machine, even if its folder has the same name.

To check data/model presence without running inference or changing files:

```sh
python portable.py check
```
