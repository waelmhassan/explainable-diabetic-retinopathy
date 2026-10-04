from __future__ import annotations

import copy
import json
import os
import random
import shutil
import types
import warnings
import hashlib
import math
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    ViT_B_16_Weights,
    densenet121,
    efficientnet_b4,
    resnet50,
    vgg16,
    vit_b_16,
)
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, Bottleneck

try:
    from captum.attr import LayerAttribution, LayerGradCam
except Exception as exc:  # pragma: no cover
    LayerAttribution = None
    LayerGradCam = None
    _CAPTUM_IMPORT_ERROR = exc
else:
    _CAPTUM_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

import yaml


# -----------------------------
# Config + common utils
# -----------------------------

_REQUIRED_PATH_KEYS = [
    "dataset_root",
    "aptos_root",
    "aptos_train_csv",
    "aptos_valid_csv",
    "aptos_test_csv",
    "aptos_train_image_dir",
    "aptos_valid_image_dir",
    "aptos_test_image_dir",
    "artifacts_root",
    "manifests_dir",
    "checkpoints_dir",
    "predictions_dir",
    "logs_dir",
    "reports_root",
    "figures_dir",
    "tables_dir",
]


def _infer_project_root(cfg_path: Path) -> Path:
    if cfg_path.parent.name == "configs":
        return cfg_path.parent.parent.resolve()
    return Path.cwd().resolve()


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")

    project_root = _infer_project_root(cfg_path)

    paths = dict(cfg.get("paths", {}))
    missing = [k for k in _REQUIRED_PATH_KEYS if k not in paths]
    if missing:
        raise ValueError(f"Missing required config paths: {missing}")

    for key, value in paths.items():
        p = Path(str(value))
        if not p.is_absolute():
            p = (project_root / p).resolve()
        paths[key] = str(p)

    cfg["paths"] = paths
    cfg["project_root"] = str(project_root)

    for key in [
        "dataset_root",
        "aptos_root",
        "aptos_train_image_dir",
        "aptos_valid_image_dir",
        "aptos_test_image_dir",
        "artifacts_root",
        "manifests_dir",
        "checkpoints_dir",
        "predictions_dir",
        "logs_dir",
        "reports_root",
        "figures_dir",
        "tables_dir",
    ]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    return cfg


def _cfg(cfg_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg_or_path, dict):
        cfg = dict(cfg_or_path)
        if "project_root" not in cfg:
            raise ValueError("Config dict must contain project_root. Use load_project_config(path).")
        for key in [
            "manifests_dir",
            "checkpoints_dir",
            "predictions_dir",
            "logs_dir",
            "figures_dir",
            "tables_dir",
        ]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
        return cfg
    return load_project_config(cfg_or_path)


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_alias_copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == dst_path.resolve():
        return
    shutil.copy2(src_path, dst_path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_device(requested: str = "auto") -> torch.device:
    req = str(requested).lower().strip()
    mps_available = torch.backends.mps.is_available()

    if req in {"", "auto", "default"}:
        if mps_available:
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if req == "mps":
        if mps_available:
            return torch.device("mps")
        raise RuntimeError("Requested device=mps but MPS is not available")

    if req in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device=cuda but CUDA is not available")

    if req == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device request: {requested}. Use one of: auto, mps, cuda, cpu.")


def _resolve_xai_device(conf: dict[str, Any]) -> torch.device:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    requested = xai_cfg.get("device", None)
    if requested is None or str(requested).strip() == "":
        requested = conf.get("training", {}).get("device", "auto")
    return _resolve_device(str(requested))


# -----------------------------
# Data
# -----------------------------


def _infer_laterality(filename: str) -> str:
    name = filename.lower()
    if "_left_" in name:
        return "left"
    if "_right_" in name:
        return "right"
    return "unknown"


def _class_name_from_id(class_id: int, label_order: list[str]) -> str:
    cid = int(class_id)
    if cid < 0 or cid >= len(label_order):
        raise ValueError(f"class_id out of range: {cid}")
    return str(label_order[cid])


def _normalize_label_token(raw: str) -> str:
    return "".join(ch.lower() for ch in str(raw).strip() if ch.isalnum())


def _parse_aptos_csv(
    csv_path: str | Path,
    image_dir: str | Path,
    label_order: list[str],
    source_name: str,
    source_dataset: str = "aptos2019",
) -> pd.DataFrame:
    csv_file = Path(csv_path)
    image_root = Path(image_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image dir not found: {image_root}")

    df = pd.read_csv(csv_file)
    required_cols = ["id_code", "diagnosis"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")

    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        id_code = str(row["id_code"]).strip()
        filename = f"{id_code}.png"
        class_id = int(row["diagnosis"])
        class_name = _class_name_from_id(class_id, label_order)
        image_path = image_root / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        rows.append(
            {
                "sample_id": f"{source_name}_{id_code}_{int(row_idx)}",
                "patient_id": id_code,
                "split": source_name,
                "filename": filename,
                "image_path": str(image_path),
                "class_id": int(class_id),
                "class_name": class_name,
                "laterality": _infer_laterality(filename),
                "source_dataset": str(source_dataset),
            }
        )

    return pd.DataFrame(rows)


def _parse_roboflow_classes_csv(
    classes_csv: str | Path,
    image_dir: str | Path,
    label_order: list[str],
    source_name: str,
    source_dataset: str = "roboflow",
) -> pd.DataFrame:
    csv_file = Path(classes_csv)
    image_root = Path(image_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"Roboflow classes CSV not found: {csv_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Roboflow image dir not found: {image_root}")

    df = pd.read_csv(csv_file)
    if "filename" not in df.columns:
        raise ValueError(f"Roboflow classes CSV must include filename column: {csv_file}")

    class_cols = [c for c in df.columns if str(c).strip().lower() != "filename"]
    if not class_cols:
        raise ValueError(f"No class columns found in Roboflow classes CSV: {csv_file}")

    label_token_to_id = {_normalize_label_token(name): idx for idx, name in enumerate(label_order)}
    col_to_class_id: dict[str, int] = {}
    unknown_cols: list[str] = []
    for col in class_cols:
        token = _normalize_label_token(col)
        if token in label_token_to_id:
            col_to_class_id[col] = int(label_token_to_id[token])
        else:
            unknown_cols.append(str(col))
    if unknown_cols:
        raise ValueError(
            f"Roboflow class columns not found in label_order: {unknown_cols}. "
            f"label_order={label_order}"
        )

    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        filename = str(row["filename"]).strip()
        image_path = image_root / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing Roboflow image file: {image_path}")

        active_cols: list[str] = []
        for col in class_cols:
            try:
                val = int(round(float(row[col])))
            except (TypeError, ValueError):
                val = 0
            if val == 1:
                active_cols.append(col)
        if len(active_cols) != 1:
            raise ValueError(
                f"Expected exactly one active class for row={row_idx}, filename={filename}, "
                f"but found {len(active_cols)} active columns: {active_cols}"
            )

        active_col = active_cols[0]
        class_id = int(col_to_class_id[active_col])
        class_name = _class_name_from_id(class_id, label_order)
        patient_token = filename.split(".rf.")[0]

        rows.append(
            {
                "sample_id": f"{source_name}_{patient_token}_{int(row_idx)}",
                "patient_id": f"rf::{patient_token}",
                "split": source_name,
                "filename": filename,
                "image_path": str(image_path),
                "class_id": int(class_id),
                "class_name": class_name,
                "laterality": _infer_laterality(filename),
                "source_dataset": str(source_dataset),
            }
        )

    return pd.DataFrame(rows)


def _load_aptos_full_pool(conf: dict[str, Any], label_order: list[str]) -> pd.DataFrame:
    required_aptos_keys = [
        "aptos_train_csv",
        "aptos_valid_csv",
        "aptos_test_csv",
        "aptos_train_image_dir",
        "aptos_valid_image_dir",
        "aptos_test_image_dir",
    ]
    missing_aptos = [k for k in required_aptos_keys if k not in conf["paths"]]
    if missing_aptos:
        raise KeyError(f"Missing APTOS path keys in config: {missing_aptos}")

    aptos_train = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_train_csv"],
        image_dir=conf["paths"]["aptos_train_image_dir"],
        label_order=label_order,
        source_name="source_aptos_train",
        source_dataset="aptos2019",
    )
    aptos_valid = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_valid_csv"],
        image_dir=conf["paths"]["aptos_valid_image_dir"],
        label_order=label_order,
        source_name="source_aptos_valid",
        source_dataset="aptos2019",
    )
    aptos_test = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_test_csv"],
        image_dir=conf["paths"]["aptos_test_image_dir"],
        label_order=label_order,
        source_name="source_aptos_test",
        source_dataset="aptos2019",
    )
    aptos_full = pd.concat([aptos_train, aptos_valid, aptos_test], ignore_index=True)
    if aptos_full["patient_id"].astype(str).duplicated().any():
        dup = int(aptos_full["patient_id"].astype(str).duplicated().sum())
        raise RuntimeError(f"Duplicate patient_id entries detected in APTOS pool: {dup}")
    return aptos_full


def _load_roboflow_full_pool(conf: dict[str, Any], label_order: list[str]) -> pd.DataFrame:
    paths_cfg = conf.get("paths", {})
    root = Path(str(paths_cfg.get("roboflow_root", Path(paths_cfg["dataset_root"]) / "roboflow")))
    train_dir = Path(str(paths_cfg.get("roboflow_train_dir", root / "train")))
    valid_dir = Path(str(paths_cfg.get("roboflow_valid_dir", root / "valid")))
    train_csv = Path(str(paths_cfg.get("roboflow_train_labels_csv", train_dir / "_classes.csv")))
    valid_csv = Path(str(paths_cfg.get("roboflow_valid_labels_csv", valid_dir / "_classes.csv")))

    rf_train = _parse_roboflow_classes_csv(
        classes_csv=train_csv,
        image_dir=train_dir,
        label_order=label_order,
        source_name="source_roboflow_train",
        source_dataset="roboflow",
    )
    rf_valid = _parse_roboflow_classes_csv(
        classes_csv=valid_csv,
        image_dir=valid_dir,
        label_order=label_order,
        source_name="source_roboflow_valid",
        source_dataset="roboflow",
    )
    return pd.concat([rf_train, rf_valid], ignore_index=True)


def _data_source(conf: dict[str, Any]) -> str:
    return str(conf.get("data", {}).get("source", "aptos_only")).strip().lower()


def _load_dataset_pool(conf: dict[str, Any], label_order: list[str]) -> pd.DataFrame:
    source = _data_source(conf)
    aptos_aliases = {"aptos_only", "aptos", "aptos2019"}
    roboflow_aliases = {"roboflow_only", "roboflow", "rf"}
    mixed_aliases = {"mixed", "aptos_roboflow", "aptos_plus_roboflow"}

    if source in aptos_aliases:
        return _load_aptos_full_pool(conf, label_order=label_order)
    if source in roboflow_aliases:
        return _load_roboflow_full_pool(conf, label_order=label_order)
    if source in mixed_aliases:
        aptos_df = _load_aptos_full_pool(conf, label_order=label_order)
        roboflow_df = _load_roboflow_full_pool(conf, label_order=label_order)
        merged = pd.concat([aptos_df, roboflow_df], ignore_index=True)
        merged["patient_id"] = merged["source_dataset"].astype(str) + "::" + merged["patient_id"].astype(str)
        return merged
    raise ValueError(
        f"Unsupported data.source={source}. Use one of: aptos_only, roboflow_only, mixed."
    )


def _split_train_val_test(
    full_df: pd.DataFrame,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    stratify_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios_sum = train_ratio + test_ratio + val_ratio
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, test={test_ratio}, val={val_ratio}, sum={ratios_sum}"
        )
    if min(train_ratio, test_ratio, val_ratio) <= 0:
        raise ValueError("All split ratios must be > 0")

    train_df, holdout_df = train_test_split(
        full_df,
        train_size=train_ratio,
        random_state=stratify_seed,
        stratify=full_df["class_id"],
    )

    holdout_ratio = test_ratio + val_ratio
    rel_test_ratio = test_ratio / holdout_ratio

    test_df, val_df = train_test_split(
        holdout_df,
        train_size=rel_test_ratio,
        random_state=stratify_seed,
        stratify=holdout_df["class_id"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _split_train_val_only(
    full_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    stratify_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_sum = float(train_ratio + val_ratio)
    if ratio_sum <= 0:
        raise ValueError(f"train_ratio + val_ratio must be > 0, got {ratio_sum}")
    train_size_rel = float(train_ratio / ratio_sum)
    train_df, val_df = train_test_split(
        full_df,
        train_size=train_size_rel,
        random_state=stratify_seed,
        stratify=full_df["class_id"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _data_protocol(conf: dict[str, Any]) -> str:
    return str(conf.get("data", {}).get("protocol", "default")).strip().lower()


def _is_benchmark(conf: dict[str, Any]) -> bool:
    return _data_protocol(conf) == "benchmark"


def _use_legacy_aliases(conf: dict[str, Any]) -> bool:
    return not _is_benchmark(conf)


def _slug_token(raw: Any, fallback: str = "value") -> str:
    token = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(raw))
    token = "_".join([part for part in token.split("_") if part])
    return token or fallback


def _profile_dataset_tag(conf: dict[str, Any]) -> str:
    source_raw = _data_source(conf)
    aliases = {
        "aptos_only": "aptos2019",
        "aptos": "aptos2019",
        "aptos2019": "aptos2019",
        "roboflow_only": "roboflow",
        "roboflow": "roboflow",
        "rf": "roboflow",
        "mixed": "aptos_roboflow",
        "aptos_roboflow": "aptos_roboflow",
        "aptos_plus_roboflow": "aptos_roboflow",
    }
    if source_raw in aliases:
        return aliases[source_raw]
    if "aptos" in source_raw:
        return "aptos2019"
    if "roboflow" in source_raw:
        return "roboflow"
    return _slug_token(source_raw, fallback="dataset")


def _profile_split_tag(conf: dict[str, Any]) -> str:
    data_cfg = conf.get("data", {})
    test_ratio = _to_ratio_fraction(data_cfg.get("profile_test_ratio", data_cfg.get("test_ratio", 0.15)), "profile_test_ratio")
    train_ratio = max(0.0, 1.0 - test_ratio)
    val_ratio = _to_ratio_fraction(
        data_cfg.get("profile_val_ratio_within_train", data_cfg.get("val_ratio", 0.10)),
        "profile_val_ratio_within_train",
    )
    return f"{int(round(train_ratio * 100))}-{int(round(test_ratio * 100))}-v{int(round(val_ratio * 100))}"


def _profile_profile_tag(conf: dict[str, Any]) -> str:
    return f"{_profile_dataset_tag(conf)}_{_profile_split_tag(conf)}"


def _manifest_suffix(conf: dict[str, Any], seed: int | None = None) -> str:
    if not _is_benchmark(conf):
        return ""
    use_seed = int(seed if seed is not None else conf.get("data", {}).get("stratify_seed", 1988))
    return f"_{_profile_profile_tag(conf)}_seed{use_seed}"


def _manifest_filename_map(conf: dict[str, Any], seed: int | None = None) -> dict[str, str]:
    suffix = _manifest_suffix(conf, seed=seed)
    if suffix:
        return {
            "full_data": f"full_data{suffix}.csv",
            "test_full": f"test_full{suffix}.csv",
            "train_split": f"train_split{suffix}.csv",
            "val_split": f"val_split{suffix}.csv",
            "summary": f"summary{suffix}.json",
        }
    return {
        "full_data": "full_data.csv",
        "test_full": "test_full.csv",
        "train_split": "train_split.csv",
        "val_split": "val_split.csv",
        "summary": "summary.json",
    }


def _manifest_outputs(conf: dict[str, Any], seed: int | None = None) -> dict[str, str]:
    manifests_dir = Path(conf["paths"]["manifests_dir"])
    names = _manifest_filename_map(conf, seed=seed)
    return {
        "full_data": str((manifests_dir / names["full_data"]).resolve()),
        "test_full": str((manifests_dir / names["test_full"]).resolve()),
        "train_split": str((manifests_dir / names["train_split"]).resolve()),
        "val_split": str((manifests_dir / names["val_split"]).resolve()),
        "summary": str((manifests_dir / names["summary"]).resolve()),
    }


def _split_two_stage_stratified_pool(
    full_df: pd.DataFrame,
    test_ratio: float,
    val_ratio_within_train: float,
    stratify_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "patient_id" not in full_df.columns:
        raise ValueError("benchmark split requires 'patient_id' column.")

    train_pool, test_split = train_test_split(
        full_df,
        test_size=float(test_ratio),
        random_state=int(stratify_seed),
        stratify=full_df["class_id"],
    )

    train_pool = train_pool.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)

    # Keep deterministic train/val split from train_pool for early stopping + calibration.
    val_ratio = float(max(0.01, min(0.5, val_ratio_within_train)))
    train_split, val_split = train_test_split(
        train_pool,
        test_size=val_ratio,
        random_state=int(stratify_seed),
        stratify=train_pool["class_id"],
    )

    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    train_ids = set(train_split["patient_id"].astype(str))
    val_ids = set(val_split["patient_id"].astype(str))
    test_ids = set(test_split["patient_id"].astype(str))
    if train_ids & test_ids:
        raise RuntimeError("train/test overlap detected in patient_id for two-stage stratified split.")
    if val_ids & test_ids:
        raise RuntimeError("val/test overlap detected in patient_id for two-stage stratified split.")

    return train_split, val_split, test_split, train_pool


def freeze_current_test_manifest(cfg: str | Path | dict[str, Any], seed: int = 1988) -> str:
    conf = _cfg(cfg)
    label_order = list(conf["data"]["label_order"])
    full_df = _load_dataset_pool(conf, label_order=label_order)
    _, _, test_split = _split_train_val_test(
        full_df=full_df,
        train_ratio=float(conf["data"]["train_ratio"]),
        test_ratio=float(conf["data"]["test_ratio"]),
        val_ratio=float(conf["data"]["val_ratio"]),
        stratify_seed=int(seed),
    )
    test_split = test_split.assign(split="test")
    dataset_tag = _profile_dataset_tag(conf)
    fixed_path = Path(conf["paths"]["manifests_dir"]) / f"fixed_test_{dataset_tag}_seed{int(seed)}.csv"
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    test_split.to_csv(fixed_path, index=False)
    return str(fixed_path.resolve())


def prepare_data_manifests(cfg: str | Path | dict[str, Any], seed: int = 1988) -> dict[str, str]:
    conf = _cfg(cfg)
    label_order = list(conf["data"]["label_order"])
    manifests = _manifest_outputs(conf, seed=seed)
    protocol = _data_protocol(conf)
    full_df = _load_dataset_pool(conf, label_order=label_order)

    if _is_benchmark(conf):
        test_ratio = float(conf.get("data", {}).get("profile_test_ratio", conf["data"].get("test_ratio", 0.15)))
        val_ratio_within_train = float(conf.get("data", {}).get("profile_val_ratio_within_train", conf["data"].get("val_ratio", 0.10)))
        split_seed = int(seed)

        train_split, val_split, test_split, train_pool = _split_two_stage_stratified_pool(
            full_df=full_df,
            test_ratio=test_ratio,
            val_ratio_within_train=val_ratio_within_train,
            stratify_seed=split_seed,
        )

        train_split = train_split.assign(split="train")
        val_split = val_split.assign(split="val")
        test_split = test_split.assign(split="test")
        all_full = pd.concat([train_pool.assign(split="train_pool"), test_split], ignore_index=True)

        all_full.to_csv(manifests["full_data"], index=False)
        test_split.to_csv(manifests["test_full"], index=False)
        train_split.to_csv(manifests["train_split"], index=False)
        val_split.to_csv(manifests["val_split"], index=False)

        n_total = len(all_full)
        summary = {
            "protocol": protocol,
            "seed": int(split_seed),
            "full_data_rows": int(n_total),
            "train_pool_rows": int(len(train_pool)),
            "train_split_rows": int(len(train_split)),
            "val_split_rows": int(len(val_split)),
            "test_full_rows": int(len(test_split)),
            "profile_test_ratio": float(test_ratio),
            "profile_val_ratio_within_train": float(val_ratio_within_train),
            "realized_train_ratio": float(len(train_split) / n_total) if n_total else 0.0,
            "realized_val_ratio": float(len(val_split) / n_total) if n_total else 0.0,
            "realized_test_ratio": float(len(test_split) / n_total) if n_total else 0.0,
            "train_val_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(val_split["patient_id"].astype(str)))),
            "train_test_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
            "val_test_disjoint": bool(set(val_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
            "source_counts": {str(k): int(v) for k, v in all_full["source_dataset"].astype(str).value_counts().to_dict().items()},
            "class_counts_train": {str(k): int(v) for k, v in train_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
            "class_counts_val": {str(k): int(v) for k, v in val_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
            "class_counts_test": {str(k): int(v) for k, v in test_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
        }
        _save_json(manifests["summary"], summary)
        return {
            "full_data": manifests["full_data"],
            "test_full": manifests["test_full"],
            "train_split": manifests["train_split"],
            "val_split": manifests["val_split"],
        }

    required_split_keys = ["train_ratio", "test_ratio", "val_ratio"]
    missing_split_keys = [k for k in required_split_keys if k not in conf.get("data", {})]
    if missing_split_keys:
        raise KeyError(
            f"Missing data split keys in config: {missing_split_keys}. "
            "Expected data.train_ratio, data.test_ratio, data.val_ratio."
        )

    train_split, val_split, test_split = _split_train_val_test(
        full_df=full_df,
        train_ratio=float(conf["data"]["train_ratio"]),
        test_ratio=float(conf["data"]["test_ratio"]),
        val_ratio=float(conf["data"]["val_ratio"]),
        stratify_seed=int(seed),
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)
    all_full = pd.concat([train_split, val_split, test_split], ignore_index=True)

    train_split = train_split.assign(split="train")
    val_split = val_split.assign(split="val")
    test_split = test_split.assign(split="test")

    all_full.to_csv(manifests["full_data"], index=False)
    test_split.to_csv(manifests["test_full"], index=False)
    train_split.to_csv(manifests["train_split"], index=False)
    val_split.to_csv(manifests["val_split"], index=False)

    n_total = len(all_full)
    realized_train = float(len(train_split) / n_total) if n_total else 0.0
    realized_val = float(len(val_split) / n_total) if n_total else 0.0
    realized_test = float(len(test_split) / n_total) if n_total else 0.0
    summary = {
        "protocol": protocol,
        "seed": int(seed),
        "full_data_rows": int(n_total),
        "train_split_rows": int(len(train_split)),
        "val_split_rows": int(len(val_split)),
        "test_full_rows": int(len(test_split)),
        "config_train_ratio": float(conf["data"]["train_ratio"]),
        "config_val_ratio": float(conf["data"]["val_ratio"]),
        "config_test_ratio": float(conf["data"]["test_ratio"]),
        "realized_train_ratio": realized_train,
        "realized_val_ratio": realized_val,
        "realized_test_ratio": realized_test,
        "train_val_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(val_split["patient_id"].astype(str)))),
        "train_test_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
        "val_test_disjoint": bool(set(val_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
        "source_counts": {str(k): int(v) for k, v in all_full["source_dataset"].astype(str).value_counts().to_dict().items()},
        "class_counts_train": {str(k): int(v) for k, v in train_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
        "class_counts_val": {str(k): int(v) for k, v in val_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
        "class_counts_test": {str(k): int(v) for k, v in test_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
    }
    _save_json(manifests["summary"], summary)

    return {
        "full_data": manifests["full_data"],
        "test_full": manifests["test_full"],
        "train_split": manifests["train_split"],
        "val_split": manifests["val_split"],
    }


def _build_transform(
    image_size: int,
    train: bool,
    aug_cfg: dict[str, float],
    preprocessing_cfg: dict[str, Any] | None = None,
    skip_resize: bool = False,
) -> T.Compose:
    prep = preprocessing_cfg or {}
    ops: list[Any] = []
    if train:
        if bool(aug_cfg.get("profile_style", False)):
            ops.append(T.RandomHorizontalFlip(p=float(aug_cfg.get("profile_horizontal_flip", 0.5))))
            ops.append(T.RandomVerticalFlip(p=float(aug_cfg.get("profile_vertical_flip", 0.0))))
            profile_rot = float(aug_cfg.get("profile_rotation_degrees", 20.0))
            if profile_rot > 0.0:
                ops.append(T.RandomRotation(degrees=profile_rot))
            profile_translate = float(aug_cfg.get("profile_translate", 0.03))
            profile_scale_min = float(aug_cfg.get("profile_scale_min", 0.95))
            profile_scale_max = float(aug_cfg.get("profile_scale_max", 1.05))
            profile_shear = float(aug_cfg.get("profile_shear_degrees", 8.0))
            ops.append(
                T.RandomAffine(
                    degrees=0.0,
                    translate=(profile_translate, profile_translate),
                    scale=(profile_scale_min, profile_scale_max),
                    shear=(-profile_shear, profile_shear),
                )
            )
            ops.append(
                T.ColorJitter(
                    brightness=float(aug_cfg.get("profile_brightness", 0.15)),
                    contrast=float(aug_cfg.get("profile_contrast", 0.15)),
                    saturation=float(aug_cfg.get("profile_saturation", 0.0)),
                    hue=float(aug_cfg.get("profile_hue", 0.0)),
                )
            )
        else:
            ops.append(T.RandomHorizontalFlip(p=float(aug_cfg.get("train_horizontal_flip", 0.5))))
            ops.append(T.RandomVerticalFlip(p=float(aug_cfg.get("train_vertical_flip", 0.5))))
            rot_deg = float(aug_cfg.get("train_rotation_degrees", 180.0))
            if rot_deg > 0.0:
                ops.append(T.RandomRotation(degrees=rot_deg))
            affine_translate = float(aug_cfg.get("train_affine_translate", 0.05))
            affine_scale_min = float(aug_cfg.get("train_affine_scale_min", 0.90))
            affine_scale_max = float(aug_cfg.get("train_affine_scale_max", 1.10))
            ops.append(
                T.RandomAffine(
                    degrees=0.0,
                    translate=(affine_translate, affine_translate),
                    scale=(affine_scale_min, affine_scale_max),
                )
            )
            ops.append(
                T.ColorJitter(
                    brightness=float(aug_cfg.get("color_jitter_brightness", 0.30)),
                    contrast=float(aug_cfg.get("color_jitter_contrast", 0.30)),
                    saturation=float(aug_cfg.get("color_jitter_saturation", 0.15)),
                    hue=float(aug_cfg.get("color_jitter_hue", 0.03)),
                )
            )

    if not skip_resize:
        ops.append(T.Resize((image_size, image_size)))
    norm_mean = prep.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std = prep.get("norm_std", [0.229, 0.224, 0.225])
    if not (isinstance(norm_mean, list) and len(norm_mean) == 3 and isinstance(norm_std, list) and len(norm_std) == 3):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[float(x) for x in norm_mean], std=[float(x) for x in norm_std]),
        ]
    )
    return T.Compose(ops)


def _apply_fundus_preprocessing(image: Image.Image, image_size: int, preprocessing_cfg: dict[str, Any] | None = None) -> Image.Image:
    cfg = preprocessing_cfg or {}
    if not bool(cfg.get("enabled", True)):
        return image

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # CLAHE on luminance for local contrast boost.
    clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
    grid = max(2, int(cfg.get("clahe_tile_grid", 8)))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l_chan = clahe.apply(l_chan)
    arr = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    # Ben Graham style local-mean subtraction.
    sigma = float(cfg.get("gaussian_sigma", max(1.0, image_size / 30.0)))
    weight = float(cfg.get("ben_graham_weight", 4.0))
    bias = float(cfg.get("ben_graham_bias", 128.0))
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    arr = cv2.addWeighted(arr, weight, blur, -weight, bias)

    # Circular crop to suppress outer border artifacts.
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    crop_ratio = float(cfg.get("circle_crop_ratio", 0.90))
    radius = max(1, int(min(h, w) * crop_ratio * 0.5))
    cv2.circle(mask, (w // 2, h // 2), radius, color=255, thickness=-1)
    arr_masked = np.zeros_like(arr)
    arr_masked[mask > 0] = arr[mask > 0]
    arr = arr_masked.astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


class _FundusDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        aug_cfg: dict[str, float],
        preprocessing_cfg: dict[str, Any] | None = None,
        train: bool = False,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.image_size = int(image_size)
        self.preprocessing_cfg = preprocessing_cfg or {}
        self.preprocessing_enabled = bool(self.preprocessing_cfg.get("enabled", True))
        self.transform = _build_transform(
            image_size=image_size,
            train=train,
            aug_cfg=aug_cfg,
            preprocessing_cfg=self.preprocessing_cfg,
            skip_resize=self.preprocessing_enabled,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = _apply_fundus_preprocessing(image, image_size=self.image_size, preprocessing_cfg=self.preprocessing_cfg)
        x = self.transform(image)
        return {
            "image": x,
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "laterality": str(row["laterality"]),
            "image_path": str(row["image_path"]),
        }


def _load_image_for_inference(
    image_path: str | Path,
    image_size: int,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> tuple[Image.Image, torch.Tensor]:
    cfg = preprocessing_cfg or {}
    preprocessing_enabled = bool(cfg.get("enabled", True))
    image = Image.open(image_path).convert("RGB")
    image = _apply_fundus_preprocessing(image, image_size=image_size, preprocessing_cfg=cfg)
    transform = _build_transform(
        image_size=image_size,
        train=False,
        aug_cfg={},
        preprocessing_cfg=cfg,
        skip_resize=preprocessing_enabled,
    )
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def _manifest_path(conf: dict[str, Any], split: str, seed: int | None = None) -> Path:
    paths = _manifest_outputs(conf, seed=seed)
    mapping = {
        "train": Path(paths["train_split"]),
        "val": Path(paths["val_split"]),
        "test": Path(paths["test_full"]),
    }
    if split not in mapping:
        raise ValueError(f"Unsupported split={split}")
    return mapping[split]


def _backbone_name(conf: dict[str, Any]) -> str:
    raw = str(conf.get("training", {}).get("backbone", "resnet50")).strip().lower()
    aliases = {
        "resnet50": "resnet50",
        "resnet-50": "resnet50",
        "efficientnet_b4": "efficientnet_b4",
        "efficientnet-b4": "efficientnet_b4",
        "efficientnetb4": "efficientnet_b4",
        "vgg16": "vgg16",
        "vgg-16": "vgg16",
        "vgg_16": "vgg16",
        "densenet121": "densenet121",
        "densenet-121": "densenet121",
        "densenet_121": "densenet121",
        "vit_b16": "vit_b16",
        "vit_b_16": "vit_b16",
        "vit-base": "vit_b16",
        "vit_base": "vit_b16",
        "vitb16": "vit_b16",
    }
    if raw in aliases:
        return aliases[raw]
    raise ValueError(
        f"Unsupported backbone: {raw}. Supported: resnet50, efficientnet_b4, "
        "vgg16, densenet121, vit_b16"
    )


def _model_image_size(conf: dict[str, Any]) -> int:
    requested = int(conf["data"]["image_size"])
    backbone = _backbone_name(conf)
    if backbone == "vit_b16":
        return 224
    return requested


# -----------------------------
# Model
# -----------------------------


class DRClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        use_pretrained: bool = True,
        dropout: float = 0.0,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        backbone_name = str(backbone).strip().lower()
        p_drop = float(np.clip(dropout, 0.0, 0.9))

        if backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
            self.net = resnet50(weights=weights)
            in_features = self.net.fc.in_features
            if p_drop > 0.0:
                self.net.fc = nn.Sequential(
                    nn.Dropout(p=p_drop),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.net.fc = nn.Linear(in_features, num_classes)
        elif backbone_name == "efficientnet_b4":
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.net = efficientnet_b4(weights=weights)
            in_features = int(self.net.classifier[-1].in_features)
            if p_drop > 0.0:
                self.net.classifier = nn.Sequential(
                    nn.Dropout(p=p_drop),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.net.classifier = nn.Linear(in_features, num_classes)
        elif backbone_name == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.net = vgg16(weights=weights)
            in_features = int(self.net.classifier[-1].in_features)
            if p_drop > 0.0:
                self.net.classifier[-1] = nn.Sequential(
                    nn.Dropout(p=p_drop),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.net.classifier[-1] = nn.Linear(in_features, num_classes)
        elif backbone_name == "densenet121":
            weights = DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.net = densenet121(weights=weights)
            in_features = int(self.net.classifier.in_features)
            if p_drop > 0.0:
                self.net.classifier = nn.Sequential(
                    nn.Dropout(p=p_drop),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.net.classifier = nn.Linear(in_features, num_classes)
        elif backbone_name == "vit_b16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.net = vit_b_16(weights=weights)
            in_features = self.net.heads.head.in_features
            if p_drop > 0.0:
                self.net.heads.head = nn.Sequential(
                    nn.Dropout(p=p_drop),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.net.heads.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LogitWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        # Keep backward hooks safe for attribution libraries that fail on view/in-place interactions.
        if isinstance(logits, torch.Tensor):
            return logits.clone()
        return logits


# -----------------------------
# Metrics + calibration
# -----------------------------


def _classification_metrics(df: pd.DataFrame, num_classes: int) -> tuple[dict[str, float], pd.DataFrame]:
    if len(df) == 0:
        empty_cm = pd.DataFrame(np.zeros((num_classes, num_classes), dtype=int))
        return {"n": 0.0}, empty_cm

    y_true = df["true_class"].astype(int).to_numpy()
    y_pred = df["pred_class"].astype(int).to_numpy()

    metrics = {
        "n": float(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }

    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    if set(prob_cols).issubset(df.columns):
        y_prob = df[prob_cols].to_numpy(dtype=float)
        y_onehot = np.eye(num_classes)[y_true]
        try:
            auc = float(roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        metrics["roc_auc_ovr_macro"] = auc

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)

    # Sensitivity is recall in one-vs-rest; we expose both for report readability.
    metrics["sensitivity_macro"] = float(np.mean(sensitivity))
    metrics["specificity_macro"] = float(np.mean(specificity))
    return metrics, pd.DataFrame(cm.astype(int))


def _per_class_metrics_from_confusion(cm_df: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    cm = cm_df.to_numpy(dtype=np.float64)
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = cm.sum(axis=1)

    rows: list[dict[str, Any]] = []
    for i in range(n_classes):
        label = label_order[i] if i < len(label_order) else str(i)
        rows.append(
            {
                "class_id": int(i),
                "class_name": label,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "sensitivity": float(recall[i]),
                "specificity": float(specificity[i]),
                "f1": float(f1[i]),
            }
        )
    return pd.DataFrame(rows)


def _build_reliability_table(df: pd.DataFrame, n_bins: int, interpolate_empty_bins: bool = False) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Cannot build reliability table from empty dataframe")

    use = df.copy()
    use["correct"] = (use["pred_class"].astype(int) == use["true_class"].astype(int)).astype(float)
    use["confidence"] = use["confidence"].astype(float).clip(0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        low, high = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            mask = (use["confidence"] >= low) & (use["confidence"] <= high)
        else:
            mask = (use["confidence"] >= low) & (use["confidence"] < high)

        sub = use[mask]
        if len(sub) == 0:
            acc = np.nan
            conf = np.nan
            gap = np.nan
        else:
            acc = float(sub["correct"].mean())
            conf = float(sub["confidence"].mean())
            gap = abs(acc - conf)

        rows.append(
            {
                "bin_index": i,
                "low": low,
                "high": high,
                "n": int(len(sub)),
                "bin_acc": float(acc) if not np.isnan(acc) else np.nan,
                "bin_conf": float(conf) if not np.isnan(conf) else np.nan,
                "calibration_gap": float(gap) if not np.isnan(gap) else np.nan,
            }
        )

    rel = pd.DataFrame(rows)
    if interpolate_empty_bins:
        if rel["calibration_gap"].notna().any():
            rel["calibration_gap"] = rel["calibration_gap"].interpolate(limit_direction="both")
        else:
            rel["calibration_gap"] = 0.0
        rel["calibration_gap"] = rel["calibration_gap"].fillna(0.0)
    return rel


def _expected_calibration_error(df: pd.DataFrame, n_bins: int) -> float:
    if len(df) == 0:
        return float("nan")
    use = df.copy()
    use["correct"] = (use["pred_class"].astype(int) == use["true_class"].astype(int)).astype(float)
    use["confidence"] = use["confidence"].astype(float).clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    n_total = float(len(use))
    ece = 0.0
    for i in range(int(n_bins)):
        low, high = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mask = (use["confidence"] >= low) & (use["confidence"] <= high)
        else:
            mask = (use["confidence"] >= low) & (use["confidence"] < high)
        sub = use[mask]
        if len(sub) == 0:
            continue
        acc = float(sub["correct"].mean())
        conf = float(sub["confidence"].mean())
        ece += (len(sub) / n_total) * abs(acc - conf)
    return float(ece)


def _border_mask(shape: tuple[int, int], border_ratio: float = 0.10) -> np.ndarray:
    h, w = shape
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return mask


def _retina_circle_mask(shape: tuple[int, int], radius_ratio: float = 0.45) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) * radius_ratio
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _attribution_mass_ratios(attr_map: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(attr_map, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError("attr_map must be 2D")

    total = float(arr.sum())
    if total <= 1e-12:
        return {"border_ratio": 0.0, "retina_ratio": 0.0}

    border = _border_mask(arr.shape, border_ratio=0.10)
    retina = _retina_circle_mask(arr.shape, radius_ratio=0.45)

    return {
        "border_ratio": float(arr[border].sum() / total),
        "retina_ratio": float(arr[retina].sum() / total),
    }


def _mask_by_score_map(image_tensor: torch.Tensor, score_map: np.ndarray, top_k_ratio: float, random_seed: int = 1988) -> tuple[torch.Tensor, torch.Tensor]:
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [1, C, H, W]")

    c, h, w = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"score_map must be 2D, got {arr.shape}")

    if arr.shape != (h, w):
        arr = np.array(
            Image.fromarray(arr.astype(np.float32), mode="F").resize((w, h), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    flat = arr.reshape(-1)
    k = max(1, int(round(flat.size * top_k_ratio)))

    top_idx = np.argsort(flat)[-k:]
    rng = np.random.default_rng(random_seed)
    rand_idx = rng.choice(flat.size, size=k, replace=False)

    top_mask = np.zeros(flat.size, dtype=bool)
    top_mask[top_idx] = True
    rand_mask = np.zeros(flat.size, dtype=bool)
    rand_mask[rand_idx] = True

    top_mask_2d = top_mask.reshape(h, w)
    rand_mask_2d = rand_mask.reshape(h, w)

    x_top = image_tensor.clone()
    x_rand = image_tensor.clone()

    top_mask_t = torch.tensor(top_mask_2d, dtype=torch.bool, device=image_tensor.device)
    rand_mask_t = torch.tensor(rand_mask_2d, dtype=torch.bool, device=image_tensor.device)

    for ch in range(c):
        x_top[0, ch][top_mask_t] = 0.0
        x_rand[0, ch][rand_mask_t] = 0.0

    return x_top, x_rand


@torch.no_grad()
def _faithfulness_delta(model: nn.Module, image_tensor: torch.Tensor, score_map: np.ndarray, pred_class: int, top_k_ratio: float = 0.20, random_seed: int = 1988) -> float:
    logits_base = model(image_tensor)
    probs_base = torch.softmax(logits_base, dim=1)
    base_prob = float(probs_base[0, pred_class].item())

    x_top, x_rand = _mask_by_score_map(image_tensor=image_tensor, score_map=score_map, top_k_ratio=top_k_ratio, random_seed=random_seed)
    probs_top = torch.softmax(model(x_top), dim=1)
    probs_rand = torch.softmax(model(x_rand), dim=1)

    drop_top = base_prob - float(probs_top[0, pred_class].item())
    drop_rand = base_prob - float(probs_rand[0, pred_class].item())
    return float(drop_top - drop_rand)


def _parse_faithfulness_k_list(raw_value: Any) -> list[float]:
    default_k = [0.05, 0.10, 0.20, 0.30]
    values: list[float] = []
    if isinstance(raw_value, (list, tuple)):
        for v in raw_value:
            try:
                k = float(v)
            except (TypeError, ValueError):
                continue
            if 0.0 < k < 1.0:
                values.append(k)
    if not values:
        values = default_k
    values = sorted({round(v, 4) for v in values})
    return values


def _k_to_col_name(k_ratio: float) -> str:
    return f"faith_delta_k{int(round(k_ratio * 100)):02d}"


def _faithfulness_multi_k(
    model: nn.Module,
    image_tensor: torch.Tensor,
    score_map: np.ndarray,
    pred_class: int,
    k_list: list[float],
    random_seed: int = 1988,
) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    for k in k_list:
        col = _k_to_col_name(k)
        out[col] = float(
            _faithfulness_delta(
                model=model,
                image_tensor=image_tensor,
                score_map=score_map,
                pred_class=pred_class,
                top_k_ratio=float(k),
                random_seed=random_seed,
            )
        )
    if out:
        aopc_delta = float(np.mean(list(out.values())))
    else:
        aopc_delta = float("nan")
    return out, aopc_delta


def _bootstrap_pass_rate_ci(pass_values: np.ndarray, n_boot: int = 1000, seed: int = 1988) -> tuple[float, float]:
    vals = np.asarray(pass_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan")
    n_boot = max(1, int(n_boot))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _xai_pass_rule_thresholds(conf: dict[str, Any]) -> tuple[float, float]:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    border_ratio_max = float(xai_cfg.get("pass_border_ratio_max", 0.35))
    faith_delta_min = float(xai_cfg.get("pass_faith_delta_min", 0.0))
    return border_ratio_max, faith_delta_min


def _xai_pass_flag(border_ratio: float, faith_delta: float, border_ratio_max: float, faith_delta_min: float) -> int:
    if np.isnan(border_ratio) or np.isnan(faith_delta):
        return 0
    return int((float(border_ratio) <= float(border_ratio_max)) and (float(faith_delta) > float(faith_delta_min)))


def _mcnemar_exact_pvalue(n01: int, n10: int) -> float:
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n <= 0:
        return float("nan")
    k = min(n01, n10)
    p_tail = 0.0
    for i in range(0, k + 1):
        p_tail += math.comb(n, i)
    p_two_sided = min(1.0, 2.0 * p_tail * (0.5**n))
    return float(p_two_sided)


def _mcnemar_chi2_approx(n01: int, n10: int, yates: bool = True) -> tuple[float, float]:
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n <= 0:
        return float("nan"), float("nan")

    diff = abs(n01 - n10)
    if yates:
        chi2 = ((max(diff - 1, 0)) ** 2) / n
    else:
        chi2 = (diff**2) / n

    # For chi-square(df=1), survival function is erfc(sqrt(x/2)).
    pval = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return float(chi2), float(pval)


def _build_xai_method_stats_row(
    method: str,
    df: pd.DataFrame,
    pass_col: str,
    n_boot: int,
    seed: int,
    border_ratio_max: float,
    faith_delta_min: float,
    evaluation_scope: str = "",
    status_note: str = "",
) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "method": method,
            "n": 0,
            "pass_rate": float("nan"),
            "pass_rate_ci95_low": float("nan"),
            "pass_rate_ci95_high": float("nan"),
            "mean_border_ratio": float("nan"),
            "mean_retina_ratio": float("nan"),
            "mean_faith_delta_k20": float("nan"),
            "mean_aopc_delta": float("nan"),
            "aopc_pass_rate": float("nan"),
            "pass_rule_border_ratio_max": float(border_ratio_max),
            "pass_rule_faith_delta_k20_min": float(faith_delta_min),
            "evaluation_scope": str(evaluation_scope),
            "status_note": status_note or "no_rows",
        }

    pass_vals = df[pass_col].astype(float).to_numpy()
    ci_low, ci_high = _bootstrap_pass_rate_ci(pass_vals, n_boot=n_boot, seed=seed)
    if "faith_delta_k20" in df.columns:
        faith_k20 = pd.to_numeric(df["faith_delta_k20"], errors="coerce")
    else:
        faith_k20 = pd.to_numeric(df.get("faithfulness_delta", np.nan), errors="coerce")

    if "aopc_pass" in df.columns:
        aopc_pass_rate = float(pd.to_numeric(df["aopc_pass"], errors="coerce").mean())
    else:
        aopc_pass_rate = float("nan")

    pass_rate_value = float(np.mean(pass_vals))
    return {
        "method": method,
        "n": int(len(df)),
        "pass_rate": float(pass_rate_value),
        "pass_rate_ci95_low": float(ci_low),
        "pass_rate_ci95_high": float(ci_high),
        "mean_border_ratio": float(pd.to_numeric(df["border_ratio"], errors="coerce").mean()),
        "mean_retina_ratio": float(pd.to_numeric(df["retina_ratio"], errors="coerce").mean()),
        "mean_faith_delta_k20": float(faith_k20.mean()),
        "mean_aopc_delta": float(pd.to_numeric(df.get("aopc_delta", np.nan), errors="coerce").mean()),
        "aopc_pass_rate": float(aopc_pass_rate),
        "pass_rule_border_ratio_max": float(border_ratio_max),
        "pass_rule_faith_delta_k20_min": float(faith_delta_min),
        "evaluation_scope": str(evaluation_scope),
        "status_note": status_note,
    }


def _build_xai_pass_by_correctness_table(method: str, df: pd.DataFrame, pass_col: str) -> pd.DataFrame:
    out_cols = [
        "method",
        "group",
        "n",
        "pass_rate",
        "mean_border_ratio",
        "mean_faith_delta_k20",
    ]
    if len(df) == 0 or pass_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    work = df.copy()
    if "faith_delta_k20" in work.columns:
        work["_faith_k20"] = pd.to_numeric(work["faith_delta_k20"], errors="coerce")
    else:
        work["_faith_k20"] = pd.to_numeric(work.get("faithfulness_delta", np.nan), errors="coerce")
    work["_pass"] = pd.to_numeric(work[pass_col], errors="coerce")
    work["_border"] = pd.to_numeric(work.get("border_ratio", np.nan), errors="coerce")

    groups: list[tuple[str, pd.DataFrame]] = [("all", work)]
    if {"target_class", "pred_class"}.issubset(work.columns):
        truth = pd.to_numeric(work["target_class"], errors="coerce")
        pred = pd.to_numeric(work["pred_class"], errors="coerce")
        is_correct = truth == pred
        groups.append(("correct", work[is_correct]))
        groups.append(("wrong", work[~is_correct]))

    rows: list[dict[str, Any]] = []
    for group_name, part in groups:
        if len(part) == 0:
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "n": 0,
                    "pass_rate": float("nan"),
                    "mean_border_ratio": float("nan"),
                    "mean_faith_delta_k20": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "method": method,
                "group": group_name,
                "n": int(len(part)),
                "pass_rate": float(part["_pass"].mean()),
                "mean_border_ratio": float(part["_border"].mean()),
                "mean_faith_delta_k20": float(part["_faith_k20"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


def _build_xai_pass_by_class_table(
    method: str,
    df: pd.DataFrame,
    pass_col: str,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    out_cols = ["method", "class_id", "class_name", "n", "pass_rate"]
    if len(df) == 0 or pass_col not in df.columns or "target_class" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    work = df.copy()
    work["target_class"] = pd.to_numeric(work["target_class"], errors="coerce")
    work["_pass"] = pd.to_numeric(work[pass_col], errors="coerce")
    grouped = (
        work.dropna(subset=["target_class"])
        .groupby("target_class", dropna=False)["_pass"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"target_class": "class_id", "mean": "pass_rate", "count": "n"})
    )
    grouped["class_id"] = grouped["class_id"].astype(int)
    if class_names:
        grouped["class_name"] = grouped["class_id"].apply(
            lambda cid: class_names[cid] if 0 <= int(cid) < len(class_names) else str(cid)
        )
    else:
        grouped["class_name"] = grouped["class_id"].astype(str)
    grouped["method"] = method
    return grouped[out_cols]


def _build_xai_pairwise_stats(rq1_df: pd.DataFrame, rq2_df: pd.DataFrame) -> pd.DataFrame:
    left_cols = ["sample_id", "gradcam_pass"]
    right_cols = ["sample_id", "shap_pass"]
    if not set(left_cols).issubset(rq1_df.columns) or not set(right_cols).issubset(rq2_df.columns):
        return pd.DataFrame(
            [
                {
                    "n_paired": 0,
                    "gradcam_pass_rate": float("nan"),
                    "shap_pass_rate": float("nan"),
                    "delta_pass_rate": float("nan"),
                    "n00": 0,
                    "n01": 0,
                    "n10": 0,
                    "n11": 0,
                    "mcnemar_chi2": float("nan"),
                    "mcnemar_pvalue_chi2": float("nan"),
                    "mcnemar_pvalue_exact": float("nan"),
                    "status_note": "missing_pass_columns",
                }
            ]
        )

    pair = rq1_df[left_cols].merge(rq2_df[right_cols], on="sample_id", how="inner")
    if len(pair) == 0:
        return pd.DataFrame(
            [
                {
                    "n_paired": 0,
                    "gradcam_pass_rate": float("nan"),
                    "shap_pass_rate": float("nan"),
                    "delta_pass_rate": float("nan"),
                    "n00": 0,
                    "n01": 0,
                    "n10": 0,
                    "n11": 0,
                    "mcnemar_chi2": float("nan"),
                    "mcnemar_pvalue_chi2": float("nan"),
                    "mcnemar_pvalue_exact": float("nan"),
                    "status_note": "no_paired_samples",
                }
            ]
        )

    g = pair["gradcam_pass"].astype(int).to_numpy()
    s = pair["shap_pass"].astype(int).to_numpy()
    n00 = int(np.sum((g == 0) & (s == 0)))
    n01 = int(np.sum((g == 0) & (s == 1)))
    n10 = int(np.sum((g == 1) & (s == 0)))
    n11 = int(np.sum((g == 1) & (s == 1)))

    status_note = "ok"
    pvalue = float("nan")
    chi2 = float("nan")
    pvalue_chi2 = float("nan")
    if (n01 + n10) == 0:
        status_note = "no_discordant_pairs"
    else:
        pvalue = _mcnemar_exact_pvalue(n01=n01, n10=n10)
        chi2, pvalue_chi2 = _mcnemar_chi2_approx(n01=n01, n10=n10, yates=True)

    grad_rate = float(np.mean(g))
    shap_rate = float(np.mean(s))
    return pd.DataFrame(
        [
            {
                "n_paired": int(len(pair)),
                "gradcam_pass_rate": grad_rate,
                "shap_pass_rate": shap_rate,
                "delta_pass_rate": float(shap_rate - grad_rate),
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "mcnemar_chi2": chi2,
                "mcnemar_pvalue_chi2": pvalue_chi2,
                "mcnemar_pvalue_exact": pvalue,
                "status_note": status_note,
            }
        ]
    )


# -----------------------------
# Prediction + calibration
# -----------------------------


def _to_ratio_fraction(value: Any, key: str) -> float:
    ratio = float(value)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if ratio <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return ratio


def _split_policy_tag(conf: dict[str, Any]) -> str:
    data_cfg = conf.get("data", {})
    if _is_benchmark(conf):
        del data_cfg
        return _profile_profile_tag(conf)
    train_ratio = _to_ratio_fraction(data_cfg.get("train_ratio", 0.75), "train_ratio")
    test_ratio = _to_ratio_fraction(data_cfg.get("test_ratio", 0.15), "test_ratio")
    val_ratio = _to_ratio_fraction(data_cfg.get("val_ratio", 0.10), "val_ratio")
    return f"{int(round(train_ratio * 100))}-{int(round(test_ratio * 100))}-{int(round(val_ratio * 100))}"


def _latest_run_record_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"latest_run_seed{seed}.json"


def _legacy_checkpoint_alias_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"classifier_seed{seed}.pt"


def _legacy_calibration_alias_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"classifier_seed{seed}_calibration.json"


def _legacy_predictions_alias_path(conf: dict[str, Any], seed: int, split: str) -> Path:
    return Path(conf["paths"]["predictions_dir"]) / f"predictions_seed{seed}_{split}.csv"


def _legacy_train_history_alias_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"train_history_seed{seed}.json"


def _legacy_gradcam_status_alias_path(conf: dict[str, Any], seed: int, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"gradcam_status_seed{seed}_{split}.json"


def _legacy_shap_status_alias_path(conf: dict[str, Any], seed: int, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"shap_status_seed{seed}_{split}.json"


def _legacy_run_log_alias_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"run_complete_workflow_seed{seed}.json"


def _new_run_id(conf: dict[str, Any], seed: int) -> str:
    backbone = _backbone_name(conf)
    split_policy = _split_policy_tag(conf)
    timestamp_visible = datetime.now().strftime("%Y%m%dT%H%M")
    timestamp_entropy = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    signature_payload = {
        "seed": int(seed),
        "backbone": backbone,
        "split_policy": split_policy,
        "config_signature": _checkpoint_config_signature(conf),
    }
    signature_text = json.dumps(signature_payload, sort_keys=True)
    short_hash = hashlib.sha1(f"{signature_text}|{timestamp_entropy}".encode("utf-8")).hexdigest()[:6]
    return f"dr_{backbone}_{split_policy}_seed{seed}_{timestamp_visible}_{short_hash}"


def _run_id_from_checkpoint_path(conf: dict[str, Any], seed: int, checkpoint_path: str | Path) -> str:
    ckpt = Path(checkpoint_path)
    stem = ckpt.stem
    if stem.startswith("dr_"):
        return stem
    backbone = _backbone_name(conf)
    split_policy = _split_policy_tag(conf)
    if ckpt.exists():
        ts = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y%m%dT%H%M")
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M")
    short_hash = hashlib.sha1(f"{stem}|{seed}|{backbone}|{split_policy}".encode("utf-8")).hexdigest()[:6]
    return f"dr_{backbone}_{split_policy}_seed{seed}_{ts}_{short_hash}"


def _checkpoint_path_for_run_id(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"{run_id}.pt"


def _calibration_path_for_run_id(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"{run_id}_calibration.json"


def _predictions_path_for_run_id(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["predictions_dir"]) / f"{run_id}_{split}_predictions.csv"


def _train_history_log_path(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_train_history.json"


def _gradcam_status_log_path(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_{split}_gradcam_status.json"


def _shap_status_log_path(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_{split}_shap_status.json"


def _workflow_run_log_path(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_run_complete_workflow.json"


def _table_path(conf: dict[str, Any], stem: str, seed: int | None = None, split: str | None = None, ext: str = "csv") -> Path:
    base = Path(conf["paths"]["tables_dir"])
    if _is_benchmark(conf):
        parts = [str(stem)]
        if seed is not None:
            parts.append(f"seed{int(seed)}")
        if split:
            parts.append(str(split))
        parts.append(_profile_profile_tag(conf))
        return base / f"{'_'.join(parts)}.{ext}"

    seed_part = f"_seed{int(seed)}" if seed is not None else ""
    split_part = f"_{split}" if split else ""
    return base / f"{stem}{seed_part}{split_part}.{ext}"


def _save_latest_run_record(
    conf: dict[str, Any],
    seed: int,
    run_id: str,
    checkpoint_path: str | Path,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    base_payload = {
        "seed": int(seed),
        "run_id": run_id,
        "backbone": _backbone_name(conf),
        "split_policy": _split_policy_tag(conf),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    existing_payload: dict[str, Any] = {}
    latest_path = _latest_run_record_path(conf, seed)
    if latest_path.exists():
        try:
            loaded = _load_json(latest_path)
            if isinstance(loaded, dict):
                existing_payload = loaded
        except Exception:
            existing_payload = {}

    if str(existing_payload.get("run_id", "")) == str(run_id):
        payload = dict(existing_payload)
        payload.update(base_payload)
    else:
        payload = base_payload

    if extra_fields:
        payload.update(extra_fields)
    _save_json(latest_path, payload)


def _load_latest_run_record(conf: dict[str, Any], seed: int) -> dict[str, Any] | None:
    p = _latest_run_record_path(conf, seed)
    if not p.exists():
        return None
    try:
        payload = _load_json(p)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    ckpt = payload.get("checkpoint_path")
    run_id = payload.get("run_id")
    if not ckpt or not run_id:
        return None
    return payload


def _latest_named_checkpoint(conf: dict[str, Any], seed: int) -> Path | None:
    ckpt_dir = Path(conf["paths"]["checkpoints_dir"])
    pattern = f"dr_{_backbone_name(conf)}_{_split_policy_tag(conf)}_seed{seed}_*.pt"
    matches = [p for p in ckpt_dir.glob(pattern) if p.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _resolve_checkpoint_and_run_id(
    conf: dict[str, Any],
    seed: int,
    checkpoint: str | Path | None = None,
    require_existing: bool = True,
) -> tuple[Path, str]:
    if checkpoint is not None:
        ckpt = Path(checkpoint).expanduser().resolve()
        if require_existing and not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
        _save_latest_run_record(conf, seed, run_id, ckpt)
        return ckpt, run_id

    record = _load_latest_run_record(conf, seed)
    if record is not None:
        ckpt = Path(str(record["checkpoint_path"]))
        run_id = str(record["run_id"])
        if ckpt.exists():
            if not ckpt.stem.startswith("dr_"):
                migrated_ckpt = _checkpoint_path_for_run_id(conf, run_id)
                _write_alias_copy(ckpt, migrated_ckpt)
                _save_latest_run_record(conf, seed, run_id, migrated_ckpt)
                return migrated_ckpt, run_id
            return ckpt, run_id

    ckpt = _latest_named_checkpoint(conf, seed)
    if ckpt is not None:
        run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
        _save_latest_run_record(conf, seed, run_id, ckpt)
        return ckpt, run_id

    if _use_legacy_aliases(conf):
        legacy_ckpt = _legacy_checkpoint_alias_path(conf, seed)
        if legacy_ckpt.exists():
            run_id = _run_id_from_checkpoint_path(conf, seed, legacy_ckpt)
            migrated_ckpt = _checkpoint_path_for_run_id(conf, run_id)
            _write_alias_copy(legacy_ckpt, migrated_ckpt)
            _save_latest_run_record(conf, seed, run_id, migrated_ckpt)
            return migrated_ckpt, run_id

    if require_existing:
        raise FileNotFoundError(
            f"No checkpoint found for seed={seed} under {Path(conf['paths']['checkpoints_dir'])}"
        )

    run_id = _new_run_id(conf, seed)
    return _checkpoint_path_for_run_id(conf, run_id), run_id


def _find_matching_checkpoint_by_signature(
    conf: dict[str, Any],
    seed: int,
    config_signature: dict[str, Any],
) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    latest_named = _latest_named_checkpoint(conf, seed)
    if latest_named is not None:
        ckpt_dir = Path(conf["paths"]["checkpoints_dir"])
        pattern = f"dr_{_backbone_name(conf)}_{_split_policy_tag(conf)}_seed{seed}_*.pt"
        named = [p for p in ckpt_dir.glob(pattern) if p.is_file()]
        named.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(named)

    if _use_legacy_aliases(conf):
        legacy = _legacy_checkpoint_alias_path(conf, seed)
        if legacy.exists():
            candidates.append(legacy)

    seen: set[str] = set()
    for ckpt in candidates:
        resolved = str(ckpt.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = torch.load(ckpt, map_location="cpu")
            saved_sig = payload.get("config_signature")
        except Exception:
            continue
        if saved_sig == config_signature:
            run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
            return ckpt, run_id
    return None


def _checkpoint_path(conf: dict[str, Any], seed: int) -> Path:
    ckpt, _ = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    return ckpt


def _calibration_path(conf: dict[str, Any], seed: int) -> Path:
    _, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    return _calibration_path_for_run_id(conf, run_id)


def _predictions_path(conf: dict[str, Any], seed: int, split: str) -> Path:
    _, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    return _predictions_path_for_run_id(conf, run_id, split)


def _run_id_from_predictions_path(predictions_path: str | Path, split: str) -> str:
    stem = Path(predictions_path).stem
    suffix = f"_{split}_predictions"
    if stem.startswith("dr_") and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return ""


def _checkpoint_config_signature(conf: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_protocol": _data_protocol(conf),
        "data_source": str(conf.get("data", {}).get("source", "aptos_only")),
        "profile_test_ratio": float(conf.get("data", {}).get("profile_test_ratio", conf.get("data", {}).get("test_ratio", 0.15))),
        "profile_val_ratio_within_train": float(
            conf.get("data", {}).get("profile_val_ratio_within_train", conf.get("data", {}).get("val_ratio", 0.10))
        ),
        "backbone": _backbone_name(conf),
        "image_size": int(conf["data"]["image_size"]),
        "effective_image_size": _model_image_size(conf),
        "num_classes": int(conf["data"]["num_classes"]),
        "use_pretrained": bool(conf["training"].get("use_pretrained", True)),
        "dropout": float(conf["training"].get("dropout", 0.0)),
        "label_smoothing": float(conf["training"].get("label_smoothing", 0.0)),
        "optimizer_name": str(conf["training"].get("optimizer_name", "adamw")),
        "lr": float(conf["training"]["lr"]),
        "weight_decay": float(conf["training"]["weight_decay"]),
        "use_weighted_sampler": bool(conf["training"].get("use_weighted_sampler", True)),
        "use_class_weights": bool(conf["training"].get("use_class_weights", True)),
        "loss_name": str(conf["training"].get("loss_name", "auto")),
        "use_focal_loss": bool(conf["training"].get("use_focal_loss", False)),
        "focal_gamma": float(conf["training"].get("focal_gamma", 2.0)),
        "use_ordinal_loss": bool(conf["training"].get("use_ordinal_loss", True)),
        "ordinal_loss_weight": float(conf["training"].get("ordinal_loss_weight", 0.3)),
        "grad_accum_steps": int(conf["training"].get("grad_accum_steps", 1)),
        "profile_class_balanced_sampling": bool(conf["training"].get("profile_class_balanced_sampling", False)),
        "profile_target_per_class": int(conf.get("augmentation", {}).get("profile_target_per_class", 0)),
        "use_differential_lr": bool(conf["training"].get("use_differential_lr", True)),
        "backbone_lr_multiplier": float(conf["training"].get("backbone_lr_multiplier", 0.01)),
        "layer4_lr_multiplier": float(conf["training"].get("layer4_lr_multiplier", 0.1)),
        "head_lr_multiplier": float(conf["training"].get("head_lr_multiplier", 10.0)),
        "scheduler_name": str(conf["training"].get("scheduler_name", "reduce_on_plateau")),
        "use_scheduler": bool(conf["training"].get("use_scheduler", True)),
        "scheduler_factor": float(conf["training"].get("scheduler_factor", 0.5)),
        "scheduler_patience": int(conf["training"].get("scheduler_patience", 3)),
        "scheduler_min_lr": float(conf["training"].get("scheduler_min_lr", 1e-8)),
        "scheduler_min_lr_ratio": float(conf["training"].get("scheduler_min_lr_ratio", 0.01)),
        "augmentation": dict(conf.get("augmentation", {})),
        "preprocessing": dict(conf.get("preprocessing", {})),
    }


def _class_weights(class_ids: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(class_ids), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _ordinal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ordinal_weight: float = 0.3,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    probs = torch.softmax(logits, dim=1)
    class_idx = torch.arange(num_classes, device=logits.device, dtype=probs.dtype)
    target_f = targets.to(dtype=probs.dtype).unsqueeze(1)
    dist_sq = (class_idx.unsqueeze(0) - target_f) ** 2
    scale = float(max(1, (num_classes - 1) ** 2))
    ordinal_penalty = (probs * (dist_sq / scale)).sum(dim=1).mean()
    return ce + float(ordinal_weight) * ordinal_penalty


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()


@torch.no_grad()
def _predict_manifest(
    model: DRClassifier,
    manifest_path: str | Path,
    conf: dict[str, Any],
    split: str,
    device: torch.device,
    temperature: float = 1.0,
) -> pd.DataFrame:
    image_size = _model_image_size(conf)
    ds = _FundusDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )

    model.eval()
    rows: list[dict[str, Any]] = []
    temp = max(1e-4, float(temperature))

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy().tolist()
        sample_ids = list(batch["sample_id"])
        laterality = list(batch["laterality"])
        image_paths = list(batch["image_path"])

        logits = model(images)
        logits = logits / temp
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        probs_np = probs.detach().cpu().numpy()
        conf_np = confs.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()

        for i, sid in enumerate(sample_ids):
            row = {
                "sample_id": sid,
                "true_class": int(labels[i]),
                "pred_class": int(pred_np[i]),
                "confidence": float(conf_np[i]),
                "laterality": laterality[i],
                "split": split,
                "image_path": image_paths[i],
            }
            for c in range(probs_np.shape[1]):
                row[f"prob_{c}"] = float(probs_np[i, c])
            rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def _collect_logits_and_labels(
    model: DRClassifier,
    manifest_path: str | Path,
    conf: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = _model_image_size(conf)
    ds = _FundusDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())

    if not logits_list:
        return torch.empty((0, int(conf["data"]["num_classes"])), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] == 0:
        return 1.0

    logits_f = logits.float()
    labels_f = labels.long()
    criterion = torch.nn.CrossEntropyLoss()
    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        t = torch.clamp(temperature, min=0.5, max=10.0)
        loss = criterion(logits_f / t, labels_f)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(torch.clamp(temperature.detach(), min=0.5, max=10.0).item())


def _load_model(
    conf: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> DRClassifier:
    ckpt, _ = _resolve_checkpoint_and_run_id(
        conf,
        seed=seed,
        checkpoint=checkpoint,
        require_existing=True,
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = DRClassifier(
        num_classes=int(conf["data"]["num_classes"]),
        use_pretrained=False,
        dropout=float(conf["training"].get("dropout", 0.0)),
        backbone=_backbone_name(conf),
    ).to(device)

    payload = torch.load(ckpt, map_location=device)
    state_dict = dict(payload["state_dict"])
    target_state = model.state_dict()

    backbone = _backbone_name(conf)
    if backbone == "resnet50":
        # Backward/forward compatibility for checkpoints saved before/after dropout head.
        if "net.fc.weight" in state_dict and "net.fc.1.weight" in target_state:
            state_dict["net.fc.1.weight"] = state_dict.pop("net.fc.weight")
        if "net.fc.bias" in state_dict and "net.fc.1.bias" in target_state:
            state_dict["net.fc.1.bias"] = state_dict.pop("net.fc.bias")
        if "net.fc.1.weight" in state_dict and "net.fc.weight" in target_state:
            state_dict["net.fc.weight"] = state_dict.pop("net.fc.1.weight")
        if "net.fc.1.bias" in state_dict and "net.fc.bias" in target_state:
            state_dict["net.fc.bias"] = state_dict.pop("net.fc.1.bias")
    elif backbone == "efficientnet_b4":
        # Backward/forward compatibility for checkpoints saved before/after dropout head.
        if "net.classifier.weight" in state_dict and "net.classifier.1.weight" in target_state:
            state_dict["net.classifier.1.weight"] = state_dict.pop("net.classifier.weight")
        if "net.classifier.bias" in state_dict and "net.classifier.1.bias" in target_state:
            state_dict["net.classifier.1.bias"] = state_dict.pop("net.classifier.bias")
        if "net.classifier.1.weight" in state_dict and "net.classifier.weight" in target_state:
            state_dict["net.classifier.weight"] = state_dict.pop("net.classifier.1.weight")
        if "net.classifier.1.bias" in state_dict and "net.classifier.bias" in target_state:
            state_dict["net.classifier.bias"] = state_dict.pop("net.classifier.1.bias")
    elif backbone == "vit_b16":
        if "net.heads.head.weight" in state_dict and "net.heads.head.1.weight" in target_state:
            state_dict["net.heads.head.1.weight"] = state_dict.pop("net.heads.head.weight")
        if "net.heads.head.bias" in state_dict and "net.heads.head.1.bias" in target_state:
            state_dict["net.heads.head.1.bias"] = state_dict.pop("net.heads.head.bias")
        if "net.heads.head.1.weight" in state_dict and "net.heads.head.weight" in target_state:
            state_dict["net.heads.head.weight"] = state_dict.pop("net.heads.head.1.weight")
        if "net.heads.head.1.bias" in state_dict and "net.heads.head.bias" in target_state:
            state_dict["net.heads.head.bias"] = state_dict.pop("net.heads.head.1.bias")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def train_dr_classifier(
    cfg: str | Path | dict[str, Any],
    seed: int = 1988,
    manifests: dict[str, str] | None = None,
    reuse_if_exists: bool = True,
) -> str:
    conf = _cfg(cfg)
    _set_seed(int(seed))
    current_sig = _checkpoint_config_signature(conf)

    if reuse_if_exists:
        matched = _find_matching_checkpoint_by_signature(conf, seed=seed, config_signature=current_sig)
        if matched is not None:
            existing_ckpt, existing_run_id = matched
            if not existing_ckpt.stem.startswith("dr_"):
                migrated_ckpt = _checkpoint_path_for_run_id(conf, existing_run_id)
                _write_alias_copy(existing_ckpt, migrated_ckpt)
                existing_ckpt = migrated_ckpt
            _save_latest_run_record(conf, seed=seed, run_id=existing_run_id, checkpoint_path=existing_ckpt)
            if _use_legacy_aliases(conf):
                _write_alias_copy(existing_ckpt, _legacy_checkpoint_alias_path(conf, seed))
            print(f"Reusing existing checkpoint: {existing_ckpt}")
            return str(existing_ckpt)

    # No exact signature match: always start a fresh run ID/path to avoid
    # overwriting a previous checkpoint from a different config signature.
    run_id = _new_run_id(conf, seed=seed)
    ckpt = _checkpoint_path_for_run_id(conf, run_id)
    train_started_at = datetime.now().isoformat(timespec="seconds")
    train_start_perf = time.perf_counter()
    _save_latest_run_record(
        conf,
        seed=seed,
        run_id=run_id,
        checkpoint_path=ckpt,
        extra_fields={
            "status": "training_started",
            "train_started_at": train_started_at,
        },
    )

    if manifests is None:
        train_manifest = _manifest_path(conf, "train", seed=seed)
        val_manifest = _manifest_path(conf, "val", seed=seed)
        if not train_manifest.exists() or not val_manifest.exists():
            manifests = prepare_data_manifests(conf, seed=seed)
            train_manifest = Path(manifests["train_split"])
            val_manifest = Path(manifests["val_split"])
    else:
        train_manifest = Path(manifests["train_split"])
        val_manifest = Path(manifests["val_split"])

    device = _resolve_device(conf["training"].get("device", "auto"))
    image_size = _model_image_size(conf)

    train_df = pd.read_csv(train_manifest)
    class_w = _class_weights(
        class_ids=train_df["class_id"].astype(int).tolist(),
        num_classes=int(conf["data"]["num_classes"]),
        device=device,
    )

    train_ds = _FundusDataset(
        manifest_path=train_manifest,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=True,
    )
    val_ds = _FundusDataset(
        manifest_path=val_manifest,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )

    num_classes = int(conf["data"]["num_classes"])
    class_ids = train_df["class_id"].astype(int).to_numpy()
    counts = np.bincount(class_ids, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    sample_weights_np = (1.0 / counts[class_ids]).astype(np.float32)

    use_weighted_sampler = bool(conf["training"].get("use_weighted_sampler", True))
    profile_class_balanced_sampling = bool(conf["training"].get("profile_class_balanced_sampling", False)) and _is_benchmark(conf)
    profile_target_per_class = int(conf.get("augmentation", {}).get("profile_target_per_class", 0))
    sampler = None
    if use_weighted_sampler:
        sample_weights = torch.tensor(sample_weights_np, dtype=torch.double)
        if profile_class_balanced_sampling and profile_target_per_class > 0:
            num_samples = int(max(1, profile_target_per_class) * max(1, num_classes))
        else:
            num_samples = int(len(sample_weights_np))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False if sampler is not None else True,
        sampler=sampler,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )

    model = DRClassifier(
        num_classes=int(conf["data"]["num_classes"]),
        use_pretrained=bool(conf["training"].get("use_pretrained", True)),
        dropout=float(conf["training"].get("dropout", 0.0)),
        backbone=_backbone_name(conf),
    ).to(device)

    backbone_name = _backbone_name(conf)
    base_lr = float(conf["training"]["lr"])
    wd = float(conf["training"]["weight_decay"])
    optimizer_name = str(conf["training"].get("optimizer_name", "adamw")).strip().lower()
    if optimizer_name not in {"adamw", "adam"}:
        raise ValueError(f"Unsupported optimizer_name={optimizer_name}. Use 'adamw' or 'adam'.")
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    requested_differential_lr = bool(conf["training"].get("use_differential_lr", True))
    use_differential_lr = requested_differential_lr and backbone_name in {
        "resnet50", "vit_b16", "efficientnet_b4", "vgg16", "densenet121"
    }
    if requested_differential_lr and not use_differential_lr:
        print(f"Differential LR is not supported for backbone={backbone_name}; falling back to single LR.")

    if use_differential_lr:
        backbone_mult = float(conf["training"].get("backbone_lr_multiplier", 0.01))
        layer4_mult = float(conf["training"].get("layer4_lr_multiplier", 0.1))
        head_mult = float(conf["training"].get("head_lr_multiplier", 10.0))

        param_groups: list[dict[str, Any]] = []
        if backbone_name == "resnet50":
            layer4_params = list(model.net.layer4.parameters())
            head_params = list(model.net.fc.parameters())
            excluded = {id(p) for p in layer4_params + head_params}
            backbone_params = [p for p in model.net.parameters() if id(p) not in excluded]
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": base_lr * backbone_mult})
            if layer4_params:
                param_groups.append({"params": layer4_params, "lr": base_lr * layer4_mult})
            if head_params:
                param_groups.append({"params": head_params, "lr": base_lr * head_mult})
        else:
            if backbone_name == "vit_b16":
                head_module = model.net.heads
            else:
                head_module = model.net.classifier
            head_params = list(head_module.parameters())
            excluded = {id(p) for p in head_params}
            backbone_params = [p for p in model.net.parameters() if id(p) not in excluded]
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": base_lr * backbone_mult})
            if head_params:
                param_groups.append({"params": head_params, "lr": base_lr * head_mult})

        if not param_groups:
            param_groups = [{"params": model.parameters(), "lr": base_lr}]
        optimizer = optimizer_cls(param_groups, weight_decay=wd)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=base_lr, weight_decay=wd)

    use_class_weights = bool(conf["training"].get("use_class_weights", True))
    class_weights_for_loss = class_w if use_class_weights else None
    label_smoothing = float(conf["training"].get("label_smoothing", 0.0))
    loss_name = str(conf["training"].get("loss_name", "auto")).strip().lower()
    use_focal_loss = bool(conf["training"].get("use_focal_loss", False))
    use_ordinal_loss = bool(conf["training"].get("use_ordinal_loss", True))
    if loss_name == "focal":
        use_focal_loss = True
        use_ordinal_loss = False
    elif loss_name == "ordinal":
        use_ordinal_loss = True
        use_focal_loss = False
    elif loss_name == "cross_entropy":
        use_ordinal_loss = False
        use_focal_loss = False
    elif loss_name == "auto":
        # Backward-compatible behavior: ordinal loss if enabled, otherwise CE.
        use_focal_loss = bool(conf["training"].get("use_focal_loss", False))

    focal_gamma = float(conf["training"].get("focal_gamma", 2.0))
    ordinal_loss_weight = float(conf["training"].get("ordinal_loss_weight", 0.3))
    grad_accum_steps = int(max(1, conf["training"].get("grad_accum_steps", 1)))
    max_epochs = int(conf["training"]["epochs"])
    patience_limit = int(conf["training"]["early_stopping_patience"])

    def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if use_focal_loss:
            return _focal_loss(
                logits=logits,
                targets=targets,
                gamma=focal_gamma,
                class_weights=class_weights_for_loss,
                label_smoothing=label_smoothing,
            )
        if use_ordinal_loss:
            return _ordinal_ce_loss(
                logits=logits,
                targets=targets,
                num_classes=num_classes,
                class_weights=class_weights_for_loss,
                label_smoothing=label_smoothing,
                ordinal_weight=ordinal_loss_weight,
            )
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights_for_loss,
            label_smoothing=label_smoothing,
        )

    use_scheduler = bool(conf["training"].get("use_scheduler", True))
    scheduler_name = str(conf["training"].get("scheduler_name", "reduce_on_plateau")).strip().lower()
    scheduler = None
    scheduler_mode = "none"
    if use_scheduler and scheduler_name in {"cosine", "cosine_annealing", "cosineannealing"}:
        min_lr_value = float(conf["training"].get("scheduler_min_lr", 1e-8))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs),
            eta_min=min_lr_value,
        )
        scheduler_mode = "cosine"
    elif use_scheduler:
        min_lr_ratio = float(conf["training"].get("scheduler_min_lr_ratio", 0.01))
        if use_differential_lr:
            min_lrs = [max(1e-10, float(pg["lr"]) * min_lr_ratio) for pg in optimizer.param_groups]
            min_lr_value: float | list[float] = min_lrs
        else:
            min_lr_value = float(conf["training"].get("scheduler_min_lr", 1e-8))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(conf["training"].get("scheduler_factor", 0.5)),
            patience=int(conf["training"].get("scheduler_patience", 3)),
            min_lr=min_lr_value,
        )
        scheduler_mode = "plateau"

    best_f1 = -1.0
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "lr": [],
        "lr_groups": [],
        "epoch_duration_seconds": [],
        "lr_group_names": [f"group_{i}" for i in range(len(optimizer.param_groups))],
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name if use_scheduler else "none",
        "loss_name": "focal" if use_focal_loss else ("ordinal" if use_ordinal_loss else "cross_entropy"),
        "grad_accum_steps": int(grad_accum_steps),
        "train_started_at": train_started_at,
    }

    log_every_batches_default = 100 if _is_benchmark(conf) else 0
    log_every_batches = int(conf["training"].get("log_every_batches", log_every_batches_default))
    log_every_batches = max(0, log_every_batches)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)
        num_train_batches = max(1, len(train_loader))
        epoch_start = time.perf_counter()

        if log_every_batches > 0:
            print(
                f"seed={seed} epoch={epoch:02d}/{max_epochs} started "
                f"batches={num_train_batches}",
                flush=True,
            )

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = _loss_fn(logits, y)
            (loss / grad_accum_steps).backward()
            if (batch_idx % grad_accum_steps == 0) or (batch_idx == num_train_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_sum += loss.item() * x.size(0)
            train_count += int(x.size(0))

            if log_every_batches > 0 and (
                batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_train_batches
            ):
                elapsed_s = max(1e-6, time.perf_counter() - epoch_start)
                batches_per_s = batch_idx / elapsed_s
                eta_s = max(0.0, (num_train_batches - batch_idx) / max(1e-6, batches_per_s))
                running_train_loss = train_sum / max(1, train_count)
                print(
                    f"seed={seed} epoch={epoch:02d} "
                    f"batch={batch_idx:04d}/{num_train_batches} "
                    f"train_loss_running={running_train_loss:.4f} "
                    f"elapsed={elapsed_s/60.0:.1f}m eta={eta_s/60.0:.1f}m",
                    flush=True,
                )

        train_loss = train_sum / max(1, train_count)

        model.eval()
        val_sum = 0.0
        val_count = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = _loss_fn(logits, y)
                val_sum += loss.item() * x.size(0)
                val_count += int(x.size(0))

                pred = torch.argmax(logits, dim=1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())

        val_loss = val_sum / max(1, val_count)
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if scheduler is not None and scheduler_mode == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and scheduler_mode == "cosine":
            scheduler.step()
        current_lr = float(max(pg["lr"] for pg in optimizer.param_groups))

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_macro_f1"].append(float(val_f1))
        history["lr"].append(current_lr)
        history["lr_groups"].append([float(pg["lr"]) for pg in optimizer.param_groups])
        epoch_elapsed_s = max(1e-6, time.perf_counter() - epoch_start)
        history["epoch_duration_seconds"].append(float(epoch_elapsed_s))

        print(
            f"seed={seed} epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={current_lr:.6f} "
            f"epoch_time={epoch_elapsed_s/60.0:.1f}m",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": int(seed),
                    "run_id": run_id,
                    "best_val_macro_f1": float(best_f1),
                    "config_signature": _checkpoint_config_signature(conf),
                },
                ckpt,
            )
        else:
            wait += 1

        if wait >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break

    if not ckpt.exists():
        raise RuntimeError(f"Training completed but no checkpoint was saved: {ckpt}")

    train_finished_at = datetime.now().isoformat(timespec="seconds")
    train_duration_seconds = max(1e-6, time.perf_counter() - train_start_perf)
    history["train_finished_at"] = train_finished_at
    history["train_duration_seconds"] = float(train_duration_seconds)
    history["train_duration_minutes"] = float(train_duration_seconds / 60.0)
    history["epochs_ran"] = int(len(history["train_loss"]))
    history["best_val_macro_f1"] = float(best_f1) if np.isfinite(best_f1) else None

    train_history_path = _train_history_log_path(conf, run_id)
    _save_json(train_history_path, history)
    if _use_legacy_aliases(conf):
        _write_alias_copy(train_history_path, _legacy_train_history_alias_path(conf, seed))
        _write_alias_copy(ckpt, _legacy_checkpoint_alias_path(conf, seed))
    _save_latest_run_record(
        conf,
        seed=seed,
        run_id=run_id,
        checkpoint_path=ckpt,
        extra_fields={
            "status": "trained",
            "train_started_at": train_started_at,
            "train_finished_at": train_finished_at,
            "train_duration_seconds": float(train_duration_seconds),
            "train_duration_minutes": float(train_duration_seconds / 60.0),
            "epochs_ran": int(len(history["train_loss"])),
            "best_val_macro_f1": float(best_f1) if np.isfinite(best_f1) else None,
            "train_history_path": str(train_history_path.resolve()),
        },
    )

    return str(ckpt)


def build_validation_calibration_table(cfg: str | Path | dict[str, Any], seed: int = 1988, manifests: dict[str, str] | None = None, checkpoint: str | Path | None = None) -> str:
    conf = _cfg(cfg)
    calib_cfg = conf.get("calibration", {})
    n_bins = int(calib_cfg.get("n_bins", 15))
    temperature_max_iter = int(calib_cfg.get("temperature_max_iter", 50))
    interpolate_empty_bins = bool(calib_cfg.get("interpolate_empty_bins", False))

    if manifests is None:
        val_manifest = _manifest_path(conf, "val", seed=seed)
        if not val_manifest.exists():
            manifests = prepare_data_manifests(conf, seed=seed)
            val_manifest = Path(manifests["val_split"])
    else:
        val_manifest = Path(manifests["val_split"])

    device = _resolve_device(conf["training"].get("device", "auto"))

    if checkpoint is None:
        try:
            ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
        except FileNotFoundError:
            trained_ckpt = train_dr_classifier(conf, seed=seed, manifests=manifests)
            ckpt_path, run_id = _resolve_checkpoint_and_run_id(
                conf,
                seed=seed,
                checkpoint=trained_ckpt,
                require_existing=True,
            )
    else:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=checkpoint,
            require_existing=True,
        )

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    val_logits, val_labels = _collect_logits_and_labels(model, val_manifest, conf, device=device)
    temperature = _fit_temperature(
        logits=val_logits,
        labels=val_labels,
        max_iter=temperature_max_iter,
    )
    val_df_uncal = _predict_manifest(
        model,
        val_manifest,
        conf,
        split="val",
        device=device,
        temperature=1.0,
    )
    val_df = _predict_manifest(
        model,
        val_manifest,
        conf,
        split="val",
        device=device,
        temperature=temperature,
    )
    val_predictions_path = _predictions_path_for_run_id(conf, run_id, "val")
    val_df.to_csv(val_predictions_path, index=False)
    if _use_legacy_aliases(conf):
        _write_alias_copy(val_predictions_path, _legacy_predictions_alias_path(conf, seed, "val"))

    rel = _build_reliability_table(val_df, n_bins=n_bins, interpolate_empty_bins=interpolate_empty_bins)
    rel_path = _table_path(conf, "calibration_bins", seed=seed)
    rel.to_csv(rel_path, index=False)
    ece_before = _expected_calibration_error(val_df_uncal, n_bins=n_bins)
    ece_after = _expected_calibration_error(val_df, n_bins=n_bins)
    calib_summary_path = _table_path(conf, "calibration_summary", seed=seed)
    pd.DataFrame(
        [
            {
                "seed": int(seed),
                "run_id": run_id,
                "n_bins": int(n_bins),
                "interpolate_empty_bins": int(interpolate_empty_bins),
                "temperature": float(temperature),
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "ece_delta": float(ece_after - ece_before) if (not np.isnan(ece_before) and not np.isnan(ece_after)) else float("nan"),
            }
        ]
    ).to_csv(calib_summary_path, index=False)

    payload = {
        "seed": int(seed),
        "run_id": run_id,
        "n_bins": n_bins,
        "temperature": float(temperature),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "rows": rel.to_dict(orient="records"),
        "table_path": str(rel_path),
        "calibration_bins_path": str(rel_path),
        "calibration_summary_path": str(calib_summary_path),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
    }
    out = _calibration_path_for_run_id(conf, run_id)
    _save_json(out, payload)
    if _use_legacy_aliases(conf):
        _write_alias_copy(out, _legacy_calibration_alias_path(conf, seed))
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)
    return str(out)


def run_split_inference(cfg: str | Path | dict[str, Any], seed: int = 1988, split: str = "test") -> str:
    conf = _cfg(cfg)
    manifest = _manifest_path(conf, split, seed=seed)
    if not manifest.exists():
        prepare_data_manifests(conf, seed=seed)

    device = _resolve_device(conf["training"].get("device", "auto"))

    try:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    except FileNotFoundError:
        trained_ckpt = train_dr_classifier(conf, seed=seed)
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=trained_ckpt,
            require_existing=True,
        )

    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        build_validation_calibration_table(conf, seed=seed, checkpoint=ckpt_path)
        calibration_path = _calibration_path_for_run_id(conf, run_id)

    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    pred_df = _predict_manifest(
        model,
        manifest,
        conf,
        split=split,
        device=device,
        temperature=temperature,
    )

    # Contract order
    ordered_cols = [
        "sample_id",
        "true_class",
        "pred_class",
        "confidence",
        "laterality",
        "split",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
        "image_path",
    ]
    pred_df = pred_df[ordered_cols]

    out = _predictions_path_for_run_id(conf, run_id, split)
    pred_df.to_csv(out, index=False)
    if _use_legacy_aliases(conf):
        _write_alias_copy(out, _legacy_predictions_alias_path(conf, seed, split))
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)
    return str(out)


# -----------------------------
# XAI
# -----------------------------


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.maximum(a, 0)
    d = float(a.max() - a.min())
    if d <= 1e-8:
        return np.zeros_like(a)
    return (a - a.min()) / d


def _overlay(base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, cmap_name: str = "jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * (base_image / 255.0) + alpha * colored
    return np.clip(overlay, 0, 1)


def _save_overlay_image(overlay: np.ndarray, out_path: str | Path, dpi: int = 180) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout()
    plt.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=0)
    plt.close()
    return str(out)


def _infer_backbone_from_model(model: nn.Module) -> str:
    net = getattr(model, "net", model)
    if all(hasattr(net, name) for name in ["layer2", "layer3", "layer4"]):
        return "resnet50"
    net_name = net.__class__.__name__.strip().lower()
    if net_name == "vgg":
        return "vgg16"
    if net_name == "densenet":
        return "densenet121"
    if hasattr(net, "features") and isinstance(getattr(net, "features"), nn.Sequential):
        return "efficientnet_b4"
    if hasattr(net, "heads"):
        return "vit_b16"
    return net_name


def _resolve_gradcam_target_layer(
    model: DRClassifier,
    layer_name: str,
    backbone_hint: str = "",
) -> tuple[nn.Module | None, str, str]:
    layer_key = str(layer_name).strip().lower()
    allowed = {"layer2", "layer3", "layer4"}
    if layer_key not in allowed:
        return None, "", f"Unsupported gradcam layer '{layer_name}'. Supported layers: layer2, layer3, layer4"

    net = model.net
    backbone = str(backbone_hint).strip().lower() or _infer_backbone_from_model(model)
    if backbone.startswith("vit"):
        return None, "", f"Grad-CAM requires CNN feature maps; backbone={backbone} is not supported."

    # ResNet-family mapping (explicit named stages).
    if all(hasattr(net, name) for name in ["layer2", "layer3", "layer4"]):
        target = getattr(net, layer_key, None)
        if target is not None:
            return target, f"net.{layer_key}", ""

    # EfficientNet family mapping (explicit feature stages requested by design).
    if backbone == "efficientnet_b4" and hasattr(net, "features") and isinstance(getattr(net, "features"), nn.Sequential):
        stage_map = {"layer2": 4, "layer3": 6, "layer4": 8}
        features = net.features
        req_idx = int(stage_map[layer_key])
        if len(features) > 0:
            idx = req_idx if req_idx < len(features) else int(round((req_idx / 8.0) * (len(features) - 1)))
            idx = int(min(max(idx, 0), len(features) - 1))
            return features[idx], f"net.features.{idx}", ""

    # Generic CNN fallback: map layer2/3/4 to early/mid/late Conv2d stages.
    conv_layers = [(name, module) for name, module in net.named_modules() if isinstance(module, nn.Conv2d)]
    if len(conv_layers) == 0:
        return None, "", "Grad-CAM requires at least one Conv2d spatial layer."

    n_conv = len(conv_layers)
    idx_map = {
        "layer2": int(min(max(round(n_conv * 0.33) - 1, 0), n_conv - 1)),
        "layer3": int(min(max(round(n_conv * 0.66) - 1, 0), n_conv - 1)),
        "layer4": n_conv - 1,
    }
    sel_idx = idx_map[layer_key]
    sel_name, sel_module = conv_layers[sel_idx]
    resolved_name = f"net.{sel_name}" if sel_name else "net"
    return sel_module, resolved_name, ""


def _is_shap_inplace_view_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "backwardhookfunctionbackward" in msg
        or ("modified inplace" in msg and "view" in msg)
        or ("custom function" in msg and "inplace" in msg)
    )


def _is_mps_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("mps" in msg and "out of memory" in msg) or "mps backend out of memory" in msg


def _should_retry_shap_on_cpu(exc: Exception, device: torch.device) -> bool:
    if getattr(device, "type", "") != "mps":
        return False
    return _is_shap_inplace_view_error(exc) or _is_mps_oom_error(exc)


def _empty_mps_cache_if_available() -> None:
    try:
        if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _generate_gradcam(
    model: DRClassifier,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_id: int,
    layer_name: str,
    output_path: str | Path,
    device: torch.device,
    overlay_dpi: int = 180,
    backbone_hint: str = "",
) -> tuple[str, np.ndarray]:
    if _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(f"captum import failed: {_CAPTUM_IMPORT_ERROR}")

    model.eval()
    target_layer, resolved_layer_name, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
        backbone_hint=backbone_hint,
    )
    if target_layer is None:
        raise ValueError(reason or f"Unable to resolve Grad-CAM layer '{layer_name}'.")

    gradcam = LayerGradCam(lambda x: model(x), target_layer)
    attr = gradcam.attribute(input_tensor.to(device), target=class_id)

    input_h, input_w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
    attr_input = LayerAttribution.interpolate(attr, (input_h, input_w))

    heat_input = attr_input.squeeze().detach().cpu().numpy()
    if heat_input.ndim == 3:
        heat_input = np.mean(heat_input, axis=0)
    heat_input = _normalize_map(heat_input)

    base_h, base_w = original_image.shape[0], original_image.shape[1]
    heat_display = np.array(
        Image.fromarray(heat_input.astype(np.float32), mode="F").resize((base_w, base_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    heat_display = _normalize_map(heat_display)

    overlay = _overlay(original_image, heat_display, alpha=0.4, cmap_name="jet")
    artifact = _save_overlay_image(overlay, output_path, dpi=overlay_dpi)
    return artifact, heat_input


def _bottleneck_forward_shap_safe(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out, inplace=False)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = F.relu(out, inplace=False)
    return out


def _basicblock_forward_shap_safe(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = F.relu(out, inplace=False)
    return out


def _mbconv_forward_shap_safe(self: MBConv, x: torch.Tensor) -> torch.Tensor:
    result = self.block(x)
    if self.use_res_connect:
        result = self.stochastic_depth(result)
        result = result + x
    return result


def _fused_mbconv_forward_shap_safe(self: FusedMBConv, x: torch.Tensor) -> torch.Tensor:
    result = self.block(x)
    if self.use_res_connect:
        result = self.stochastic_depth(result)
        result = result + x
    return result


def _make_shap_compatible(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "inplace"):
            try:
                module.inplace = False
            except Exception:
                pass
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)):
            module.inplace = False
        if isinstance(module, Bottleneck):
            module.forward = types.MethodType(_bottleneck_forward_shap_safe, module)
        elif isinstance(module, BasicBlock):
            module.forward = types.MethodType(_basicblock_forward_shap_safe, module)
        elif isinstance(module, MBConv):
            module.forward = types.MethodType(_mbconv_forward_shap_safe, module)
        elif isinstance(module, FusedMBConv):
            module.forward = types.MethodType(_fused_mbconv_forward_shap_safe, module)


def _build_shap_explainer_with_known_warning_filter(
    wrapper: nn.Module,
    background: torch.Tensor,
) -> Any:
    # SHAP DeepExplainer logs "unrecognized nn.Module" for several harmless
    # modules in torchvision EfficientNet (e.g., SiLU, StochasticDepth).
    # Filter only these known warnings to keep notebook logs clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: SiLU",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: StochasticDepth",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        return shap.DeepExplainer(wrapper, background)


def _shap_values_with_known_warning_filter(explainer: Any, inputs: torch.Tensor) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: SiLU",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: StochasticDepth",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        return explainer.shap_values(inputs, check_additivity=False)


def _pick_shap_map(shap_values: object, class_id: int, sample_index: int = 0) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_id][sample_index])

    arr = np.asarray(shap_values)
    if arr.ndim == 5:
        return arr[sample_index, :, :, :, class_id]
    if arr.ndim == 4:
        return arr[sample_index]
    raise ValueError(f"Unsupported SHAP array shape: {arr.shape}")


def _shap_to_2d(shap_map: np.ndarray, mode: str = "positive") -> np.ndarray:
    arr = np.asarray(shap_map, dtype=np.float32)
    if arr.ndim == 3:
        if mode == "signed":
            arr = np.mean(arr, axis=0)
        elif mode == "abs":
            arr = np.mean(np.abs(arr), axis=0)
        else:
            arr = np.mean(np.maximum(arr, 0.0), axis=0)
    elif arr.ndim != 2:
        raise ValueError(f"SHAP map must be 2D or 3D, got ndim={arr.ndim}")

    mode_key = str(mode).strip().lower()
    if mode_key == "signed":
        max_abs = float(np.max(np.abs(arr)))
        if max_abs <= 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / max_abs).astype(np.float32)
    if mode_key == "abs":
        return _normalize_map(np.abs(arr))
    # default: positive evidence for predicted class
    return _normalize_map(np.maximum(arr, 0.0))


def _save_map_overlay(attr_map: np.ndarray, base_image: np.ndarray, output_path: str | Path, overlay_dpi: int = 180) -> str:
    arr = np.asarray(attr_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("attr_map must be 2D")

    if base_image.ndim == 2:
        base_image = np.stack([base_image] * 3, axis=-1)

    if base_image.shape[:2] != arr.shape[:2]:
        base_image = np.array(
            Image.fromarray(base_image.astype(np.uint8)).resize((arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR)
        )

    overlay = _overlay(base_image, arr, alpha=0.4, cmap_name="coolwarm")
    return _save_overlay_image(overlay, output_path, dpi=overlay_dpi)


def _choose_xai_targets(
    df: pd.DataFrame,
    high_conf_threshold: float,
    max_targets: int | None = None,
    num_classes: int = 5,
    class_col: str = "true_class",
    balance_by_class: bool = True,
    fill_missing_from_all: bool = True,
) -> pd.DataFrame:
    target = df[df["confidence"] >= high_conf_threshold].copy()
    if len(target) == 0:
        target = df.sort_values("confidence", ascending=False).head(min(32, len(df))).copy()

    if not balance_by_class or len(target) == 0:
        if max_targets is not None and int(max_targets) > 0:
            target = target.head(int(max_targets)).copy()
        return target.reset_index(drop=True)

    target = target.sort_values("confidence", ascending=False).copy()
    available_class_col = class_col if class_col in target.columns else ("pred_class" if "pred_class" in target.columns else class_col)
    if available_class_col not in target.columns:
        if max_targets is not None and int(max_targets) > 0:
            target = target.head(int(max_targets)).copy()
        return target.reset_index(drop=True)

    limit = int(max_targets) if (max_targets is not None and int(max_targets) > 0) else len(target)
    limit = max(1, limit)
    class_ids = list(range(int(num_classes)))
    per_class = max(1, limit // max(1, len(class_ids)))

    selected_parts: list[pd.DataFrame] = []
    for cid in class_ids:
        part = target[target[available_class_col].astype(int) == int(cid)].head(per_class)
        if len(part):
            selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else target.head(0).copy()

    if fill_missing_from_all and len(selected) < limit:
        missing_classes = {
            int(cid)
            for cid in class_ids
            if len(selected[selected[available_class_col].astype(int) == int(cid)]) == 0
        }
        if missing_classes and available_class_col in df.columns:
            pool = df.sort_values("confidence", ascending=False)
            for cid in sorted(missing_classes):
                add = pool[pool[available_class_col].astype(int) == int(cid)].head(1)
                if len(add):
                    selected = pd.concat([selected, add], ignore_index=True)

    if len(selected) < limit:
        used_ids = set(selected["sample_id"].astype(str).tolist()) if "sample_id" in selected.columns else set()
        filler = target[~target["sample_id"].astype(str).isin(used_ids)] if "sample_id" in target.columns else target
        need = max(0, limit - len(selected))
        if need > 0:
            selected = pd.concat([selected, filler.head(need)], ignore_index=True)

    selected = selected.head(limit).copy()
    return selected.reset_index(drop=True)


def _parse_gradcam_layers(raw_layers: Any, default_layer: str = "layer4") -> list[str]:
    allowed = {"layer2", "layer3", "layer4"}
    values: list[str] = []
    if isinstance(raw_layers, (list, tuple)):
        for v in raw_layers:
            key = str(v).strip().lower()
            if key in allowed and key not in values:
                values.append(key)
    elif isinstance(raw_layers, str):
        key = raw_layers.strip().lower()
        if key in allowed:
            values.append(key)
    default_key = str(default_layer).strip().lower()
    if default_key in allowed and default_key not in values:
        values.insert(0, default_key)
    if not values:
        values = [default_key if default_key in allowed else "layer4"]
    return values


def run_xai_analysis(
    cfg: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    checkpoint: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    shap_mode: str | None = None,
    shap_max_samples: int | None = None,
) -> dict[str, str]:
    conf = _cfg(cfg)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    xai_cfg = conf.get("xai", {})
    pass_border_ratio_max, pass_faith_delta_min = _xai_pass_rule_thresholds(conf)
    k_list = _parse_faithfulness_k_list(xai_cfg.get("faithfulness_k_list", [0.05, 0.10, 0.20, 0.30]))
    k_cols = [_k_to_col_name(k) for k in k_list]
    max_targets = int(xai_cfg.get("max_targets", 0))
    if max_targets <= 0:
        max_targets = None
    fig_dpi = int(xai_cfg.get("figure_dpi", 180))
    bootstrap_iters = int(xai_cfg.get("stats_bootstrap_iters", 1000))
    bootstrap_seed = int(xai_cfg.get("stats_bootstrap_seed", 1988))
    xai_log_every_samples = int(xai_cfg.get("log_every_samples", 16))
    xai_start_time = time.time()

    if checkpoint is None:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    else:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=checkpoint,
            require_existing=True,
        )
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)

    if predictions_csv is None:
        pred_path = Path(run_split_inference(conf, seed=seed, split=split))
    else:
        pred_path = Path(predictions_csv)

    pred_df = pd.read_csv(pred_path)
    high_conf_thr = float(conf["evaluation"].get("high_conf_threshold", 0.80))
    balance_targets = bool(xai_cfg.get("balance_targets_by_class", True))
    fill_missing_class = bool(xai_cfg.get("balance_fill_from_all", True))
    target_class_col = str(xai_cfg.get("target_balance_class_col", "true_class"))
    target_df = _choose_xai_targets(
        pred_df,
        high_conf_threshold=high_conf_thr,
        max_targets=max_targets,
        num_classes=int(conf["data"]["num_classes"]),
        class_col=target_class_col,
        balance_by_class=balance_targets,
        fill_missing_from_all=fill_missing_class,
    )

    use_mode = str(shap_mode if shap_mode is not None else xai_cfg.get("shap_mode", "subset")).strip().lower()
    if use_mode not in {"subset", "full"}:
        raise ValueError(f"Unsupported shap_mode={use_mode}")
    max_samples = int(shap_max_samples if shap_max_samples is not None else xai_cfg.get("shap_max_samples", 128))
    enforce_consistent_targets = bool(xai_cfg.get("enforce_consistent_targets", True))
    xai_target_df = target_df.copy()
    if enforce_consistent_targets and use_mode == "subset" and max_samples > 0:
        xai_target_df = xai_target_df.head(max_samples).copy()
    xai_target_df = xai_target_df.reset_index(drop=True)

    xai_targets_path = Path(conf["paths"]["tables_dir"]) / f"xai_targets_seed{seed}_{split}.csv"
    xai_target_df.to_csv(xai_targets_path, index=False)

    if high_conf_thr > 0.0:
        evaluation_scope = f"high_confidence_subset(confidence>={high_conf_thr:.2f})"
    else:
        evaluation_scope = "all_confidence"
    if max_targets is not None and max_targets > 0:
        evaluation_scope = f"{evaluation_scope}; max_targets={int(max_targets)}"

    print(
        f"[XAI] run_id={run_id} split={split} device={device} "
        f"targets={len(xai_target_df)} backbone={backbone} "
        f"shared_targets={enforce_consistent_targets} shap_mode={use_mode} "
        f"pass_rule=(border<={pass_border_ratio_max:.2f}, faith_k20>{pass_faith_delta_min:.2f})"
    )

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)

    # Grad-CAM (RQ1)
    grad_rows: list[dict[str, Any]] = []
    grad_dir = Path(conf["paths"]["figures_dir"]) / "gradcam"
    gradcam_status_path = _gradcam_status_log_path(conf, run_id, split)
    gradcam_status_note = ""
    layer_summary_path: Path | None = None

    default_gradcam_layer = str(conf["xai"].get("gradcam_layer", "layer4"))
    gradcam_layers_eval = _parse_gradcam_layers(
        raw_layers=conf["xai"].get("gradcam_layers_eval", [default_gradcam_layer]),
        default_layer=default_gradcam_layer,
    )
    layer_rows: dict[str, list[dict[str, Any]]] = {layer: [] for layer in gradcam_layers_eval}
    unsupported_layers: dict[str, str] = {}
    supported_layers: list[str] = []
    layer_errors: dict[str, str] = {}

    for layer_name in gradcam_layers_eval:
        _, _, reason = _resolve_gradcam_target_layer(
            model=model,
            layer_name=layer_name,
            backbone_hint=backbone,
        )
        if reason:
            unsupported_layers[layer_name] = reason
        else:
            supported_layers.append(layer_name)

    if not supported_layers:
        reason = "; ".join(sorted(set(unsupported_layers.values()))) or "No Grad-CAM compatible feature layer was found."
        _save_json(
            gradcam_status_path,
            {
                "status": "skipped",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "backbone": backbone,
                "requested_layers": gradcam_layers_eval,
                "supported_layers": [],
                "unsupported_layers": unsupported_layers,
                "reason": reason,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        gradcam_status_note = f"skipped: {reason}"
    else:
        total_grad_jobs = int(len(xai_target_df) * len(supported_layers))
        grad_job_idx = 0
        for _, row in xai_target_df.iterrows():
            image_pil, image_tensor = _load_image_for_inference(
                row["image_path"],
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            image_tensor = image_tensor.to(device)
            target_class_value = int(row[target_class_col]) if target_class_col in row.index else int(row["pred_class"])

            for layer_name in supported_layers:
                grad_job_idx += 1
                out_path = grad_dir / f"{row['sample_id']}_cls{int(row['pred_class'])}_{layer_name}.png"
                try:
                    artifact, heat_input = _generate_gradcam(
                        model=model,
                        input_tensor=image_tensor,
                        original_image=np.array(image_pil),
                        class_id=int(row["pred_class"]),
                        layer_name=layer_name,
                        output_path=out_path,
                        device=device,
                        overlay_dpi=fig_dpi,
                        backbone_hint=backbone,
                    )
                except Exception as exc:
                    layer_errors[layer_name] = str(exc)
                    if xai_log_every_samples > 0 and (
                        grad_job_idx == 1
                        or grad_job_idx % xai_log_every_samples == 0
                        or grad_job_idx == total_grad_jobs
                    ):
                        elapsed_m = (time.time() - xai_start_time) / 60.0
                        print(
                            f"[XAI][GradCAM] job={grad_job_idx}/{total_grad_jobs} "
                            f"layer={layer_name} status=error elapsed={elapsed_m:.1f}m"
                        )
                    continue

                ratios = _attribution_mass_ratios(heat_input)
                faith_by_k, aopc_delta = _faithfulness_multi_k(
                    model=model,
                    image_tensor=image_tensor,
                    score_map=heat_input,
                    pred_class=int(row["pred_class"]),
                    k_list=k_list,
                    random_seed=bootstrap_seed,
                )
                faith_legacy = float(faith_by_k.get("faith_delta_k20", np.nan))
                if np.isnan(faith_legacy):
                    faith_legacy = _faithfulness_delta(
                        model=model,
                        image_tensor=image_tensor,
                        score_map=heat_input,
                        pred_class=int(row["pred_class"]),
                        top_k_ratio=0.20,
                        random_seed=bootstrap_seed,
                    )
                pass_flag = _xai_pass_flag(
                    border_ratio=float(ratios["border_ratio"]),
                    faith_delta=float(faith_legacy),
                    border_ratio_max=pass_border_ratio_max,
                    faith_delta_min=pass_faith_delta_min,
                )
                aopc_pass = int(aopc_delta > 0.0) if not np.isnan(aopc_delta) else 0

                layer_rows[layer_name].append(
                    {
                        "sample_id": row["sample_id"],
                        "target_class": target_class_value,
                        "pred_class": int(row["pred_class"]),
                        "confidence": float(row["confidence"]),
                        "gradcam_layer": layer_name,
                        "artifact_path": artifact,
                        "border_ratio": float(ratios["border_ratio"]),
                        "retina_ratio": float(ratios["retina_ratio"]),
                        "faithfulness_delta": float(faith_legacy),
                        "gradcam_pass": pass_flag,
                        **faith_by_k,
                        "aopc_delta": float(aopc_delta),
                        "aopc_pass": int(aopc_pass),
                    }
                )
                if xai_log_every_samples > 0 and (
                    grad_job_idx == 1
                    or grad_job_idx % xai_log_every_samples == 0
                    or grad_job_idx == total_grad_jobs
                ):
                    elapsed_m = (time.time() - xai_start_time) / 60.0
                    print(
                        f"[XAI][GradCAM] job={grad_job_idx}/{total_grad_jobs} "
                        f"layer={layer_name} status=ok elapsed={elapsed_m:.1f}m"
                    )

        layer_scores: list[tuple[str, float, int]] = []
        for layer_name in supported_layers:
            rows = layer_rows.get(layer_name, [])
            if len(rows) == 0:
                layer_scores.append((layer_name, -np.inf, 0))
                continue
            layer_df = pd.DataFrame(rows)
            score = float(pd.to_numeric(layer_df["aopc_delta"], errors="coerce").mean())
            if np.isnan(score):
                score = -np.inf
            layer_scores.append((layer_name, score, int(len(rows))))

        selected_layer = default_gradcam_layer if default_gradcam_layer in supported_layers else supported_layers[0]
        if layer_scores:
            ranked_scores = sorted(layer_scores, key=lambda x: x[1], reverse=True)
            selected_layer = ranked_scores[0][0]
        grad_rows = layer_rows.get(selected_layer, [])

        layer_summary_path = Path(conf["paths"]["tables_dir"]) / f"gradcam_layer_selection_seed{seed}_{split}.csv"
        pd.DataFrame(layer_scores, columns=["gradcam_layer", "mean_aopc_delta", "n_rows"]).to_csv(layer_summary_path, index=False)

        if len(grad_rows) == 0:
            reason = "No Grad-CAM rows were produced from supported layers."
            if layer_errors:
                reason = f"{reason} Errors: {layer_errors}"
            _save_json(
                gradcam_status_path,
                {
                    "status": "failed",
                    "seed": int(seed),
                    "run_id": run_id,
                    "split": split,
                    "backbone": backbone,
                    "requested_layers": gradcam_layers_eval,
                    "supported_layers": supported_layers,
                    "unsupported_layers": unsupported_layers,
                    "errors": layer_errors,
                    "selected_layer": selected_layer,
                    "reason": reason,
                    "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                    "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                    "evaluation_scope": evaluation_scope,
                    "shared_targets_enforced": bool(enforce_consistent_targets),
                    "target_rows": int(len(xai_target_df)),
                },
            )
            gradcam_status_note = f"failed: {reason}"
        else:
            status = "success" if len(layer_errors) == 0 and len(unsupported_layers) == 0 else "partial_success"
            _save_json(
                gradcam_status_path,
                {
                    "status": status,
                    "seed": int(seed),
                    "run_id": run_id,
                    "split": split,
                    "backbone": backbone,
                    "requested_layers": gradcam_layers_eval,
                    "supported_layers": supported_layers,
                    "unsupported_layers": unsupported_layers,
                    "errors": layer_errors,
                    "selected_layer": selected_layer,
                    "rows": int(len(grad_rows)),
                    "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                    "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                    "evaluation_scope": evaluation_scope,
                    "shared_targets_enforced": bool(enforce_consistent_targets),
                    "target_rows": int(len(xai_target_df)),
                },
            )
            gradcam_status_note = status

    if _use_legacy_aliases(conf):
        _write_alias_copy(gradcam_status_path, _legacy_gradcam_status_alias_path(conf, seed, split))

    rq1_df = pd.DataFrame(
        grad_rows,
        columns=[
            "sample_id",
            "target_class",
            "pred_class",
            "confidence",
            "gradcam_layer",
            "artifact_path",
            "border_ratio",
            "retina_ratio",
            "faithfulness_delta",
            "gradcam_pass",
            *k_cols,
            "aopc_delta",
            "aopc_pass",
        ],
    )
    rq1_path = Path(conf["paths"]["tables_dir"]) / f"rq1_gradcam_seed{seed}_{split}.csv"
    rq1_df.to_csv(rq1_path, index=False)

    # SHAP (RQ2)
    shap_rows: list[dict[str, Any]] = []
    shap_status_path = _shap_status_log_path(conf, run_id, split)
    shap_status_note = ""
    shap_attempted_device = str(device)
    shap_final_device = str(device)
    shap_fallback_used = False
    shap_error_primary = ""
    shap_error = ""

    try:
        if _SHAP_IMPORT_ERROR is not None:
            raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")

        bg_size = int(xai_cfg.get("shap_background_size", 64))
        shap_score_mode = str(xai_cfg.get("shap_score_mode", "positive")).strip().lower()
        if enforce_consistent_targets:
            shap_target = xai_target_df.copy()
        elif use_mode == "full":
            shap_target = target_df.copy()
        else:
            shap_target = target_df.head(max_samples).copy()

        if len(shap_target) == 0:
            raise RuntimeError("No target samples selected for SHAP")

        train_manifest = _manifest_path(conf, "train", seed=seed)
        train_df = pd.read_csv(train_manifest)
        if len(train_df) == 0:
            raise RuntimeError("Train manifest is empty; cannot build SHAP background")

        bg_df = train_df.groupby("class_id", group_keys=False).head(max(1, bg_size // int(conf["data"]["num_classes"])))
        if len(bg_df) < bg_size:
            extra = train_df.head(bg_size - len(bg_df))
            bg_df = pd.concat([bg_df, extra], ignore_index=True)
        bg_df = bg_df.head(bg_size)
        shap_dir = Path(conf["paths"]["figures_dir"]) / "shap"

        def _run_shap_for_device(shap_device: torch.device) -> list[dict[str, Any]]:
            if str(shap_device) == str(device):
                shap_model = model
            else:
                shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
            _make_shap_compatible(shap_model)
            wrapper = _LogitWrapper(shap_model).to(shap_device)

            bg_tensors = []
            for _, b in bg_df.iterrows():
                _, bt = _load_image_for_inference(
                    b["image_path"],
                    image_size=image_size,
                    preprocessing_cfg=conf.get("preprocessing", {}),
                )
                bg_tensors.append(bt.squeeze(0))
            background = torch.stack(bg_tensors, dim=0).to(shap_device)
            explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background)

            rows: list[dict[str, Any]] = []
            total_shap = int(len(shap_target))
            shap_start = time.time()
            for shap_idx, (_, row) in enumerate(shap_target.iterrows(), start=1):
                base_image, t = _load_image_for_inference(
                    row["image_path"],
                    image_size=image_size,
                    preprocessing_cfg=conf.get("preprocessing", {}),
                )
                t = t.to(shap_device)

                shap_values = _shap_values_with_known_warning_filter(explainer, t)
                smap = _pick_shap_map(shap_values, class_id=int(row["pred_class"]), sample_index=0)
                s2d = _shap_to_2d(smap, mode=shap_score_mode)
                target_class_value = int(row[target_class_col]) if target_class_col in row.index else int(row["pred_class"])

                out_path = shap_dir / f"{row['sample_id']}_cls{int(row['pred_class'])}.png"
                artifact = _save_map_overlay(s2d, np.array(base_image), out_path, overlay_dpi=fig_dpi)

                ratios = _attribution_mass_ratios(s2d)
                faith_by_k, aopc_delta = _faithfulness_multi_k(
                    model=shap_model,
                    image_tensor=t,
                    score_map=s2d,
                    pred_class=int(row["pred_class"]),
                    k_list=k_list,
                    random_seed=bootstrap_seed,
                )
                faith_legacy = float(faith_by_k.get("faith_delta_k20", np.nan))
                if np.isnan(faith_legacy):
                    faith_legacy = _faithfulness_delta(
                        model=shap_model,
                        image_tensor=t,
                        score_map=s2d,
                        pred_class=int(row["pred_class"]),
                        top_k_ratio=0.20,
                        random_seed=bootstrap_seed,
                    )
                pass_flag = _xai_pass_flag(
                    border_ratio=float(ratios["border_ratio"]),
                    faith_delta=float(faith_legacy),
                    border_ratio_max=pass_border_ratio_max,
                    faith_delta_min=pass_faith_delta_min,
                )
                aopc_pass = int(aopc_delta > 0.0) if not np.isnan(aopc_delta) else 0

                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "target_class": target_class_value,
                        "pred_class": int(row["pred_class"]),
                        "confidence": float(row["confidence"]),
                        "shap_score_mode": shap_score_mode,
                        "artifact_path": artifact,
                        "border_ratio": float(ratios["border_ratio"]),
                        "retina_ratio": float(ratios["retina_ratio"]),
                        "faithfulness_delta": float(faith_legacy),
                        "shap_pass": pass_flag,
                        **faith_by_k,
                        "aopc_delta": float(aopc_delta),
                        "aopc_pass": int(aopc_pass),
                    }
                )
                del shap_values, smap, s2d, t, base_image
                if getattr(shap_device, "type", "") == "mps":
                    _empty_mps_cache_if_available()
                    gc.collect()
                if xai_log_every_samples > 0 and (
                    shap_idx == 1
                    or shap_idx % xai_log_every_samples == 0
                    or shap_idx == total_shap
                ):
                    elapsed_m = (time.time() - shap_start) / 60.0
                    print(
                        f"[XAI][SHAP][{shap_device}] sample={shap_idx}/{total_shap} "
                        f"elapsed={elapsed_m:.1f}m"
                    )
            return rows

        try:
            shap_rows = _run_shap_for_device(device)
        except Exception as primary_exc:
            shap_error_primary = str(primary_exc)
            if _should_retry_shap_on_cpu(primary_exc, device):
                _empty_mps_cache_if_available()
                cpu_device = torch.device("cpu")
                print(f"[XAI][SHAP] retry on CPU after MPS failure: {shap_error_primary}")
                try:
                    shap_rows = _run_shap_for_device(cpu_device)
                    shap_final_device = str(cpu_device)
                    shap_fallback_used = True
                except Exception as fallback_exc:
                    shap_error = f"{fallback_exc} (primary_mps_error={shap_error_primary})"
            else:
                shap_error = shap_error_primary

    except Exception as exc:
        shap_error = str(exc)

    if shap_error:
        _save_json(
            shap_status_path,
            {
                "status": "failed",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "mode": str(shap_mode),
                "error": str(shap_error),
                "attempted_device": shap_attempted_device,
                "final_device": shap_final_device,
                "fallback_used": bool(shap_fallback_used),
                "error_primary": str(shap_error_primary),
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                "evaluation_scope": evaluation_scope,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        shap_status_note = f"failed: {shap_error}"
    else:
        _save_json(
            shap_status_path,
            {
                "status": "success",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "mode": use_mode,
                "samples": int(len(shap_rows)),
                "attempted_device": shap_attempted_device,
                "final_device": shap_final_device,
                "fallback_used": bool(shap_fallback_used),
                "error_primary": str(shap_error_primary),
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                "evaluation_scope": evaluation_scope,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        shap_status_note = "success_cpu_fallback" if shap_fallback_used else "success"

    if _use_legacy_aliases(conf):
        _write_alias_copy(shap_status_path, _legacy_shap_status_alias_path(conf, seed, split))

    total_elapsed_m = (time.time() - xai_start_time) / 60.0
    print(
        f"[XAI] completed run_id={run_id} split={split} "
        f"gradcam_rows={len(grad_rows)} shap_rows={len(shap_rows)} "
        f"elapsed={total_elapsed_m:.1f}m"
    )

    rq2_df = pd.DataFrame(
        shap_rows,
        columns=[
            "sample_id",
            "target_class",
            "pred_class",
            "confidence",
            "shap_score_mode",
            "artifact_path",
            "border_ratio",
            "retina_ratio",
            "faithfulness_delta",
            "shap_pass",
            *k_cols,
            "aopc_delta",
            "aopc_pass",
        ],
    )
    rq2_path = Path(conf["paths"]["tables_dir"]) / f"rq2_shap_seed{seed}_{split}.csv"
    rq2_df.to_csv(rq2_path, index=False)

    from .lime_support import run_lime_targets
    import sys
    lime_df, lime_path, lime_status_path, lime_status_note = run_lime_targets(
        sys.modules[__name__], conf, xai_target_df, seed, split, ckpt_path, run_id)
    lime_ids = set(lime_df['sample_id'].astype(str))

    coverage_rows: list[dict[str, Any]] = []
    selected_ids = set(xai_target_df["sample_id"].astype(str).tolist()) if "sample_id" in xai_target_df.columns else set()
    grad_ids = set(rq1_df["sample_id"].astype(str).tolist()) if "sample_id" in rq1_df.columns else set()
    shap_ids = set(rq2_df["sample_id"].astype(str).tolist()) if "sample_id" in rq2_df.columns else set()
    for sid in sorted(selected_ids):
        coverage_rows.append(
            {
                "sample_id": sid,
                "selected_for_xai": 1,
                "gradcam_done": int(sid in grad_ids),
                "shap_done": int(sid in shap_ids),
                "lime_done": int(sid in lime_ids),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows, columns=["sample_id", "selected_for_xai", "gradcam_done", "shap_done", "lime_done"])
    coverage_path = Path(conf["paths"]["tables_dir"]) / f"xai_target_coverage_seed{seed}_{split}.csv"
    coverage_df.to_csv(coverage_path, index=False)
    if len(coverage_df):
        grad_missing = int((coverage_df["gradcam_done"] == 0).sum())
        shap_missing = int((coverage_df["shap_done"] == 0).sum())
        if grad_missing > 0 or shap_missing > 0:
            print(
                f"[XAI][WARN] target coverage gap: "
                f"gradcam_missing={grad_missing}, shap_missing={shap_missing}"
            )

    method_stats_rows = [
        _build_xai_method_stats_row(
            method="gradcam",
            df=rq1_df,
            pass_col="gradcam_pass",
            n_boot=bootstrap_iters,
            seed=bootstrap_seed,
            border_ratio_max=pass_border_ratio_max,
            faith_delta_min=pass_faith_delta_min,
            evaluation_scope=evaluation_scope,
            status_note=gradcam_status_note or ("success" if len(rq1_df) else "no_rows"),
        ),
        _build_xai_method_stats_row(
            method="shap",
            df=rq2_df,
            pass_col="shap_pass",
            n_boot=bootstrap_iters,
            seed=bootstrap_seed,
            border_ratio_max=pass_border_ratio_max,
            faith_delta_min=pass_faith_delta_min,
            evaluation_scope=evaluation_scope,
            status_note=shap_status_note or ("success" if len(rq2_df) else "no_rows"),
        ),
    ]
    method_stats_rows.append(_build_xai_method_stats_row(
        "lime", lime_df, "lime_pass", bootstrap_iters, bootstrap_seed,
        pass_border_ratio_max, pass_faith_delta_min, evaluation_scope, lime_status_note))
    method_stats_df = pd.DataFrame(method_stats_rows)
    method_stats_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_method_stats_seed{seed}_{split}.csv"
    method_stats_df.to_csv(method_stats_path, index=False)

    class_names = [str(x) for x in conf.get("data", {}).get("label_order", [])]
    pass_by_correctness_df = pd.concat(
        [
            _build_xai_pass_by_correctness_table("gradcam", rq1_df, "gradcam_pass"),
            _build_xai_pass_by_correctness_table("shap", rq2_df, "shap_pass"),
            _build_xai_pass_by_correctness_table("lime", lime_df, "lime_pass"),
        ],
        ignore_index=True,
    )
    pass_by_correctness_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pass_by_correctness_seed{seed}_{split}.csv"
    pass_by_correctness_df.to_csv(pass_by_correctness_path, index=False)

    pass_by_class_df = pd.concat(
        [
            _build_xai_pass_by_class_table("gradcam", rq1_df, "gradcam_pass", class_names=class_names),
            _build_xai_pass_by_class_table("shap", rq2_df, "shap_pass", class_names=class_names),
            _build_xai_pass_by_class_table("lime", lime_df, "lime_pass", class_names=class_names),
        ],
        ignore_index=True,
    )
    pass_by_class_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pass_by_class_seed{seed}_{split}.csv"
    pass_by_class_df.to_csv(pass_by_class_path, index=False)

    pairwise_df = _build_xai_pairwise_stats(rq1_df=rq1_df, rq2_df=rq2_df)
    pairwise_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pairwise_seed{seed}_{split}.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    # Preserve the legacy Grad-CAM/SHAP table for existing consumers, and add
    # a method-labelled table covering all three pairs on matched sample IDs.
    pair_rows = []
    for left_name, left, right_name, right in [
        ('gradcam', rq1_df, 'shap', rq2_df),
        ('gradcam', rq1_df, 'lime', lime_df),
        ('shap', rq2_df, 'lime', lime_df),
    ]:
        left_pass = left[['sample_id', left_name + '_pass']].rename(columns={left_name + '_pass': 'gradcam_pass'})
        right_pass = right[['sample_id', right_name + '_pass']].rename(columns={right_name + '_pass': 'shap_pass'})
        comparison = _build_xai_pairwise_stats(left_pass, right_pass).rename(columns={
            'gradcam_pass_rate': 'method_a_pass_rate', 'shap_pass_rate': 'method_b_pass_rate'})
        comparison['method_a'] = left_name
        comparison['method_b'] = right_name
        pair_rows.append(comparison)
    all_pairs_path = Path(conf['paths']['tables_dir']) / f'rq_xai_all_pairs_seed{seed}_{split}.csv'
    pd.concat(pair_rows, ignore_index=True).to_csv(all_pairs_path, index=False)

    # ProtoPNet future-phase stub
    proto_stub = pd.DataFrame(
        [
            {
                "method": "ProtoPNetLite",
                "status": "planned_future_phase",
                "reason": "Planned future extension outside current implementation scope",
            }
        ]
    )
    proto_stub_path = Path(conf["paths"]["tables_dir"]) / f"protopnet_stub_seed{seed}_{split}.csv"
    proto_stub.to_csv(proto_stub_path, index=False)

    return {
        "run_id": run_id,
        "rq1_table": str(rq1_path),
        "rq2_table": str(rq2_path),
        "rq3_table": lime_path,
        "lime_status": lime_status_path,
        "rq_all_pairs_table": str(all_pairs_path),
        "gradcam_layer_selection_table": str(layer_summary_path) if layer_summary_path is not None else "",
        "gradcam_status": str(gradcam_status_path),
        "rq_method_stats_table": str(method_stats_path),
        "rq_pass_by_correctness_table": str(pass_by_correctness_path),
        "rq_pass_by_class_table": str(pass_by_class_path),
        "rq_pairwise_table": str(pairwise_path),
        "shap_status": str(shap_status_path),
        "protopnet_stub": str(proto_stub_path),
        "xai_targets_table": str(xai_targets_path),
        "xai_target_coverage_table": str(coverage_path),
    }


# -----------------------------
# Evaluation
# -----------------------------


def evaluate_pipeline_outputs(cfg: str | Path | dict[str, Any], seed: int = 1988, split: str = "test", predictions_csv: str | Path | None = None) -> dict[str, str]:
    conf = _cfg(cfg)

    if predictions_csv is None:
        pred_path = Path(run_split_inference(conf, seed=seed, split=split))
    else:
        pred_path = Path(predictions_csv)

    df = pd.read_csv(pred_path)
    num_classes = int(conf["data"]["num_classes"])
    high_conf_thr = float(conf["evaluation"].get("high_conf_threshold", 0.80))
    run_id = _run_id_from_predictions_path(pred_path, split=split)
    if not run_id:
        try:
            _, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
        except FileNotFoundError:
            run_id = ""

    m_all, cm_all = _classification_metrics(df, num_classes=num_classes)
    m_high, _ = _classification_metrics(df[df["confidence"] >= high_conf_thr], num_classes=num_classes)

    metrics_rows = [
        {"run_id": run_id, "scope": "all", **m_all},
        {"run_id": run_id, "scope": "high_conf", **m_high},
    ]

    metrics_path = _table_path(conf, "metrics", seed=seed, split=split)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    headline_path = Path(export_final_headline_metrics(conf, seed=seed, split=split, metrics_csv=metrics_path))

    cm_path = _table_path(conf, "confusion", seed=seed, split=split)
    cm_all.to_csv(cm_path, index=False)

    per_class_path = _table_path(conf, "per_class", seed=seed, split=split)
    label_order = [str(x) for x in conf["data"].get("label_order", [])]
    per_class_df = _per_class_metrics_from_confusion(cm_all, label_order=label_order)
    per_class_df.to_csv(per_class_path, index=False)

    run_summary = {
        "seed": int(seed),
        "run_id": run_id,
        "split": split,
        "n_all": int(m_all.get("n", 0.0)),
        "accuracy_all": float(m_all.get("accuracy", np.nan)),
        "qwk_all": float(m_all.get("qwk", np.nan)),
        "recall_macro_all": float(m_all.get("recall_macro", np.nan)),
        "sensitivity_macro_all": float(m_all.get("sensitivity_macro", np.nan)),
        "specificity_macro_all": float(m_all.get("specificity_macro", np.nan)),
        "f1_macro_all": float(m_all.get("f1_macro", np.nan)),
        "n_high_conf": int(m_high.get("n", 0.0)),
        "accuracy_high_conf": float(m_high.get("accuracy", np.nan)),
        "qwk_high_conf": float(m_high.get("qwk", np.nan)),
        "recall_macro_high_conf": float(m_high.get("recall_macro", np.nan)),
        "sensitivity_macro_high_conf": float(m_high.get("sensitivity_macro", np.nan)),
        "specificity_macro_high_conf": float(m_high.get("specificity_macro", np.nan)),
        "f1_macro_high_conf": float(m_high.get("f1_macro", np.nan)),
    }
    run_summary_path = _table_path(conf, "run_summary", seed=seed, split=split)
    pd.DataFrame([run_summary]).to_csv(run_summary_path, index=False)

    return {
        "metrics": str(metrics_path),
        "final_headline_metrics": str(headline_path),
        "confusion": str(cm_path),
        "per_class": str(per_class_path),
        "run_summary": str(run_summary_path),
    }


def export_final_headline_metrics(
    cfg: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    split: str = "test",
    metrics_csv: str | Path | None = None,
) -> str:
    conf = _cfg(cfg)
    if metrics_csv is None:
        metrics_path = _table_path(conf, "metrics", seed=seed, split=split)
    else:
        metrics_path = Path(metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}. Run evaluate_pipeline_outputs first.")

    metrics_df = pd.read_csv(metrics_path)
    required = ["run_id", "scope", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing required metrics columns: {missing}")

    all_df = metrics_df[metrics_df["scope"].astype(str).str.lower() == "all"].copy()
    if len(all_df) == 0:
        raise ValueError("No row found with scope='all' in metrics table.")

    row = all_df.iloc[[0]][["run_id", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].copy()
    out_path = _table_path(conf, "final_headline_metrics", seed=seed, split=split)
    row.to_csv(out_path, index=False)
    return str(out_path)


def _ci95_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        margin = 1.96 * (std / math.sqrt(n))
    else:
        margin = 0.0
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
    }


def _profile_seed_list(conf: dict[str, Any], seeds: list[int] | None = None) -> list[int]:
    if seeds is not None and len(seeds) > 0:
        return [int(x) for x in seeds]
    data_seeds = conf.get("data", {}).get("split_seed_list", [])
    if isinstance(data_seeds, list) and len(data_seeds) > 0:
        return [int(x) for x in data_seeds]
    project_seeds = conf.get("project", {}).get("seed_list", [])
    if isinstance(project_seeds, list) and len(project_seeds) > 0:
        return [int(x) for x in project_seeds]
    return [1988]


def export_benchmark_scoreboard(
    cfg: str | Path | dict[str, Any] = "configs/base.yaml",
    seeds: list[int] | None = None,
    split: str = "test",
    ensure_predictions: bool = True,
) -> dict[str, str]:
    conf = _cfg(cfg)
    if not _is_benchmark(conf):
        raise ValueError("export_benchmark_scoreboard requires data.protocol=benchmark.")

    seed_list = _profile_seed_list(conf, seeds=seeds)
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for seed in seed_list:
        prepare_data_manifests(conf, seed=seed)
        if ensure_predictions:
            pred_path = Path(run_split_inference(conf, seed=seed, split=split))
            eval_out = evaluate_pipeline_outputs(conf, seed=seed, split=split, predictions_csv=pred_path)
        else:
            eval_out = evaluate_pipeline_outputs(conf, seed=seed, split=split, predictions_csv=None)

        metrics_df = pd.read_csv(eval_out["metrics"])
        all_df = metrics_df[metrics_df["scope"].astype(str).str.lower() == "all"].copy()
        if len(all_df) == 0:
            raise RuntimeError(f"No scope='all' metrics found for seed={seed}, split={split}")
        all_row = all_df.iloc[0].to_dict()
        all_row["seed"] = int(seed)
        metric_rows.append(all_row)

        per_class_df = pd.read_csv(eval_out["per_class"])
        per_class_df["seed"] = int(seed)
        per_class_df["run_id"] = str(all_row.get("run_id", ""))
        per_class_rows.append(per_class_df)

    metrics_by_seed = pd.DataFrame(metric_rows)
    per_class_all = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()

    summary_metrics = ["accuracy", "roc_auc_ovr_macro", "f1_macro", "precision_macro", "recall_macro", "qwk"]
    summary_rows: list[dict[str, Any]] = []
    for name in summary_metrics:
        stat = _ci95_summary(metrics_by_seed[name].astype(float).tolist()) if name in metrics_by_seed.columns else _ci95_summary([])
        summary_rows.append(
            {
                "metric": name,
                "n_seeds": int(stat["n"]),
                "mean": float(stat["mean"]),
                "std": float(stat["std"]),
                "ci95_low": float(stat["ci95_low"]),
                "ci95_high": float(stat["ci95_high"]),
            }
        )

    per_class_summary_rows: list[dict[str, Any]] = []
    if len(per_class_all):
        for (class_id, class_name), grp in per_class_all.groupby(["class_id", "class_name"], dropna=False):
            recall_stat = _ci95_summary(grp["recall"].astype(float).tolist())
            per_class_summary_rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "n_seeds": int(recall_stat["n"]),
                    "recall_mean": float(recall_stat["mean"]),
                    "recall_std": float(recall_stat["std"]),
                    "recall_ci95_low": float(recall_stat["ci95_low"]),
                    "recall_ci95_high": float(recall_stat["ci95_high"]),
                }
            )

    tables_dir = Path(conf["paths"]["tables_dir"])
    profile_tag = _profile_profile_tag(conf)
    seed_metrics_path = tables_dir / f"{profile_tag}_seed_metrics_{split}.csv"
    scoreboard_path = tables_dir / f"{profile_tag}_scoreboard_{split}.csv"
    per_class_path = tables_dir / f"{profile_tag}_per_class_recall_{split}.csv"
    criteria_path = tables_dir / f"{profile_tag}_success_criteria_{split}.json"

    metrics_by_seed.to_csv(seed_metrics_path, index=False)
    pd.DataFrame(summary_rows).to_csv(scoreboard_path, index=False)
    pd.DataFrame(per_class_summary_rows).to_csv(per_class_path, index=False)

    def _metric_mean(name: str) -> float:
        row = next((r for r in summary_rows if r["metric"] == name), None)
        return float(row["mean"]) if row is not None else float("nan")

    criteria_payload = {
        "seed_list": [int(s) for s in seed_list],
        "split": str(split),
        "targets": {
            "accuracy_mean_min": 0.80,
            "roc_auc_ovr_macro_mean_min": 0.93,
            "qwk_mean_min": 0.75,
            "class_2_3_4_recall_min": 0.55,
        },
        "observed": {
            "accuracy_mean": _metric_mean("accuracy"),
            "roc_auc_ovr_macro_mean": _metric_mean("roc_auc_ovr_macro"),
            "qwk_mean": _metric_mean("qwk"),
            "class_recall_summary_path": str(per_class_path.resolve()),
        },
    }
    _save_json(criteria_path, criteria_payload)

    return {
        "benchmark_seed_metrics": str(seed_metrics_path),
        "benchmark_scoreboard": str(scoreboard_path),
        "benchmark_per_class_recall": str(per_class_path),
        "benchmark_success_criteria": str(criteria_path),
    }


# -----------------------------
# High-level orchestration
# -----------------------------


def run_complete_workflow(cfg_path: str | Path = "configs/base.yaml", seed: int = 1988) -> dict[str, str]:
    conf = _cfg(cfg_path)

    manifests = prepare_data_manifests(conf, seed=seed)
    ckpt = train_dr_classifier(conf, seed=seed, manifests=manifests)
    ckpt_path, run_id = _resolve_checkpoint_and_run_id(
        conf,
        seed=seed,
        checkpoint=ckpt,
        require_existing=True,
    )
    calib = build_validation_calibration_table(conf, seed=seed, manifests=manifests, checkpoint=ckpt)
    pred = run_split_inference(conf, seed=seed, split="test")
    xai_out = run_xai_analysis(
        conf,
        seed=seed,
        split="test",
        checkpoint=ckpt_path,
        predictions_csv=pred,
        shap_mode=str(conf["xai"].get("shap_mode", "subset")),
        shap_max_samples=int(conf["xai"].get("shap_max_samples", 128)),
    )
    eval_out = evaluate_pipeline_outputs(conf, seed=seed, split="test", predictions_csv=pred)

    out = {
        "run_id": run_id,
        "manifests": str(Path(conf["paths"]["manifests_dir"])),
        "checkpoint": str(ckpt_path),
        "calibration": str(calib),
        "predictions": str(pred),
        **xai_out,
        **eval_out,
    }
    run_log = _workflow_run_log_path(conf, run_id)
    _save_json(run_log, out)
    if _use_legacy_aliases(conf):
        _write_alias_copy(run_log, _legacy_run_log_alias_path(conf, seed))
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)
    out["run_log"] = str(run_log)
    return out


def run_benchmark_experiments(
    cfg_path: str | Path = "configs/base.yaml",
    seeds: list[int] | None = None,
    split: str = "test",
    run_xai: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    conf = _cfg(cfg_path)
    if not _is_benchmark(conf):
        raise ValueError("run_benchmark_experiments requires data.protocol=benchmark.")

    seed_list = _profile_seed_list(conf, seeds=seeds)
    runs: list[dict[str, Any]] = []

    for seed in seed_list:
        manifests = prepare_data_manifests(conf, seed=seed)
        ckpt = train_dr_classifier(conf, seed=seed, manifests=manifests, reuse_if_exists=(not force_retrain))
        build_validation_calibration_table(conf, seed=seed, manifests=manifests, checkpoint=ckpt)
        pred = run_split_inference(conf, seed=seed, split=split)
        eval_out = evaluate_pipeline_outputs(conf, seed=seed, split=split, predictions_csv=pred)

        run_payload: dict[str, Any] = {
            "seed": int(seed),
            "checkpoint": str(ckpt),
            "predictions": str(pred),
            **eval_out,
        }
        if run_xai:
            run_payload.update(
                run_xai_analysis(
                    conf,
                    seed=seed,
                    split=split,
                    checkpoint=ckpt,
                    predictions_csv=pred,
                    shap_mode=str(conf["xai"].get("shap_mode", "subset")),
                    shap_max_samples=int(conf["xai"].get("shap_max_samples", 128)),
                )
            )
        runs.append(run_payload)

    scoreboard = export_benchmark_scoreboard(conf, seeds=seed_list, split=split, ensure_predictions=False)
    return {
        "protocol": "benchmark",
        "split": split,
        "seed_list": [int(s) for s in seed_list],
        "runs": runs,
        **scoreboard,
    }


def predict_single_image_with_explanations(cfg_path: str | Path = "configs/base.yaml", seed: int = 1988, image_path: str = "") -> dict[str, Any]:
    conf = _cfg(cfg_path)
    if not image_path:
        raise ValueError("image_path is required")

    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration table not found: {calibration_path}. Run calibration first.")

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))

    image, tensor = _load_image_for_inference(
        image_path,
        image_size=image_size,
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / max(1e-4, temperature)
        probs = torch.softmax(logits, dim=1)
        conf_score, pred = torch.max(probs, dim=1)

    confidence = float(conf_score.item())
    pred_class = int(pred.item())

    artifact = ""
    warning_message = ""
    try:
        out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_gradcam.png"
        artifact, _ = _generate_gradcam(
            model=model,
            input_tensor=tensor,
            original_image=np.array(image),
            class_id=pred_class,
            layer_name=str(conf["xai"].get("gradcam_layer", "layer4")),
            output_path=out_path,
            device=device,
            overlay_dpi=fig_dpi,
            backbone_hint=backbone,
        )
    except Exception as exc:
        warning_message = f"Grad-CAM unavailable for backbone={backbone}: {exc}"

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "seed": int(seed),
        "run_id": run_id,
        "device": str(device),
        "pred_class": pred_class,
        "confidence": confidence,
        "artifact_path": artifact,
        "warning_message": warning_message,
    }

    probs_list = probs.squeeze(0).detach().cpu().numpy().tolist()
    for i, p in enumerate(probs_list):
        result[f"prob_{i}"] = float(p)

    return result


def explain_single_image_detailed(
    cfg_path: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    image_path: str = "",
    gradcam_layers: list[str] | None = None,
    shap_background_size: int | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_path)
    if not image_path:
        raise ValueError("image_path is required")

    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    pass_border_ratio_max, pass_faith_delta_min = _xai_pass_rule_thresholds(conf)
    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration table not found: {calibration_path}. Run calibration first.")

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))

    image, tensor = _load_image_for_inference(
        image_path,
        image_size=image_size,
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / max(1e-4, temperature)
        probs = torch.softmax(logits, dim=1)
        conf_score, pred = torch.max(probs, dim=1)

    confidence = float(conf_score.item())
    pred_class = int(pred.item())

    warnings: list[str] = []
    gradcam_artifacts: dict[str, str] = {}
    gradcam_details: list[dict[str, Any]] = []
    default_gradcam_artifact = ""

    if gradcam_layers is None or len(gradcam_layers) == 0:
        gradcam_layers = ["layer2", "layer3", str(conf["xai"].get("gradcam_layer", "layer4"))]
    # preserve order while removing duplicates
    seen_layers: set[str] = set()
    unique_layers: list[str] = []
    for layer in gradcam_layers:
        key = str(layer).strip().lower()
        if not key or key in seen_layers:
            continue
        seen_layers.add(key)
        unique_layers.append(key)

    for layer in unique_layers:
        out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_gradcam_{layer}.png"
        try:
            artifact, heat_input = _generate_gradcam(
                model=model,
                input_tensor=tensor,
                original_image=np.array(image),
                class_id=pred_class,
                layer_name=layer,
                output_path=out_path,
                device=device,
                overlay_dpi=fig_dpi,
                backbone_hint=backbone,
            )
            if not default_gradcam_artifact:
                default_gradcam_artifact = artifact
            ratios = _attribution_mass_ratios(heat_input)
            faith = _faithfulness_delta(
                model=model,
                image_tensor=tensor,
                score_map=heat_input,
                pred_class=pred_class,
                top_k_ratio=0.20,
            )
            pass_flag = _xai_pass_flag(
                border_ratio=float(ratios["border_ratio"]),
                faith_delta=float(faith),
                border_ratio_max=pass_border_ratio_max,
                faith_delta_min=pass_faith_delta_min,
            )
            gradcam_artifacts[layer] = artifact
            gradcam_details.append(
                {
                    "layer": layer,
                    "artifact_path": artifact,
                    "border_ratio": float(ratios["border_ratio"]),
                    "retina_ratio": float(ratios["retina_ratio"]),
                    "faithfulness_delta": float(faith),
                    "gradcam_pass": pass_flag,
                }
            )
        except Exception as exc:
            warnings.append(f"Grad-CAM failed for layer={layer}: {exc}")

    shap_artifact_path = ""
    shap_details: dict[str, Any] = {
        "status": "skipped",
        "artifact_path": "",
        "border_ratio": np.nan,
        "retina_ratio": np.nan,
        "faithfulness_delta": np.nan,
        "shap_pass": np.nan,
        "error": "",
        "attempted_device": str(device),
        "final_device": str(device),
        "fallback_used": False,
        "error_primary": "",
    }

    try:
        if _SHAP_IMPORT_ERROR is not None:
            raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")

        bg_size = int(shap_background_size or conf["xai"].get("shap_background_size", 64))
        bg_size = max(1, bg_size)

        train_manifest = _manifest_path(conf, "train", seed=seed)
        if not train_manifest.exists():
            prepare_data_manifests(conf, seed=seed)
        train_df = pd.read_csv(train_manifest)
        if len(train_df) == 0:
            raise RuntimeError("Train manifest is empty; cannot build SHAP background")

        per_class = max(1, bg_size // int(conf["data"]["num_classes"]))
        bg_df = train_df.groupby("class_id", group_keys=False).head(per_class)
        if len(bg_df) < bg_size:
            extra = train_df.head(bg_size - len(bg_df))
            bg_df = pd.concat([bg_df, extra], ignore_index=True)
        bg_df = bg_df.head(bg_size)

        bg_tensors: list[torch.Tensor] = []
        for _, b in bg_df.iterrows():
            _, bt = _load_image_for_inference(
                b["image_path"],
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            bg_tensors.append(bt.squeeze(0))
        background_cpu = torch.stack(bg_tensors, dim=0)

        def _run_single_shap(shap_device: torch.device) -> dict[str, Any]:
            if str(shap_device) == str(device):
                shap_model = model
            else:
                shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
            _make_shap_compatible(shap_model)
            wrapper = _LogitWrapper(shap_model).to(shap_device)
            background = background_cpu.to(shap_device)
            explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background)
            shap_input = tensor.to(shap_device)
            shap_values = _shap_values_with_known_warning_filter(explainer, shap_input)
            smap = _pick_shap_map(shap_values, class_id=pred_class, sample_index=0)
            s2d = _shap_to_2d(smap)

            shap_out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_shap_cls{pred_class}.png"
            artifact_path = _save_map_overlay(s2d, np.array(image), shap_out_path, overlay_dpi=fig_dpi)
            shap_ratios = _attribution_mass_ratios(s2d)
            shap_faith = _faithfulness_delta(
                model=shap_model,
                image_tensor=shap_input,
                score_map=s2d,
                pred_class=pred_class,
                top_k_ratio=0.20,
            )
            shap_pass = _xai_pass_flag(
                border_ratio=float(shap_ratios["border_ratio"]),
                faith_delta=float(shap_faith),
                border_ratio_max=pass_border_ratio_max,
                faith_delta_min=pass_faith_delta_min,
            )
            return {
                "status": "success",
                "artifact_path": artifact_path,
                "border_ratio": float(shap_ratios["border_ratio"]),
                "retina_ratio": float(shap_ratios["retina_ratio"]),
                "faithfulness_delta": float(shap_faith),
                "shap_pass": shap_pass,
                "error": "",
                "attempted_device": str(device),
                "final_device": str(shap_device),
                "fallback_used": bool(str(shap_device) != str(device)),
                "error_primary": "",
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
            }

        primary_error = ""
        try:
            shap_details = _run_single_shap(device)
        except Exception as primary_exc:
            primary_error = str(primary_exc)
            if _should_retry_shap_on_cpu(primary_exc, device):
                _empty_mps_cache_if_available()
                try:
                    shap_details = _run_single_shap(torch.device("cpu"))
                    shap_details["fallback_used"] = True
                    shap_details["error_primary"] = primary_error
                    warnings.append(f"SHAP fallback used CPU after MPS failure: {primary_error}")
                except Exception as fallback_exc:
                    raise RuntimeError(f"{fallback_exc} (primary_mps_error={primary_error})") from fallback_exc
            else:
                raise
        shap_artifact_path = str(shap_details.get("artifact_path", ""))
        if str(shap_details.get("final_device", "")) == "mps":
            _empty_mps_cache_if_available()
            gc.collect()
    except Exception as exc:
        shap_details = {
            "status": "failed",
            "artifact_path": "",
            "border_ratio": np.nan,
            "retina_ratio": np.nan,
            "faithfulness_delta": np.nan,
            "shap_pass": np.nan,
            "error": str(exc),
            "attempted_device": str(device),
            "final_device": str(device),
            "fallback_used": False,
            "error_primary": "",
            "pass_rule_border_ratio_max": float(pass_border_ratio_max),
            "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
        }
        warnings.append(f"SHAP failed: {exc}")

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "seed": int(seed),
        "run_id": run_id,
        "device": str(device),
        "pred_class": pred_class,
        "confidence": confidence,
        "artifact_path": default_gradcam_artifact,
        "warning_message": "; ".join(warnings),
        "gradcam_artifacts": gradcam_artifacts,
        "gradcam_details": gradcam_details,
        "shap_artifact_path": shap_artifact_path,
        "shap_details": shap_details,
        "xai_warnings": warnings,
    }

    probs_list = probs.squeeze(0).detach().cpu().numpy().tolist()
    for i, p in enumerate(probs_list):
        result[f"prob_{i}"] = float(p)

    return result


def run_single_case_demo(
    cfg_path: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    image_path: str = "",
    split: str = "test",
    gradcam_layers: list[str] | None = None,
    shap_background_size: int | None = None,
    shap_panel_size: tuple[float, float] = (3.2, 3.6),
    shap_dpi: int | None = None,
) -> dict[str, Any]:
    """
    Run a compact single-case demo flow for notebooks:
    1) pick one image (or use provided image_path),
    2) run detailed Grad-CAM + SHAP,
    3) export Grad-CAM panel and SHAP class grid,
    4) return a concise summary payload.
    """
    conf = _cfg(cfg_path)
    split_key = str(split).strip().lower() or "test"
    if split_key not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got: {split}")

    manifest_path = _manifest_path(conf, split_key, seed=seed)
    if not manifest_path.exists():
        prepare_data_manifests(conf, seed=seed)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for split={split_key}: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if len(manifest_df) == 0:
        raise RuntimeError(f"Empty manifest for split={split_key}: {manifest_path}")

    sampled_row: pd.Series | None = None
    if not image_path:
        sampled_row = manifest_df.sample(n=1, random_state=int(seed)).iloc[0]
        image_path = str(sampled_row["image_path"])

    img_path = Path(str(image_path))
    if not img_path.is_absolute():
        img_path = (Path(conf["project_root"]) / img_path).resolve()
    else:
        img_path = img_path.resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    true_class: int | None = None
    if sampled_row is not None:
        true_class = int(sampled_row["class_id"])
    else:
        # Best-effort lookup of true class from manifest for correctness border/captions.
        target = str(img_path)
        for _, row in manifest_df.iterrows():
            row_path = Path(str(row["image_path"]))
            if not row_path.is_absolute():
                row_path = (Path(conf["project_root"]) / row_path).resolve()
            else:
                row_path = row_path.resolve()
            if str(row_path) == target:
                true_class = int(row["class_id"])
                break

    requested_layers = gradcam_layers if gradcam_layers is not None else ["layer2", "layer3", str(conf["xai"].get("gradcam_layer", "layer4"))]
    seen_layers: set[str] = set()
    gradcam_layer_order: list[str] = []
    for layer in requested_layers:
        layer_key = str(layer).strip().lower()
        if not layer_key or layer_key in seen_layers:
            continue
        seen_layers.add(layer_key)
        gradcam_layer_order.append(layer_key)
    if not gradcam_layer_order:
        gradcam_layer_order = ["layer4"]

    single_result = explain_single_image_detailed(
        cfg_path=conf,
        seed=seed,
        image_path=str(img_path),
        gradcam_layers=gradcam_layer_order,
        shap_background_size=shap_background_size,
    )

    num_classes = int(conf["data"]["num_classes"])
    label_order = list(conf["data"]["label_order"])
    prob_table = pd.DataFrame(
        {
            "class_id": list(range(num_classes)),
            "class_name": label_order,
            "probability": [float(single_result.get(f"prob_{i}", 0.0)) for i in range(num_classes)],
        }
    ).sort_values("probability", ascending=False).reset_index(drop=True)

    pred_class = int(single_result.get("pred_class", -1))
    pred_label = label_order[pred_class] if 0 <= pred_class < len(label_order) else str(pred_class)
    true_label = label_order[true_class] if (true_class is not None and 0 <= true_class < len(label_order)) else None

    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    single_dir = Path(conf["paths"]["figures_dir"]) / "single"
    single_dir.mkdir(parents=True, exist_ok=True)

    # Build one compact Grad-CAM panel: [Input | layer2 | layer3 | layer4].
    gradcam_panel_path = single_dir / f"{img_path.stem}_gradcam_panel.png"
    raw = Image.open(img_path).convert("RGB")
    model_view = _apply_fundus_preprocessing(
        raw,
        image_size=int(_model_image_size(conf)),
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    panels: list[tuple[str, np.ndarray]] = [("Input (Model View)", np.array(model_view))]
    for layer in gradcam_layer_order:
        artifact = str((single_result.get("gradcam_artifacts") or {}).get(layer, ""))
        if artifact and Path(artifact).exists():
            panels.append((f"Grad-CAM {layer}", np.asarray(plt.imread(artifact))))

    fig_g, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), facecolor="white")
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image_arr) in zip(axes, panels):
        ax.imshow(image_arr)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig_g.tight_layout()
    fig_g.savefig(gradcam_panel_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig_g)

    bg_size = int(max(1, shap_background_size if shap_background_size is not None else conf.get("xai", {}).get("shap_background_size", 64)))
    out_dpi = int(shap_dpi if shap_dpi is not None else fig_dpi)
    shap_grid_path = single_dir / f"{img_path.stem}_shap_grid.png"
    fig_s = plot_shap_grid(
        cfg_path=conf,
        seed=seed,
        image_paths=[str(img_path)],
        background_size=bg_size,
        vmax_percentile=99.5,
        save_path=shap_grid_path,
        dpi=out_dpi,
        true_classes=[int(true_class)] if true_class is not None else None,
        show_correctness_border=bool(true_class is not None),
        panel_size=shap_panel_size,
    )
    plt.close(fig_s)

    return {
        "manifest_path": str(manifest_path),
        "image_path": str(img_path),
        "split": split_key,
        "seed": int(seed),
        "run_id": str(single_result.get("run_id", "")),
        "device": str(single_result.get("device", "")),
        "true_class": int(true_class) if true_class is not None else None,
        "true_label": true_label,
        "pred_class": pred_class,
        "pred_label": pred_label,
        "confidence": float(single_result.get("confidence", float("nan"))),
        "prob_table": prob_table,
        "gradcam_panel_path": str(gradcam_panel_path),
        "shap_grid_path": str(shap_grid_path),
        "single_result": single_result,
        "xai_warnings": list(single_result.get("xai_warnings") or []),
        "shap_background_size": bg_size,
    }


@torch.no_grad()
def _predict_one_with_temperature(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> tuple[int, float, np.ndarray]:
    logits = model(input_tensor.to(device))
    logits = logits / max(1e-4, float(temperature))
    probs = torch.softmax(logits, dim=1)
    conf_score, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf_score.item()), probs.squeeze(0).detach().cpu().numpy()


def _temperature_for_run(conf: dict[str, Any], run_id: str) -> float:
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        return 1.0
    payload = _load_json(calibration_path)
    return float(payload.get("temperature", 1.0))


def _gradcam_heatmap_for_display(
    model: DRClassifier,
    input_tensor: torch.Tensor,
    class_id: int,
    layer_name: str,
    output_size: tuple[int, int],
    device: torch.device,
    backbone_hint: str = "",
) -> np.ndarray:
    if _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(f"captum import failed: {_CAPTUM_IMPORT_ERROR}")

    target_layer, _, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
        backbone_hint=backbone_hint,
    )
    if target_layer is None:
        raise ValueError(reason or f"Unable to resolve Grad-CAM layer '{layer_name}'.")

    gradcam = LayerGradCam(lambda x: model(x), target_layer)
    attr = gradcam.attribute(input_tensor.to(device), target=int(class_id))
    attr_input = LayerAttribution.interpolate(attr, (int(input_tensor.shape[-2]), int(input_tensor.shape[-1])))
    heat = attr_input.squeeze().detach().cpu().numpy()
    if heat.ndim == 3:
        heat = np.mean(heat, axis=0)
    heat = _normalize_map(heat)
    out_w, out_h = int(output_size[0]), int(output_size[1])
    heat = np.array(
        Image.fromarray(heat.astype(np.float32), mode="F").resize((out_w, out_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return _normalize_map(heat)


def plot_gradcam_grid(
    cfg_path: str | Path | dict[str, Any],
    seed: int,
    samples: list[tuple[str, int]],
    ncols: int = 2,
    gradcam_layer: str = "layer4",
    cmap_name: str = "jet",
    alpha: float = 0.45,
    figsize_per_pair: tuple[float, float] = (5.0, 2.8),
    save_path: str | Path | None = None,
    dpi: int | None = None,
) -> plt.Figure:
    """
    Publication-style Grad-CAM grid:
    [Original | Grad-CAM] pairs with green/red border by correctness.
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)
    _, _, gradcam_reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=gradcam_layer,
        backbone_hint=backbone,
    )
    if gradcam_reason:
        raise ValueError(f"plot_gradcam_grid cannot run Grad-CAM for backbone={backbone}: {gradcam_reason}")

    n = len(samples)
    cols = max(1, int(ncols))
    nrows = int(np.ceil(n / cols))

    fig_w = float(figsize_per_pair[0]) * cols * 2
    fig_h = float(figsize_per_pair[1]) * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    outer = gridspec.GridSpec(nrows, cols, figure=fig, hspace=0.45, wspace=0.08)

    for idx, (image_path, true_class) in enumerate(samples):
        row, col = divmod(idx, cols)
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.03)
        ax_orig = fig.add_subplot(inner[0])
        ax_cam = fig.add_subplot(inner[1])

        image_pil, tensor = _load_image_for_inference(
            image_path,
            image_size=image_size,
            preprocessing_cfg=conf.get("preprocessing", {}),
        )
        pred_class, confidence, _ = _predict_one_with_temperature(model, tensor, device, temperature)
        base = np.array(image_pil)
        heat = _gradcam_heatmap_for_display(
            model=model,
            input_tensor=tensor,
            class_id=pred_class,
            layer_name=gradcam_layer,
            output_size=(base.shape[1], base.shape[0]),
            device=device,
            backbone_hint=backbone,
        )
        overlay = _overlay(base, heat, alpha=float(alpha), cmap_name=str(cmap_name))

        ax_orig.imshow(base)
        ax_orig.set_title("Original Image", fontsize=7.5, pad=3)
        ax_orig.axis("off")

        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM", fontsize=7.5, pad=3)
        ax_cam.axis("off")

        correct = int(pred_class) == int(true_class)
        border_color = "#27ae60" if correct else "#e74c3c"
        p1 = ax_orig.get_position()
        p2 = ax_cam.get_position()
        x0 = min(p1.x0, p2.x0)
        y0 = min(p1.y0, p2.y0)
        x1 = max(p1.x1, p2.x1)
        y1 = max(p1.y1, p2.y1)
        pad = 0.004
        rect = mpatches.Rectangle(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            fill=False,
            edgecolor=border_color,
            linewidth=1,
            transform=fig.transFigure,
            clip_on=False,
            zorder=20,
        )
        fig.add_artist(rect)

        true_label = label_order[int(true_class)] if 0 <= int(true_class) < len(label_order) else str(true_class)
        pred_label = label_order[int(pred_class)] if 0 <= int(pred_class) < len(label_order) else str(pred_class)
        inner_ax = fig.add_subplot(outer[row, col])
        inner_ax.set_axis_off()
        inner_ax.text(
            0.5,
            -0.10,
            (
                f"True label: {true_label} ({int(true_class)}),  "
                f"Predicted label: {pred_label} ({int(pred_class)}),  "
                f"Pred prob: {float(confidence):.3f}"
            ),
            transform=inner_ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color="black",
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(facecolor="#27ae60", label="Correct prediction"),
        mpatches.Patch(facecolor="#e74c3c", label="Wrong prediction"),
    ]
    fig.subplots_adjust(bottom=0.16)
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(
        0.5,
        0.005,
        "Figure: Grad-CAM Visualization. Green border = correct prediction, Red border = wrong prediction.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#444",
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=out_dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    return fig


def plot_shap_grid(
    cfg_path: str | Path | dict[str, Any],
    seed: int,
    image_paths: list[str],
    background_size: int | None = None,
    vmax_percentile: float = 99.5,
    shap_cmap: str = "RdBu_r",
    bg_cmap: str = "gray",
    bg_alpha: float = 0.35,
    shap_alpha: float = 0.85,
    save_path: str | Path | None = None,
    dpi: int | None = None,
    true_classes: list[int] | None = None,
    show_correctness_border: bool = False,
    panel_size: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot SHAP per class for each input image.
    Layout: one row per image, columns [original, class0, class1, ...].
    """
    if _SHAP_IMPORT_ERROR is not None or shap is None:
        raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")
    if len(image_paths) == 0:
        raise ValueError("image_paths must not be empty")
    if true_classes is not None and len(true_classes) != len(image_paths):
        raise ValueError(
            f"true_classes length ({len(true_classes)}) must match image_paths length ({len(image_paths)})."
        )

    from matplotlib.colors import TwoSlopeNorm

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    num_classes = int(conf["data"]["num_classes"])
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)

    train_df = pd.read_csv(_manifest_path(conf, "train", seed=seed))
    bg_default = int(conf.get("xai", {}).get("shap_background_size", 64))
    bg_size = int(max(1, background_size if background_size is not None else bg_default))
    per_class = max(1, bg_size // max(1, num_classes))
    bg_df = train_df.groupby("class_id", group_keys=False).head(per_class)
    if len(bg_df) < bg_size:
        bg_df = pd.concat([bg_df, train_df.head(bg_size - len(bg_df))], ignore_index=True)
    bg_df = bg_df.head(bg_size)

    bg_tensors: list[torch.Tensor] = []
    for _, row in bg_df.iterrows():
        _, bt = _load_image_for_inference(
            row["image_path"],
            image_size=image_size,
            preprocessing_cfg=conf.get("preprocessing", {}),
        )
        bg_tensors.append(bt.squeeze(0))
    background_cpu = torch.stack(bg_tensors, dim=0)

    def _collect_rows_for_device(shap_device: torch.device) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        if str(shap_device) == str(device):
            shap_model = model
        else:
            shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
        _make_shap_compatible(shap_model)
        wrapper = _LogitWrapper(shap_model).to(shap_device)
        explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background_cpu.to(shap_device))

        rows: list[dict[str, Any]] = []
        vals: list[np.ndarray] = []
        for img_idx, image_path in enumerate(image_paths):
            image_pil, tensor = _load_image_for_inference(
                image_path,
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            pred_class, confidence, _ = _predict_one_with_temperature(model, tensor, device, temperature)
            shap_values = _shap_values_with_known_warning_filter(explainer, tensor.to(shap_device))

            class_maps: list[np.ndarray] = []
            for cls_idx in range(num_classes):
                smap = _pick_shap_map(shap_values, class_id=cls_idx, sample_index=0)
                arr = np.asarray(smap, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(0)
                class_maps.append(arr)
                vals.append(np.abs(arr.ravel()))

            rows.append(
                {
                    "image_path": str(image_path),
                    "image_pil": image_pil,
                    "pred_class": int(pred_class),
                    "true_class": int(true_classes[img_idx]) if true_classes is not None else None,
                    "confidence": float(confidence),
                    "class_maps": class_maps,
                }
            )
            del shap_values, class_maps
            if getattr(shap_device, "type", "") == "mps":
                _empty_mps_cache_if_available()
                gc.collect()
        return rows, vals

    shap_final_device = str(device)
    shap_fallback_used = False
    shap_primary_error = ""
    try:
        rows_data, all_vals = _collect_rows_for_device(device)
    except Exception as primary_exc:
        shap_primary_error = str(primary_exc)
        if _should_retry_shap_on_cpu(primary_exc, device):
            _empty_mps_cache_if_available()
            try:
                rows_data, all_vals = _collect_rows_for_device(torch.device("cpu"))
                shap_final_device = "cpu"
                shap_fallback_used = True
            except Exception as fallback_exc:
                raise RuntimeError(f"{fallback_exc} (primary_mps_error={shap_primary_error})") from fallback_exc
        else:
            raise

    vmax = 1e-6
    if all_vals:
        vmax = max(float(np.percentile(np.concatenate(all_vals), float(vmax_percentile))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap(str(shap_cmap))

    nrows = len(rows_data)
    ncols = 1 + num_classes
    if panel_size is None:
        panel_w, panel_h = 2.2, 2.8
    else:
        panel_w = float(panel_size[0])
        panel_h = float(panel_size[1])
        panel_w = max(1.5, panel_w)
        panel_h = max(2.0, panel_h)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), facecolor="white")
    axes = np.array(axes).reshape(nrows, ncols)

    im = None
    row_border_specs: list[tuple[int, str]] = []
    for r, row in enumerate(rows_data):
        image_pil = row["image_pil"]
        pred_class = int(row["pred_class"])
        true_class_raw = row.get("true_class")
        true_class = int(true_class_raw) if true_class_raw is not None else None
        confidence = float(row["confidence"])
        class_maps = row["class_maps"]

        axes[r, 0].imshow(np.array(image_pil.convert("L")), cmap=str(bg_cmap))
        axes[r, 0].axis("off")
        axes[r, 0].set_title("Original", fontsize=8)
        if true_class is None:
            caption = f"pred={label_order[pred_class]} conf={confidence:.3f}"
        else:
            true_label = label_order[int(true_class)] if 0 <= int(true_class) < len(label_order) else str(true_class)
            pred_label = label_order[int(pred_class)] if 0 <= int(pred_class) < len(label_order) else str(pred_class)
            is_correct = int(pred_class) == int(true_class)
            verdict = "correct" if is_correct else "wrong"
            caption = f"true={true_label} pred={pred_label} ({verdict}) conf={confidence:.3f}"
            if show_correctness_border:
                row_border_specs.append((r, "#27ae60" if is_correct else "#e74c3c"))
        axes[r, 0].text(
            0.5,
            -0.08,
            caption,
            transform=axes[r, 0].transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )

        for cls_idx, smap in enumerate(class_maps):
            ax = axes[r, cls_idx + 1]
            bg = np.array(
                image_pil.convert("L").resize((smap.shape[1], smap.shape[0]), Image.Resampling.BILINEAR)
            )
            ax.imshow(bg, cmap=str(bg_cmap), alpha=float(bg_alpha), vmin=0, vmax=255, interpolation="bilinear")
            im = ax.imshow(smap, cmap=cmap, norm=norm, alpha=float(shap_alpha), interpolation="nearest")
            ax.axis("off")
            score = 1.0 if cls_idx == pred_class else 0.0
            ax.set_title(f"{score:.1f}", fontsize=8, color="#e74c3c" if cls_idx == pred_class else "#333")

    if im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.02])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation="horizontal")
        cb.set_label("SHAP value", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Avoid tight_layout() here because we manually add a colorbar axes.
    # tight_layout triggers a noisy warning and can produce unstable geometry.
    bottom_margin = 0.09 if im is not None else 0.05
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=bottom_margin, wspace=0.02, hspace=0.32)
    if show_correctness_border and row_border_specs:
        for row_idx, border_color in row_border_specs:
            p_first = axes[row_idx, 0].get_position()
            p_last = axes[row_idx, ncols - 1].get_position()
            x0 = min(p_first.x0, p_last.x0)
            y0 = min(p_first.y0, p_last.y0)
            x1 = max(p_first.x1, p_last.x1)
            y1 = max(p_first.y1, p_last.y1)
            pad = 0.0025
            rect = mpatches.Rectangle(
                (x0 - pad, y0 - pad),
                (x1 - x0) + 2 * pad,
                (y1 - y0) + 2 * pad,
                fill=False,
                edgecolor=border_color,
                linewidth=1,
                transform=fig.transFigure,
                clip_on=False,
                zorder=20,
            )
            fig.add_artist(rect)
        fig.text(
            0.5,
            0.005,
            "Green border = correct prediction, Red border = wrong prediction.",
            ha="center",
            fontsize=8,
            style="italic",
            color="#444",
        )
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=out_dpi, bbox_inches="tight", facecolor="white")
    if shap_fallback_used:
        print(f"SHAP fallback used: attempted_device={device}, final_device={shap_final_device}, primary_error={shap_primary_error}")
    return fig


def notebook_prepare_data_overview(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    samples_per_split: int = 3,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    manifests = prepare_data_manifests(conf, seed=int(seed))

    train_df = pd.read_csv(manifests["train_split"])
    val_df = pd.read_csv(manifests["val_split"])
    test_df = pd.read_csv(manifests["test_full"])

    total_rows = int(len(train_df) + len(val_df) + len(test_df))
    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": int(len(train_df))},
            {"split": "val", "rows": int(len(val_df))},
            {"split": "test", "rows": int(len(test_df))},
        ]
    )
    if total_rows > 0:
        split_summary["percentage"] = (split_summary["rows"] / float(total_rows)) * 100.0
    else:
        split_summary["percentage"] = np.nan
    split_summary["percentage"] = split_summary["percentage"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    label_order = list(conf["data"]["label_order"])
    class_map = {i: name for i, name in enumerate(label_order)}
    all_df = pd.concat(
        [
            train_df.assign(view="train"),
            val_df.assign(view="val"),
            test_df.assign(view="test"),
        ],
        ignore_index=True,
    )

    counts = (
        all_df.groupby(["view", "class_id"])
        .size()
        .reset_index(name="count")
    )
    counts["class_name"] = counts["class_id"].map(class_map)
    split_totals = counts.groupby("view")["count"].transform("sum")
    counts["split_pct"] = np.where(
        split_totals > 0,
        (counts["count"] / split_totals) * 100.0,
        np.nan,
    )
    counts["split_pct"] = counts["split_pct"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    overall = (
        all_df.groupby("class_id")
        .size()
        .reset_index(name="count")
    )
    overall["class_name"] = overall["class_id"].map(class_map)
    if len(all_df) > 0:
        overall["overall_pct"] = (overall["count"] / float(len(all_df))) * 100.0
    else:
        overall["overall_pct"] = np.nan
    overall["overall_pct"] = overall["overall_pct"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    source_summary = pd.DataFrame()
    if "source_dataset" in all_df.columns:
        source_summary = (
            all_df.groupby(["view", "source_dataset"])
            .size()
            .reset_index(name="rows")
        )
        source_summary["view_total"] = source_summary.groupby("view")["rows"].transform("sum")
        source_summary["view_pct"] = np.where(
            source_summary["view_total"] > 0,
            (source_summary["rows"] / source_summary["view_total"]) * 100.0,
            np.nan,
        )
        source_summary["view_pct"] = source_summary["view_pct"].map(
            lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
        )
        source_summary = source_summary.sort_values(["view", "source_dataset"]).reset_index(drop=True)

    pivot_counts = (
        counts.pivot(index="class_name", columns="view", values="count")
        .fillna(0)
        .reindex(label_order)
    )
    fig_counts, ax_counts = plt.subplots(figsize=(9, 4))
    pivot_counts.plot(kind="bar", ax=ax_counts)
    ax_counts.set_title("Class Distribution by Split (Counts)")
    ax_counts.set_ylabel("Count")
    ax_counts.tick_params(axis="x", rotation=20)
    fig_counts.tight_layout()

    pivot_prop = (
        counts.assign(split_pct_num=pd.to_numeric(counts["split_pct"].str.rstrip("%"), errors="coerce"))
        .pivot(index="class_name", columns="view", values="split_pct_num")
        .fillna(0.0)
        .reindex(label_order)
    )
    fig_prop, ax_prop = plt.subplots(figsize=(9, 4))
    pivot_prop.plot(kind="bar", ax=ax_prop)
    ax_prop.set_title("Class Distribution by Split (Within-Split %)")
    ax_prop.set_ylabel("Percentage (%)")
    ax_prop.tick_params(axis="x", rotation=20)
    fig_prop.tight_layout()

    samples_n = max(1, int(samples_per_split))
    split_frames: list[tuple[str, pd.DataFrame]] = [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]
    fig_samples, axes = plt.subplots(
        len(split_frames),
        samples_n,
        figsize=(4.0 * samples_n, 3.6 * len(split_frames)),
    )
    axes = np.array(axes, dtype=object)
    if axes.ndim == 1:
        if len(split_frames) == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for r, (split_name, split_df) in enumerate(split_frames):
        available = int(len(split_df))
        n = min(samples_n, available)
        sample_df = split_df.sample(n=n, random_state=int(seed)).reset_index(drop=True) if n > 0 else pd.DataFrame()
        for c in range(samples_n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= n:
                continue
            row = sample_df.iloc[c]
            img = plt.imread(str(row["image_path"]))
            class_name = class_map.get(int(row["class_id"]), str(row["class_id"]))
            ax.imshow(img)
            ax.set_title(f"{split_name} | {class_name}", fontsize=9)
    fig_samples.suptitle("Sample Fundus Images", y=1.02)
    fig_samples.tight_layout()

    return {
        "cfg": conf,
        "seed": int(seed),
        "manifest_paths": manifests,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_summary": split_summary,
        "class_distribution": counts,
        "overall_class_proportions": overall.sort_values("class_id").reset_index(drop=True),
        "source_dataset_mix": source_summary,
        "class_count_fig": fig_counts,
        "class_proportion_fig": fig_prop,
        "sample_images_fig": fig_samples,
    }


def notebook_run_training(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    manifests: dict[str, str] | None = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    reuse_if_exists = not bool(force_retrain)
    current_sig = _checkpoint_config_signature(conf)
    pre_match = _find_matching_checkpoint_by_signature(conf, seed=int(seed), config_signature=current_sig) if reuse_if_exists else None
    had_any_checkpoint = _latest_named_checkpoint(conf, int(seed)) is not None

    checkpoint_path = train_dr_classifier(
        conf,
        seed=int(seed),
        manifests=manifests,
        reuse_if_exists=reuse_if_exists,
    )

    latest_path = _latest_run_record_path(conf, int(seed))
    latest_payload = _load_json(latest_path) if latest_path.exists() else {}
    run_id = str(latest_payload.get("run_id", Path(checkpoint_path).stem))

    history_path = _train_history_log_path(conf, run_id)
    if not history_path.exists():
        fallback_history = Path(conf["paths"]["logs_dir"]) / f"train_history_seed{int(seed)}.json"
        history_path = fallback_history if fallback_history.exists() else history_path
    history = _load_json(history_path) if history_path.exists() else {}

    history_fig: plt.Figure | None = None
    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    val_macro_f1 = list(history.get("val_macro_f1", []))
    if len(train_loss) > 0 and len(val_loss) > 0:
        epochs = np.arange(1, min(len(train_loss), len(val_loss)) + 1)
        history_fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss[: len(epochs)], marker="o", label="train_loss")
        axes[0].plot(epochs, val_loss[: len(epochs)], marker="o", label="val_loss")
        axes[0].set_title("Training vs Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        f1_epochs = np.arange(1, len(val_macro_f1) + 1)
        axes[1].plot(f1_epochs, val_macro_f1, marker="o", color="tab:green")
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        history_fig.tight_layout()

    reused_checkpoint = bool(pre_match is not None and Path(pre_match[0]).resolve() == Path(checkpoint_path).resolve())
    if force_retrain:
        reuse_reason = "force_retrain"
    elif reused_checkpoint:
        reuse_reason = "matched_signature"
    elif had_any_checkpoint:
        reuse_reason = "signature_mismatch"
    else:
        reuse_reason = "no_checkpoint_found"

    return {
        "cfg": conf,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "latest_run_record_path": str(latest_path),
        "latest_run_record": latest_payload,
        "history_path": str(history_path),
        "history": history,
        "history_fig": history_fig,
        "reused_checkpoint": bool(reused_checkpoint),
        "reuse_reason": str(reuse_reason),
    }


def notebook_run_core_evaluation(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    predictions_csv: str | Path | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    predictions_path = str(predictions_csv) if predictions_csv else run_split_inference(conf, seed=int(seed), split=split_key)

    eval_outputs = evaluate_pipeline_outputs(
        conf,
        seed=int(seed),
        split=split_key,
        predictions_csv=predictions_path,
    )

    headline_path = Path(eval_outputs["final_headline_metrics"])
    metrics_path = Path(eval_outputs["metrics"])
    summary_path = Path(eval_outputs["run_summary"])
    per_class_path = Path(eval_outputs["per_class"])
    confusion_path = Path(eval_outputs["confusion"])

    if not headline_path.exists():
        _ = export_final_headline_metrics(conf, seed=int(seed), split=split_key, metrics_csv=metrics_path)

    headline_df = pd.read_csv(headline_path)
    metrics_df = pd.read_csv(metrics_path)
    summary_df = pd.read_csv(summary_path)
    per_class_df = pd.read_csv(per_class_path)
    confusion_df = pd.read_csv(confusion_path)

    headline_cols = [
        c
        for c in ["run_id", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        if c in headline_df.columns
    ]
    summary_cols = [
        c
        for c in ["split", "n_all", "qwk_all", "n_high_conf", "accuracy_high_conf", "qwk_high_conf"]
        if c in summary_df.columns
    ]
    overall_summary_df = pd.concat(
        [
            headline_df[headline_cols].reset_index(drop=True),
            summary_df[summary_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    output_paths_df = pd.DataFrame([eval_outputs]).T.rename(columns={0: "path"})

    headline_fig: plt.Figure | None = None
    if len(headline_df):
        vals = headline_df.iloc[0]
        metric_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
        metric_values = [
            float(vals["accuracy"]),
            float(vals["precision_macro"]),
            float(vals["recall_macro"]),
            float(vals["f1_macro"]),
        ]
        headline_fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.bar(metric_labels, metric_values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Headline Metrics")
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + 0.01,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        headline_fig.tight_layout()

    confusion_fig: plt.Figure | None = None
    cm = confusion_df.to_numpy()
    if cm.ndim == 2 and cm.size > 0:
        label_order = list(conf["data"]["label_order"])
        confusion_fig, ax = plt.subplots(figsize=(6, 5))
        heat = ax.imshow(cm, cmap="Blues")
        confusion_fig.colorbar(heat, ax=ax)
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels(label_order, rotation=25)
        ax.set_yticks(range(len(label_order)))
        ax.set_yticklabels(label_order)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        confusion_fig.tight_layout()

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "predictions_path": str(predictions_path),
        "eval_outputs": eval_outputs,
        "eval_output_paths_df": output_paths_df,
        "headline_df": headline_df,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "overall_summary_df": overall_summary_df,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "headline_fig": headline_fig,
        "confusion_fig": confusion_fig,
    }


def notebook_run_xai(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    manifests: dict[str, str] | None = None,
    checkpoint: str | Path | None = None,
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    split_key = str(split).strip().lower() or "test"
    xai_cfg = conf.setdefault("xai", {})

    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["max_targets"] = min(int(xai_cfg.get("max_targets", 128)), 24)
        xai_cfg["shap_max_samples"] = min(int(xai_cfg.get("shap_max_samples", 128)), 24)
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 12)
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "auto"))
    xai_cfg["log_every_samples"] = max(1, int(xai_cfg.get("log_every_samples", 16)))

    calibration_path = build_validation_calibration_table(
        conf,
        seed=int(seed),
        manifests=manifests,
        checkpoint=checkpoint,
    )
    xai_outputs = run_xai_analysis(
        conf,
        seed=int(seed),
        split=split_key,
        checkpoint=checkpoint,
        predictions_csv=None,
        shap_mode=str(xai_cfg.get("shap_mode", "subset")),
        shap_max_samples=int(xai_cfg.get("shap_max_samples", 64)),
    )

    run_id = str(xai_outputs.get("run_id", ""))
    if not run_id:
        latest_payload = _load_latest_run_record(conf, int(seed))
        if latest_payload:
            run_id = str(latest_payload.get("run_id", ""))

    pred_primary = (
        Path(conf["paths"]["predictions_dir"]) / f"{run_id}_{split_key}_predictions.csv"
        if run_id
        else Path("")
    )
    pred_alias = Path(conf["paths"]["predictions_dir"]) / f"predictions_seed{int(seed)}_{split_key}.csv"
    predictions_path = pred_primary if pred_primary.exists() else pred_alias
    predictions_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": split_key,
        "safe_mode": bool(safe_mode),
        "xai_runtime_settings": {
            "device": str(xai_cfg.get("device")),
            "max_targets": int(xai_cfg.get("max_targets", 0)),
            "shap_max_samples": int(xai_cfg.get("shap_max_samples", 0)),
            "shap_background_size": int(xai_cfg.get("shap_background_size", 0)),
            "log_every_samples": int(xai_cfg.get("log_every_samples", 1)),
        },
        "calibration_path": str(calibration_path),
        "xai_outputs": xai_outputs,
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "predictions_df": predictions_df,
    }


def notebook_load_xai_committee_summary(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    tables_dir = Path(conf["paths"]["tables_dir"])

    method_path = tables_dir / f"rq_xai_method_stats_seed{int(seed)}_{split_key}.csv"
    pairwise_path = tables_dir / f"rq_xai_pairwise_seed{int(seed)}_{split_key}.csv"
    coverage_path = tables_dir / f"xai_target_coverage_seed{int(seed)}_{split_key}.csv"
    targets_path = tables_dir / f"xai_targets_seed{int(seed)}_{split_key}.csv"
    correctness_path = tables_dir / f"rq_xai_pass_by_correctness_seed{int(seed)}_{split_key}.csv"

    method_stats_df = pd.read_csv(method_path) if method_path.exists() else pd.DataFrame()
    pairwise_df = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    targets_df = pd.read_csv(targets_path) if targets_path.exists() else pd.DataFrame()
    correctness_df = pd.read_csv(correctness_path) if correctness_path.exists() else pd.DataFrame()

    if method_stats_df.empty:
        raise RuntimeError(f"Missing or empty method stats table: {method_path}")

    plot_df = method_stats_df[method_stats_df["method"].isin(["gradcam", "shap", "lime"])].copy()
    plot_df = plot_df.set_index("method").reindex([m for m in ["gradcam", "shap", "lime"] if m in plot_df["method"].values]).reset_index()
    plot_df["method"] = plot_df["method"].replace({"gradcam": "Grad-CAM", "shap": "SHAP", "lime": "LIME"})

    for col in [
        "n",
        "pass_rate",
        "pass_rate_ci95_low",
        "pass_rate_ci95_high",
        "mean_border_ratio",
        "mean_faith_delta_k20",
        "pass_rule_border_ratio_max",
        "pass_rule_faith_delta_k20_min",
    ]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    summary_tbl = plot_df[
        [
            "method",
            "n",
            "pass_rate",
            "pass_rate_ci95_low",
            "pass_rate_ci95_high",
            "mean_border_ratio",
            "mean_faith_delta_k20",
        ]
    ].copy()
    summary_tbl.columns = [
        "Method",
        "N",
        "Explanation Pass Rate",
        "CI95 Low",
        "CI95 High",
        "Mean Border Ratio",
        "Mean Faithfulness (k=20%)",
    ]
    summary_tbl["Explanation Pass Rate"] = (
        pd.to_numeric(summary_tbl["Explanation Pass Rate"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["CI95 Low"] = (
        pd.to_numeric(summary_tbl["CI95 Low"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["CI95 High"] = (
        pd.to_numeric(summary_tbl["CI95 High"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["Mean Border Ratio"] = pd.to_numeric(summary_tbl["Mean Border Ratio"], errors="coerce").round(3)
    summary_tbl["Mean Faithfulness (k=20%)"] = pd.to_numeric(
        summary_tbl["Mean Faithfulness (k=20%)"], errors="coerce"
    ).round(3)

    rate = pd.to_numeric(plot_df["pass_rate"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(plot_df["pass_rate_ci95_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(plot_df["pass_rate_ci95_high"], errors="coerce").to_numpy(dtype=float)
    err_low = np.clip(rate - low, 0.0, None)
    err_high = np.clip(high - rate, 0.0, None)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(plot_df["method"], rate, yerr=[err_low, err_high], capsize=6, color=["#4C78A8", "#F58518", "#54A24B"][:len(plot_df)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Explanation Pass Rate")
    ax.set_title("XAI Explanation Pass Rate (95% CI)")
    for i, value in enumerate(rate):
        if np.isfinite(value):
            ax.text(i, min(float(value) + 0.03, 0.98), f"{float(value) * 100.0:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()

    delta_text = "N/A"
    p_text = "N/A"
    if len(pairwise_df):
        row = pairwise_df.iloc[0]
        delta = pd.to_numeric(row.get("delta_pass_rate", np.nan), errors="coerce")
        pval = pd.to_numeric(row.get("mcnemar_pvalue_exact", np.nan), errors="coerce")
        if np.isfinite(delta):
            delta_text = f"{float(delta) * 100.0:.1f}%"
        if np.isfinite(pval):
            p_text = f"{float(pval):.4f}"

    coverage_text = "coverage file not found"
    if len(coverage_df):
        g_miss = int((pd.to_numeric(coverage_df.get("gradcam_done", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if "gradcam_done" in coverage_df.columns else -1
        s_miss = int((pd.to_numeric(coverage_df.get("shap_done", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if "shap_done" in coverage_df.columns else -1
        coverage_text = f"coverage check: gradcam_missing={g_miss}, shap_missing={s_miss}"
        if "lime_done" in coverage_df.columns:
            coverage_text += f", lime_missing={int((coverage_df['lime_done'] == 0).sum())}"

    scope_text = "scope unavailable"
    if "evaluation_scope" in method_stats_df.columns and len(method_stats_df):
        non_null_scope = method_stats_df["evaluation_scope"].dropna()
        if len(non_null_scope):
            scope_text = str(non_null_scope.iloc[0])

    rule_border = 0.35
    rule_faith = 0.0
    if "pass_rule_border_ratio_max" in method_stats_df.columns and len(method_stats_df):
        vals = pd.to_numeric(method_stats_df["pass_rule_border_ratio_max"], errors="coerce").dropna()
        if len(vals):
            rule_border = float(vals.iloc[0])
    if "pass_rule_faith_delta_k20_min" in method_stats_df.columns and len(method_stats_df):
        vals = pd.to_numeric(method_stats_df["pass_rule_faith_delta_k20_min"], errors="coerce").dropna()
        if len(vals):
            rule_faith = float(vals.iloc[0])

    audit_n = int(len(targets_df))
    correctness_note = "correct/wrong split table unavailable"
    if len(correctness_df):
        correctness_note = "correct/wrong split is provided in Section 7 advanced table"

    bottom_line_markdown = (
        f"> **Method comparison:** the table shows the observed pass rates for each method. "
        f"SHAP minus Grad-CAM: **{delta_text}**, McNemar exact p=**{p_text}**. "
        f"LIME comparisons are in `rq_xai_all_pairs_seed{int(seed)}_{split_key}.csv`.\n"
        f"> Small samples do not establish method superiority; local LIME R² measures surrogate fit only.\n"
        f"> **What this metric means:** explanation quality under a rule, not diagnosis correctness.\n"
        f"> **Rule used:** `(border_ratio <= {rule_border:.2f})` and `(faithfulness_delta_k20 > {rule_faith:.2f})`.\n"
        f"> **Audit scope:** `{scope_text}` with `N={audit_n}` targets.\n"
        f"> **Consistency check:** {coverage_text}; {correctness_note}."
    )

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "method_stats_path": str(method_path),
        "pairwise_path": str(pairwise_path),
        "coverage_path": str(coverage_path),
        "targets_path": str(targets_path),
        "correctness_path": str(correctness_path),
        "method_stats_df": method_stats_df,
        "pairwise_df": pairwise_df,
        "coverage_df": coverage_df,
        "targets_df": targets_df,
        "correctness_df": correctness_df,
        "summary_table": summary_tbl,
        "pass_rate_fig": fig,
        "bottom_line_markdown": bottom_line_markdown,
    }


def notebook_load_xai_advanced_audit(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    xai_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    tables_dir = Path(conf["paths"]["tables_dir"])

    rq1_path = tables_dir / f"rq1_gradcam_seed{int(seed)}_{split_key}.csv"
    rq2_path = tables_dir / f"rq2_shap_seed{int(seed)}_{split_key}.csv"
    pair_path = tables_dir / f"rq_xai_pairwise_seed{int(seed)}_{split_key}.csv"
    pass_by_class_path = tables_dir / f"rq_xai_pass_by_class_seed{int(seed)}_{split_key}.csv"
    pass_by_correctness_path = tables_dir / f"rq_xai_pass_by_correctness_seed{int(seed)}_{split_key}.csv"

    rq1_df = pd.read_csv(rq1_path) if rq1_path.exists() else pd.DataFrame()
    rq2_df = pd.read_csv(rq2_path) if rq2_path.exists() else pd.DataFrame()
    pair_df = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    classwise_df = pd.read_csv(pass_by_class_path) if pass_by_class_path.exists() else pd.DataFrame()
    correctness_df = pd.read_csv(pass_by_correctness_path) if pass_by_correctness_path.exists() else pd.DataFrame()

    correctness_view = pd.DataFrame()
    if len(correctness_df):
        correctness_view = correctness_df.copy()
        correctness_view["pass_rate"] = pd.to_numeric(correctness_view["pass_rate"], errors="coerce").round(3)
        correctness_view["mean_border_ratio"] = pd.to_numeric(correctness_view["mean_border_ratio"], errors="coerce").round(3)
        correctness_view["mean_faith_delta_k20"] = pd.to_numeric(correctness_view["mean_faith_delta_k20"], errors="coerce").round(3)
        correctness_view = correctness_view.sort_values(["method", "group"]).reset_index(drop=True)

    classwise_view = pd.DataFrame()
    if len(classwise_df):
        classwise_view = classwise_df.copy()
        classwise_view["pass_rate"] = pd.to_numeric(classwise_view["pass_rate"], errors="coerce").round(3)
        classwise_view = classwise_view.sort_values(["method", "class_id"]).reset_index(drop=True)

    discord_df = pd.DataFrame()
    if (
        len(rq1_df)
        and len(rq2_df)
        and {"sample_id", "gradcam_pass"}.issubset(rq1_df.columns)
        and {"sample_id", "shap_pass"}.issubset(rq2_df.columns)
    ):
        discord_df = rq1_df[
            ["sample_id", "target_class", "pred_class", "confidence", "gradcam_pass"]
        ].merge(
            rq2_df[["sample_id", "shap_pass"]],
            on="sample_id",
            how="inner",
        )
        discord_df = discord_df[
            discord_df["gradcam_pass"].astype(int) != discord_df["shap_pass"].astype(int)
        ].sort_values("sample_id").reset_index(drop=True)

    def _load_status_payload(path_like: str | Path | None) -> dict[str, Any]:
        if path_like is None:
            return {}
        p = Path(str(path_like))
        if not p.exists():
            return {}
        try:
            return _load_json(p)
        except Exception:
            return {}

    run_id = ""
    if isinstance(xai_outputs, dict):
        run_id = str(xai_outputs.get("run_id", "") or "")
    if not run_id:
        latest_payload = _load_latest_run_record(conf, int(seed))
        if latest_payload:
            run_id = str(latest_payload.get("run_id", "") or "")

    grad_status_path: Path | None = None
    shap_status_path: Path | None = None
    if isinstance(xai_outputs, dict):
        grad_raw = str(xai_outputs.get("gradcam_status", "") or "").strip()
        shap_raw = str(xai_outputs.get("shap_status", "") or "").strip()
        grad_status_path = Path(grad_raw) if grad_raw else None
        shap_status_path = Path(shap_raw) if shap_raw else None
    if grad_status_path is None and run_id:
        grad_status_path = _gradcam_status_log_path(conf, run_id, split_key)
    if shap_status_path is None and run_id:
        shap_status_path = _shap_status_log_path(conf, run_id, split_key)

    grad_status_payload = _load_status_payload(grad_status_path)
    shap_status_payload = _load_status_payload(shap_status_path)

    status_df = pd.DataFrame(
        [
            {
                "method": "gradcam",
                "status": grad_status_payload.get("status", "unknown"),
                "reason_or_error": grad_status_payload.get("reason", ""),
                "supported_layers": ", ".join(grad_status_payload.get("supported_layers", []) or []),
                "requested_layers": ", ".join(grad_status_payload.get("requested_layers", []) or []),
                "target_rows": grad_status_payload.get("target_rows", ""),
            },
            {
                "method": "shap",
                "status": shap_status_payload.get("status", "unknown"),
                "reason_or_error": shap_status_payload.get("error", ""),
                "attempted_device": shap_status_payload.get("attempted_device", ""),
                "final_device": shap_status_payload.get("final_device", ""),
                "fallback_used": shap_status_payload.get("fallback_used", ""),
                "target_rows": shap_status_payload.get("target_rows", ""),
            },
        ]
    )

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "rq1_path": str(rq1_path),
        "rq2_path": str(rq2_path),
        "pair_path": str(pair_path),
        "pass_by_class_path": str(pass_by_class_path),
        "pass_by_correctness_path": str(pass_by_correctness_path),
        "rq1_df": rq1_df,
        "rq2_df": rq2_df,
        "pair_df": pair_df,
        "classwise_df": classwise_df,
        "correctness_df": correctness_df,
        "correctness_view": correctness_view,
        "classwise_view": classwise_view,
        "discord_df": discord_df,
        "status_df": status_df,
        "gradcam_status_path": str(grad_status_path) if grad_status_path is not None else "",
        "shap_status_path": str(shap_status_path) if shap_status_path is not None else "",
    }


def notebook_run_visual_review(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    wanted_classes: tuple[int, ...] = (0, 2, 3, 4),
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    split_key = str(split).strip().lower() or "test"
    split_key_to_manifest_key = {
        "train": "train_split",
        "val": "val_split",
        "test": "test_full",
    }
    if split_key not in split_key_to_manifest_key:
        raise ValueError(f"split must be one of train/val/test, got: {split}")

    xai_cfg = conf.setdefault("xai", {})
    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 8)
        max_shap_images = 2
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "auto"))
        max_shap_images = 4

    xai_shap_bg = int(xai_cfg.get("shap_background_size", 64))
    xai_dpi = int(xai_cfg.get("figure_dpi", 180))
    export_dpi = min(600, max(240, xai_dpi))
    shap_panel_size = (3.0, 3.4)
    gradcam_pair_size = (6.0, 3.4)

    manifests = prepare_data_manifests(conf, seed=int(seed))
    manifest_path = Path(manifests[split_key_to_manifest_key[split_key]])
    manifest_df = pd.read_csv(manifest_path)
    if len(manifest_df) == 0:
        raise RuntimeError(f"Empty {split_key} manifest: {manifest_path}")

    class_targets = [int(c) for c in wanted_classes]
    samples: list[tuple[str, int]] = []
    for cls in class_targets:
        sub = manifest_df[manifest_df["class_id"].astype(int) == int(cls)]
        if len(sub) == 0:
            continue
        row = sub.sample(n=1, random_state=int(seed)).iloc[0]
        samples.append((str(row["image_path"]), int(row["class_id"])))

    if len(samples) == 0:
        row = manifest_df.sample(n=1, random_state=int(seed)).iloc[0]
        samples = [(str(row["image_path"]), int(row["class_id"]))]

    gradcam_save_path = Path(conf["paths"]["figures_dir"]) / "gradcam_demo_grid.png"
    shap_save_path = Path(conf["paths"]["figures_dir"]) / "shap_demo_grid.png"

    gradcam_fig: plt.Figure | None = None
    shap_fig: plt.Figure | None = None
    warnings_list: list[str] = []
    try:
        gradcam_fig = plot_gradcam_grid(
            cfg_path=conf,
            seed=int(seed),
            samples=samples,
            ncols=2,
            gradcam_layer=str(xai_cfg.get("gradcam_layer", "layer4")),
            alpha=0.45,
            figsize_per_pair=gradcam_pair_size,
            save_path=gradcam_save_path,
            dpi=export_dpi,
        )
    except Exception as exc:
        warnings_list.append(f"Grad-CAM grid unavailable: {exc}")

    shap_subset = samples[:max_shap_images]
    shap_paths = [p for p, _ in shap_subset]
    shap_true_classes = [int(c) for _, c in shap_subset]
    try:
        shap_fig = plot_shap_grid(
            cfg_path=conf,
            seed=int(seed),
            image_paths=shap_paths,
            background_size=xai_shap_bg,
            vmax_percentile=99.5,
            save_path=shap_save_path,
            dpi=export_dpi,
            true_classes=shap_true_classes,
            show_correctness_border=True,
            panel_size=shap_panel_size,
        )
    except Exception as exc:
        warnings_list.append(f"SHAP grid unavailable: {exc}")

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": split_key,
        "safe_mode": bool(safe_mode),
        "manifest_path": str(manifest_path),
        "manifest_df": manifest_df,
        "samples": samples,
        "gradcam_fig": gradcam_fig,
        "shap_fig": shap_fig,
        "gradcam_path": str(gradcam_save_path),
        "shap_path": str(shap_save_path),
        "settings": {
            "device": str(xai_cfg.get("device")),
            "shap_background_size": int(xai_shap_bg),
            "shap_images": int(max_shap_images),
            "export_dpi": int(export_dpi),
            "shap_panel_size": shap_panel_size,
            "gradcam_pair_size": gradcam_pair_size,
        },
        "warnings": warnings_list,
    }


def notebook_run_single_case_report(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    image_path: str = "",
    split: str = "test",
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    xai_cfg = conf.setdefault("xai", {})

    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 8)
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "auto"))

    default_gradcam_layer = str(xai_cfg.get("gradcam_layer", "layer4"))
    requested_gradcam_layers = _parse_gradcam_layers(
        raw_layers=xai_cfg.get("gradcam_layers_eval", ["layer2", "layer3", default_gradcam_layer]),
        default_layer=default_gradcam_layer,
    )

    demo_image_path = str(image_path).strip()
    single_demo = run_single_case_demo(
        cfg_path=conf,
        seed=int(seed),
        image_path=demo_image_path,
        split=str(split).strip().lower() or "test",
        gradcam_layers=requested_gradcam_layers,
        shap_background_size=int(xai_cfg.get("shap_background_size", 64)),
        shap_panel_size=(3.2, 3.6),
        shap_dpi=min(600, max(240, int(xai_cfg.get("figure_dpi", 180)))),
    )

    prob_table = single_demo["prob_table"].copy()
    top1 = prob_table.iloc[0] if len(prob_table) else None
    top2 = prob_table.iloc[1] if len(prob_table) > 1 else None

    pred_name = str(top1["class_name"]) if top1 is not None else "N/A"
    pred_prob = float(top1["probability"]) if top1 is not None else float("nan")
    alt_txt = "N/A" if top2 is None else f"{top2['class_name']} ({float(top2['probability']):.3f})"

    triage_map = {
        0: "No DR pattern dominant. Continue routine follow-up if clinical exam is consistent.",
        1: "Mild DR pattern dominant. Consider short-interval follow-up and risk-factor optimization.",
        2: "Moderate DR pattern dominant. Recommend retina referral and closer follow-up planning.",
        3: "Severe DR pattern dominant. Escalate referral urgency to retina specialist.",
        4: "Proliferative DR pattern dominant. Treat as high-urgency retinal review candidate.",
    }
    pred_class = int(single_demo["pred_class"])
    triage_line = triage_map.get(pred_class, "Prediction outside expected range; use specialist review.")

    if np.isfinite(pred_prob) and pred_prob >= 0.80:
        conf_line = "Model confidence is high for this case."
    elif np.isfinite(pred_prob) and pred_prob >= 0.60:
        conf_line = "Model confidence is moderate; confirm with full clinical context."
    else:
        conf_line = "Model confidence is low; treat as uncertain and prioritize manual review."

    summary_markdown = (
        f"**Manifest used:** `{single_demo['manifest_path']}`\n\n"
        f"**Image:** `{single_demo['image_path']}`\n\n"
        f"**Prediction:** {single_demo['pred_label']} ({single_demo['pred_class']}) | "
        f"**Confidence:** {single_demo['confidence']:.4f}\n\n"
        f"**True label:** {single_demo.get('true_label', 'N/A')} ({single_demo.get('true_class', 'N/A')})\n\n"
        f"**Run ID:** `{single_demo['run_id']}` | **Device:** `{single_demo['device']}`"
    )
    story_markdown = "\n".join(
        [
            "### Clinical Decision Support Narrative",
            f"- Predicted DR grade: **{pred_name} ({pred_class})** with probability **{pred_prob:.3f}**.",
            f"- Next most likely alternative: **{alt_txt}**.",
            f"- Suggested triage framing: {triage_line}",
            f"- Confidence note: {conf_line}",
            "- How to use maps: highlighted regions are decision-support cues, not standalone diagnostic proof.",
            "- Clinical safeguard: final diagnosis and treatment decisions remain clinician-led.",
        ]
    )

    figure_sections = [
        {
            "title": "### Grad-CAM (Input + Requested Layers)",
            "path": str(single_demo["gradcam_panel_path"]),
        },
        {
            "title": "### SHAP (Per-Class Grid)",
            "path": str(single_demo["shap_grid_path"]),
        },
    ]

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": str(split).strip().lower() or "test",
        "safe_mode": bool(safe_mode),
        "single_demo": single_demo,
        "prob_table": prob_table,
        "summary_markdown": summary_markdown,
        "story_markdown": story_markdown,
        "figure_sections": figure_sections,
        "xai_warnings": list(single_demo.get("xai_warnings", []) or []),
    }


def clean_generated_outputs(cfg_path: str | Path = "configs/base.yaml") -> None:
    conf = _cfg(cfg_path)
    targets = [
        Path(conf["paths"]["manifests_dir"]),
        Path(conf["paths"]["checkpoints_dir"]),
        Path(conf["paths"]["predictions_dir"]),
        Path(conf["paths"]["logs_dir"]),
        Path(conf["paths"]["figures_dir"]),
        Path(conf["paths"]["tables_dir"]),
    ]

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    project_root = Path(conf["project_root"])
    for cache_path in project_root.rglob("__pycache__"):
        if cache_path.is_dir():
            shutil.rmtree(cache_path, ignore_errors=True)

    pytest_cache = project_root / ".pytest_cache"
    if pytest_cache.exists() and pytest_cache.is_dir():
        shutil.rmtree(pytest_cache, ignore_errors=True)


__all__ = [
    "load_project_config",
    "freeze_current_test_manifest",
    "prepare_data_manifests",
    "train_dr_classifier",
    "build_validation_calibration_table",
    "run_split_inference",
    "run_xai_analysis",
    "evaluate_pipeline_outputs",
    "export_final_headline_metrics",
    "export_benchmark_scoreboard",
    "run_complete_workflow",
    "run_benchmark_experiments",
    "predict_single_image_with_explanations",
    "explain_single_image_detailed",
    "run_single_case_demo",
    "plot_gradcam_grid",
    "plot_shap_grid",
    "notebook_prepare_data_overview",
    "notebook_run_training",
    "notebook_run_core_evaluation",
    "notebook_run_xai",
    "notebook_load_xai_committee_summary",
    "notebook_load_xai_advanced_audit",
    "notebook_run_visual_review",
    "notebook_run_single_case_report",
    "clean_generated_outputs",
]
