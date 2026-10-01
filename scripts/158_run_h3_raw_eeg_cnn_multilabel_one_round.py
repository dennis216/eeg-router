"""One-round pure-EEG multi-label source classifier for CHB-MIT and TUSZ.

This is the revised experiment requested after the raw-waveform design review.
Each dataset is processed separately.  Every unique 10-s bin containing one or
more detector false-positive event midpoints is mapped back to the same
canonical 16-channel raw EEG recording, resampled to 128 Hz, and represented
as a 16 x 1280 waveform.  If multiple detectors produced an event in that
common bin, the target is multi-hot (for example [1, 1, 0]).

The network is exactly:

    Conv1D(16 -> 32) + BatchNorm1d + ReLU
    Conv1D(32 -> 64) + BatchNorm1d + ReLU
    Conv1D(64 -> 128) + BatchNorm1d + ReLU
    GlobalAvgPool
    Linear(128 -> 3)

The three sigmoid outputs are trained with BCEWithLogitsLoss.  No detector
probability, event duration, QC value, HMM state, or other model-derived
quantity is an input.  Event duration is used only to define the source event
midpoint used for the fixed 10-s raw window.

Only one fresh outer GroupKFold round is run (outer fold 0) for each dataset.
The remaining outer folds are recorded as not run.  A patient-level
GroupShuffleSplit inside the development patients supplies validation for
early stopping and per-detector threshold selection.  The held-out test is
evaluated once with frozen validation thresholds.  Best-validation
checkpoints are written during training and retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse only the raw-signal/cache/model utilities from the preceding design
# script.  That script is never executed on import and contains no classifier
# output data.
_BASE_PATH = Path(__file__).resolve().parent / "157_run_h3_raw_eeg_cnn_fixed10s.py"
_SPEC = importlib.util.spec_from_file_location("bfa_raw_eeg_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load base raw EEG utility module: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL_COLUMNS = tuple(f"label_{name}" for name in MODELS)
TARGET_HZ = 128.0
WINDOW_S = 10.0
WINDOW_SAMPLES = 1280
CHANNELS = 16
N_OUTER_SPLITS = 5
OUTER_FOLD = 0
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 128
DEFAULT_PATIENCE = 3
DEFAULT_BOOTSTRAP = 2000
BASE_SEED = 20260828

CHB_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "h2h3_events.parquet"
TUSZ_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_multilabel_windows(dataset: str, events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse detector events into common 10-s midpoint bins.

    The source table contains repeated event rows for frozen folds/seeds.  They
    are deduplicated only for constructing the common raw window, while the
    audit records how many source rows/events were represented.  A detector is
    positive for a common window if at least one of its unique FP events has a
    midpoint in that bin.
    """
    work = events.copy()
    work["center_s"] = (work["start_s"].astype(float) + work["end_s"].astype(float)) / 2.0
    work["window_bin"] = np.floor(work["center_s"] / WINDOW_S).astype(np.int64)
    unique_keys = ["patient_id", "recording", "relative_path", "model", "start_s", "end_s"]
    unique_events = work.drop_duplicates(unique_keys, keep="first").copy()
    group_keys = ["patient_id", "recording", "relative_path", "window_bin"]
    rows: list[dict[str, Any]] = []
    for key, group in unique_events.groupby(group_keys, sort=True, dropna=False):
        patient_id, recording, relative_path, window_bin = key
        labels = {str(model) for model in group["model"].astype(str)}
        center_s = (int(window_bin) + 0.5) * WINDOW_S
        row: dict[str, Any] = {
            "patient_id": str(patient_id),
            "recording": str(recording),
            "relative_path": str(relative_path),
            "window_bin": int(window_bin),
            "window_start_s": float(center_s - WINDOW_S / 2.0),
            "window_end_s": float(center_s + WINDOW_S / 2.0),
            "window_center_s": float(center_s),
            "source_event_count_unique": int(len(group)),
            "source_event_count_table": int(
                len(
                    work[
                        (work["patient_id"].astype(str) == str(patient_id))
                        & (work["recording"].astype(str) == str(recording))
                        & (work["relative_path"].astype(str) == str(relative_path))
                        & (work["window_bin"].astype(int) == int(window_bin))
                    ]
                )
            ),
            "source_models": ",".join(name for name in MODELS if name in labels),
        }
        for model_name, label_column in zip(MODELS, LABEL_COLUMNS):
            row[label_column] = int(model_name in labels)
        row["label_cardinality"] = int(sum(row[column] for column in LABEL_COLUMNS))
        row["window_id"] = sha256_bytes(
            f"{dataset}|{patient_id}|{relative_path}|{int(window_bin)}".encode()
        )[:24]
        rows.append(row)
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError(f"{dataset} produced no common 10-s windows")
    windows = windows.sort_values(
        ["patient_id", "relative_path", "window_bin"], kind="mergesort"
    ).reset_index(drop=True)
    windows["window_index"] = np.arange(len(windows), dtype=np.int64)
    counts = windows[list(LABEL_COLUMNS)].sum(axis=0).to_dict()
    meta = {
        "dataset": dataset,
        "source_event_rows_after_fp_filter": int(len(work)),
        "unique_detector_events_for_windowing": int(len(unique_events)),
        "common_windows": int(len(windows)),
        "patients": int(windows["patient_id"].nunique()),
        "recordings": int(windows["recording"].nunique()),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "multi_label_windows": int((windows["label_cardinality"] > 1).sum()),
        "multi_label_window_fraction": float((windows["label_cardinality"] > 1).mean()),
        "label_aggregation": "detector positive if any unique frozen FP event midpoint falls in the common 10-s bin",
        "deduplication_scope": "identical detector/recording/start/end rows are deduplicated only for common-window construction",
    }
    return windows, meta


def select_smoke_windows(windows: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return windows
    patients = sorted(windows["patient_id"].astype(str).unique().tolist())[:5]
    parts: list[pd.DataFrame] = []
    per_patient = max(1, int(limit // max(1, len(patients))))
    for patient in patients:
        part = windows[windows["patient_id"].astype(str) == patient].head(per_patient)
        parts.append(part)
    selected = pd.concat(parts, ignore_index=True).head(int(limit)).copy()
    if selected["patient_id"].nunique() < min(5, windows["patient_id"].nunique()):
        raise RuntimeError("smoke subset did not retain enough patients for GroupKFold")
    return selected


def split_outer_and_validation(
    windows: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    indices = np.arange(len(windows), dtype=np.int64)
    groups = windows["patient_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    outer_splits = list(outer.split(indices, groups=groups))
    development, test = outer_splits[OUTER_FOLD]
    development = np.asarray(development, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    train: np.ndarray | None = None
    validation: np.ndarray | None = None
    for attempt in range(25):
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(seed + attempt),
        )
        train_local, validation_local = next(
            inner.split(
                development,
                groups=windows.iloc[development]["patient_id"].astype(str).to_numpy(),
            )
        )
        candidate_train = development[np.asarray(train_local, dtype=np.int64)]
        candidate_validation = development[np.asarray(validation_local, dtype=np.int64)]
        train_label_presence = windows.iloc[candidate_train][list(LABEL_COLUMNS)].sum(axis=0)
        if bool((train_label_presence > 0).all()):
            train = candidate_train
            validation = candidate_validation
            break
    if train is None or validation is None:
        raise RuntimeError("could not create an all-label patient-level training split")
    assignment = pd.DataFrame(
        {
            "patient_id": sorted(windows["patient_id"].astype(str).unique().tolist()),
        }
    )
    assignment["outer_fold"] = assignment["patient_id"].map(
        {
            str(patient): int(fold)
            for fold, (_development, test_indices) in enumerate(outer_splits)
            for patient in windows.iloc[test_indices]["patient_id"].astype(str).unique()
        }
    )
    assignment["partition"] = "unassigned"
    assignment.loc[assignment["outer_fold"] == OUTER_FOLD, "partition"] = "test"
    train_patients = set(windows.iloc[train]["patient_id"].astype(str))
    validation_patients = set(windows.iloc[validation]["patient_id"].astype(str))
    assignment.loc[assignment["patient_id"].isin(train_patients), "partition"] = "train"
    assignment.loc[assignment["patient_id"].isin(validation_patients), "partition"] = "validation"
    return train, validation, test, assignment


class MultiLabelRawDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, windows: pd.DataFrame, dataset: str):
        self.windows = windows.reset_index(drop=True)
        self.dataset = dataset
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_limit = 2

    def __len__(self) -> int:
        return len(self.windows)

    def _raw(self, relative_path: str) -> np.ndarray:
        key = str(relative_path)
        if key in self._cache:
            raw = self._cache.pop(key)
            self._cache[key] = raw
            return raw
        path = _BASE.cache_path(self.dataset, key)
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        raw = np.asarray(raw, dtype=np.float32)
        self._cache[key] = raw
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return raw

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.windows.iloc[int(index)]
        raw = self._raw(str(row.relative_path))
        signal = _BASE.extract_window(
            raw,
            float(row.window_start_s),
            float(row.window_end_s),
        )
        labels = row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(signal),
            torch.from_numpy(labels),
            int(row.window_index),
        )


class RecordingBatchSampler(BatchSampler):
    def __init__(
        self,
        windows: pd.DataFrame,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.windows = windows.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = [
            np.asarray(list(positions), dtype=np.int64)
            for _, positions in self.windows.groupby("relative_path", sort=True).groups.items()
        ]

    def __len__(self) -> int:
        return sum(int(math.ceil(len(group) / self.batch_size)) for group in self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self.groups))
        if self.shuffle:
            rng.shuffle(order)
        for group_index in order:
            positions = self.groups[int(group_index)].copy()
            if self.shuffle:
                rng.shuffle(positions)
            for start in range(0, len(positions), self.batch_size):
                yield positions[start : start + self.batch_size].tolist()


def make_loader(
    windows: pd.DataFrame,
    indices: np.ndarray,
    dataset: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = windows.iloc[indices].reset_index(drop=True)
    data = MultiLabelRawDataset(subset, dataset)
    sampler = RecordingBatchSampler(subset, batch_size, shuffle, seed)
    return DataLoader(
        data,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def multilabel_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
    result: dict[str, float] = {
        "events": float(len(y)),
        "accuracy": float(np.mean(np.all(predictions == y, axis=1))),
        "exact_match_accuracy": float(np.mean(np.all(predictions == y, axis=1))),
        "label_accuracy": float(np.mean(predictions == y)),
        "macro_f1": _safe_mean(
            f1_score(y[:, index], predictions[:, index], zero_division=0)
            for index in range(len(MODELS))
        ),
        "micro_f1": float(f1_score(y.ravel(), predictions.ravel(), zero_division=0)),
        "balanced_accuracy": _safe_mean(
            balanced_accuracy_score(y[:, index], predictions[:, index])
            for index in range(len(MODELS))
        ),
        "macro_sensitivity": _safe_mean(
            recall_score(y[:, index], predictions[:, index], zero_division=0)
            for index in range(len(MODELS))
        ),
        "macro_precision": _safe_mean(
            precision_score(y[:, index], predictions[:, index], zero_division=0)
            for index in range(len(MODELS))
        ),
        "predicted_label_cardinality": float(predictions.sum(axis=1).mean()),
        "observed_label_cardinality": float(y.sum(axis=1).mean()),
    }
    auc_values: list[float] = []
    ap_values: list[float] = []
    for index, name in enumerate(MODELS):
        try:
            auc = float(roc_auc_score(y[:, index], probabilities[:, index]))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y[:, index], probabilities[:, index]))
        except ValueError:
            ap = float("nan")
        auc_values.append(auc)
        ap_values.append(ap)
        result[f"{name}_auroc"] = auc
        result[f"{name}_average_precision"] = ap
        result[f"{name}_sensitivity"] = float(
            recall_score(y[:, index], predictions[:, index], zero_division=0)
        )
        result[f"{name}_precision"] = float(
            precision_score(y[:, index], predictions[:, index], zero_division=0)
        )
        result[f"{name}_f1"] = float(
            f1_score(y[:, index], predictions[:, index], zero_division=0)
        )
    result["macro_auroc"] = _safe_mean(auc_values)
    result["macro_average_precision"] = _safe_mean(ap_values)
    try:
        result["micro_auroc"] = float(roc_auc_score(y.ravel(), probabilities.ravel()))
    except ValueError:
        result["micro_auroc"] = float("nan")
    return result


def choose_validation_thresholds(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(MODELS), 0.5, dtype=float)
    candidates = np.linspace(0.05, 0.95, 91)
    for index in range(len(MODELS)):
        if len(np.unique(y[:, index])) < 2:
            continue
        scores: list[tuple[float, float, float]] = []
        for threshold in candidates:
            prediction = (probabilities[:, index] >= threshold).astype(np.int64)
            f1 = float(f1_score(y[:, index], prediction, zero_division=0))
            balanced = float(balanced_accuracy_score(y[:, index], prediction))
            scores.append((f1, balanced, -abs(float(threshold) - 0.5)))
        thresholds[index] = float(candidates[int(np.argmax(np.asarray(scores), axis=0)[0])])
    return thresholds


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
    thresholds: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    window_indices: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for inputs, targets, indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            if criterion is not None:
                losses.append(float(criterion(logits, targets).detach().cpu()))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy().astype(np.int64))
            window_indices.extend(np.asarray(indices).astype(int).tolist())
    y = np.concatenate(labels, axis=0) if labels else np.empty((0, len(MODELS)), dtype=np.int64)
    probs = (
        np.concatenate(probabilities, axis=0)
        if probabilities
        else np.empty((0, len(MODELS)), dtype=float)
    )
    pred = (probs >= thresholds[None, :]).astype(np.int64) if len(probs) else np.empty_like(y)
    output = pd.DataFrame({"window_index": window_indices})
    for index, name in enumerate(MODELS):
        output[f"y_{name}"] = y[:, index]
        output[f"prob_{name}"] = probs[:, index]
        output[f"pred_{name}"] = pred[:, index]
    return (
        multilabel_metrics(y, probs, thresholds) if len(y) else {},
        output,
        float(np.mean(losses)) if losses else float("nan"),
    )


def train_one_round(
    windows: pd.DataFrame,
    dataset: str,
    dataset_root: Path,
    out_dir: Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    del dataset_root  # raw cache is addressed by the common utility module
    seed_everything(seed)
    train_loader = make_loader(windows, train_indices, dataset, batch_size, True, seed)
    validation_loader = make_loader(windows, validation_indices, dataset, batch_size, False, seed + 1)
    test_loader = make_loader(windows, test_indices, dataset, batch_size, False, seed + 2)
    model = _BASE.RawEEGSourceCNN().to(device)
    label_counts = windows.iloc[train_indices][list(LABEL_COLUMNS)].sum(axis=0).to_numpy(dtype=float)
    negative_counts = float(len(train_indices)) - label_counts
    pos_weight = np.divide(
        negative_counts,
        np.maximum(label_counts, 1.0),
        out=np.ones_like(label_counts),
        where=np.maximum(label_counts, 1.0) > 0,
    )
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        batches = 0
        for inputs, targets, _indices in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        validation_metrics, _validation_predictions, validation_loss = evaluate_loader(
            model, validation_loader, device, criterion, np.full(len(MODELS), 0.5)
        )
        score = validation_metrics.get("macro_auroc", float("nan"))
        if not np.isfinite(score):
            score = -validation_loss if np.isfinite(validation_loss) else -float("inf")
        improved = bool(score > best_score + 1e-6)
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            torch.save(
                {
                    "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
                    "state_dict": best_state,
                    "dataset": dataset,
                    "outer_fold": OUTER_FOLD,
                    "epoch": epoch,
                    "input_shape": [CHANNELS, WINDOW_SAMPLES],
                    "input_is_raw_eeg_only": True,
                    "model_derived_features": False,
                    "test_used_for_selection": False,
                    "pos_weight": pos_weight.tolist(),
                },
                out_dir / "raw_eeg_cnn_multilabel_best_validation.pt",
            )
        row = {
            "outer_fold": OUTER_FOLD,
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batches),
            "validation_loss": validation_loss,
            "validation_macro_auroc": validation_metrics.get("macro_auroc", float("nan")),
            "validation_micro_auroc": validation_metrics.get("micro_auroc", float("nan")),
            "validation_macro_f1_at_0.5": validation_metrics.get("macro_f1", float("nan")),
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "improved": improved,
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)) if use_amp else 0,
        }
        history.append(row)
        _BASE.write_progress(
            out_dir / "progress.json",
            {
                "dataset": dataset,
                "stage": "training",
                "outer_fold": OUTER_FOLD,
                "outer_fold_total": N_OUTER_SPLITS,
                "rounds_completed": 0,
                "epoch": epoch + 1,
                "epoch_total": epochs,
                "best_epoch": best_epoch,
                "test_used_for_selection": False,
                "checkpoint": str(out_dir / "raw_eeg_cnn_multilabel_best_validation.pt"),
            },
        )
        print(json.dumps({"dataset": dataset, **row}, sort_keys=True), flush=True)
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    validation_metrics_at_05, validation_predictions, validation_loss = evaluate_loader(
        model,
        validation_loader,
        device,
        criterion,
        np.full(len(MODELS), 0.5),
    )
    validation_y = validation_predictions[[f"y_{name}" for name in MODELS]].to_numpy(dtype=int)
    validation_prob = validation_predictions[[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float)
    thresholds = choose_validation_thresholds(validation_y, validation_prob)
    validation_metrics, _validation_predictions_frozen, _ = evaluate_loader(
        model,
        validation_loader,
        device,
        criterion,
        thresholds,
    )
    test_metrics, test_predictions, test_loss = evaluate_loader(
        model,
        test_loader,
        device,
        criterion,
        thresholds,
    )
    test_predictions = test_predictions.merge(
        windows.iloc[test_indices][
            [
                "window_index",
                "window_id",
                "patient_id",
                "recording",
                "relative_path",
                "window_bin",
                "window_start_s",
                "window_end_s",
                *LABEL_COLUMNS,
            ]
        ],
        on="window_index",
        how="left",
        validate="one_to_one",
    )
    test_predictions.to_parquet(out_dir / "raw_eeg_cnn_multilabel_outer_test_predictions.parquet", index=False)
    final_checkpoint = {
        "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
        "state_dict": best_state,
        "dataset": dataset,
        "outer_fold": OUTER_FOLD,
        "best_epoch": best_epoch,
        "input_shape": [CHANNELS, WINDOW_SAMPLES],
        "input_is_raw_eeg_only": True,
        "model_derived_features": False,
        "test_used_for_selection": False,
        "thresholds_from_validation_only": thresholds.tolist(),
        "pos_weight": pos_weight.tolist(),
    }
    checkpoint_path = out_dir / "raw_eeg_cnn_multilabel_outer_fold0_checkpoint.pt"
    torch.save(final_checkpoint, checkpoint_path)
    fold_metrics = {
        "dataset": dataset,
        "outer_fold": OUTER_FOLD,
        "outer_folds_total": N_OUTER_SPLITS,
        "rounds_completed": 1,
        "status": "complete",
        "train_events": int(len(train_indices)),
        "validation_events": int(len(validation_indices)),
        "test_events": int(len(test_indices)),
        "train_patients": int(windows.iloc[train_indices]["patient_id"].nunique()),
        "validation_patients": int(windows.iloc[validation_indices]["patient_id"].nunique()),
        "test_patients": int(windows.iloc[test_indices]["patient_id"].nunique()),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history)),
        "validation_loss_at_0.5": validation_loss,
        "validation_thresholds": thresholds.tolist(),
        "test_loss": test_loss,
        "test_evaluation_count": 1,
        "test_used_for_selection": False,
        "threshold_source": "validation-only",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    for prefix, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            fold_metrics[f"{prefix}_{key}"] = value
    pd.DataFrame(history).to_csv(out_dir / "raw_eeg_cnn_multilabel_training_history.csv", index=False)
    pd.DataFrame([fold_metrics]).to_csv(out_dir / "raw_eeg_cnn_multilabel_fold_metrics.csv", index=False)
    return fold_metrics, test_predictions


def bootstrap_patients(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    patients = np.asarray(sorted(predictions["patient_id"].astype(str).unique()), dtype=object)
    by_patient = {
        patient: np.flatnonzero(predictions["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(n_bootstrap)):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in selected])
        y = predictions.iloc[indices][[f"y_{name}" for name in MODELS]].to_numpy(dtype=int)
        probs = predictions.iloc[indices][[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float)
        metrics = multilabel_metrics(y, probs, thresholds)
        for metric in (
            "macro_auroc",
            "micro_auroc",
            "accuracy",
            "label_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_sensitivity",
            "macro_precision",
        ):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "value": metrics.get(metric, float("nan")),
                    "bootstrap_unit": "patient",
                    "patient_count": len(patients),
                }
            )
    return pd.DataFrame(rows)


def run_dataset(
    dataset: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    bootstrap: int,
    limit_windows: int,
    smoke: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    formal = not smoke and limit_windows <= 0
    if formal and any(
        (out_dir / name).exists()
        for name in (
            "manifest.json",
            "raw_eeg_cnn_multilabel_fold_metrics.csv",
            "raw_eeg_cnn_multilabel_outer_test_predictions.parquet",
        )
    ):
        raise SystemExit(f"refusing to overwrite formal multi-label output: {out_dir}")
    started = time.monotonic()
    events, dataset_root, event_meta = _BASE.normalize_events(dataset)
    windows, window_meta = build_multilabel_windows(dataset, events)
    windows = select_smoke_windows(windows, limit_windows)
    cache_meta = _BASE.ensure_raw128_cache(
        dataset,
        windows,
        dataset_root,
        out_dir / "progress.json",
    )
    windows.to_parquet(out_dir / "raw_eeg_multilabel_common_windows.parquet", index=False)
    windows_input_manifest = {
        "dataset": dataset,
        "source_event_table": event_meta["source_event_table"],
        "source_event_table_sha256": event_meta["source_event_table_sha256"],
        "window_meta": window_meta,
        "windows_used": int(len(windows)),
        "patients_used": int(windows["patient_id"].nunique()),
        "recordings_used": int(windows["recording"].nunique()),
        "input_shape": [CHANNELS, WINDOW_SAMPLES],
        "sampling_hz": TARGET_HZ,
        "window_seconds": WINDOW_S,
        "window_alignment": "common 10-s bin centered at detector-event midpoint bin",
        "raw_input": "canonical 16-channel bipolar raw EEG; polyphase resampling to 128 Hz; per-channel window de-mean; fixed divide by 100",
        "classifier_input_columns": ["raw_eeg_waveform_16x1280"],
        "model_derived_features": False,
        "detector_probabilities_as_input": False,
        "event_duration_as_input": False,
        "qc_features_as_input": False,
        "hmm_features_as_input": False,
        "target": "multi-hot detector-source labels",
        "label_columns": list(LABEL_COLUMNS),
        "cache": cache_meta,
    }
    (out_dir / "raw_eeg_multilabel_input_manifest.json").write_text(
        json.dumps(windows_input_manifest, indent=2, sort_keys=True) + "\n"
    )
    train, validation, test, assignment = split_outer_and_validation(
        windows,
        BASE_SEED,
    )
    assignment.to_csv(out_dir / "raw_eeg_cnn_multilabel_patient_split.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "dataset": dataset,
                "events": int(len(events)),
                "common_windows": int(len(windows)),
                "patients": int(windows["patient_id"].nunique()),
                "recordings": int(windows["recording"].nunique()),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "architecture": "Conv1D(16->32)->BN->ReLU->Conv1D(32->64)->BN->ReLU->Conv1D(64->128)->BN->ReLU->GlobalAvgPool->Linear(128,3)",
                "loss": "sigmoid outputs with BCEWithLogitsLoss",
                "outer_fold": OUTER_FOLD,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    fold_metrics, predictions = train_one_round(
        windows,
        dataset,
        dataset_root,
        out_dir,
        train,
        validation,
        test,
        epochs=1 if smoke else epochs,
        batch_size=min(batch_size, 16) if smoke else batch_size,
        patience=1 if smoke else patience,
        seed=BASE_SEED,
        device=device,
    )
    thresholds = np.asarray(fold_metrics["validation_thresholds"], dtype=float)
    bootstrap_frame = bootstrap_patients(
        predictions,
        n_bootstrap=10 if smoke else bootstrap,
        seed=BASE_SEED + 9000,
        thresholds=thresholds,
    )
    bootstrap_frame.to_csv(out_dir / "raw_eeg_cnn_multilabel_patient_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap_frame.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(
        out_dir / "raw_eeg_cnn_multilabel_patient_bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "evaluation": "outer_fold0_test_once",
                **{
                    key: value
                    for key, value in fold_metrics.items()
                    if key.startswith("test_")
                },
                "threshold_source": "validation-only",
                "test_used_for_selection": False,
                "test_evaluation_count": 1,
            }
        ]
    ).to_csv(out_dir / "raw_eeg_cnn_multilabel_summary_metrics.csv", index=False)
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": dataset,
        "status": "complete" if formal else "smoke",
        "formal": formal,
        "source_run": SOURCE_RUN,
        "source_event_table": event_meta["source_event_table"],
        "source_event_table_sha256": event_meta["source_event_table_sha256"],
        "source_event_rows_after_fp_filter": int(len(events)),
        "common_windows_used": int(len(windows)),
        "patients_used": int(windows["patient_id"].nunique()),
        "recordings_used": int(windows["recording"].nunique()),
        "window_construction": window_meta,
        "input": windows_input_manifest,
        "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
        "output_activation": "sigmoid independently per detector",
        "loss": "BCEWithLogitsLoss with training-only per-label pos_weight",
        "patient_split": {
            "outer": "ordinary GroupKFold(n_splits=5), canonical patient_id groups",
            "outer_fold_evaluated": OUTER_FOLD,
            "outer_folds_total": N_OUTER_SPLITS,
            "inner_validation": "patient-level GroupShuffleSplit inside outer development patients",
            "rounds_completed": 1,
        },
        "metrics": [
            "per-detector AUROC",
            "macro/micro AUROC",
            "per-detector average precision",
            "exact-match accuracy",
            "label-wise accuracy",
            "balanced accuracy",
            "macro/micro F1",
            "per-detector sensitivity/precision/F1",
            "patient bootstrap confidence intervals",
        ],
        "thresholds": {
            "source": "validation-only",
            "method": "per-detector validation F1 maximization; test never used for threshold selection",
            "values": thresholds.tolist(),
        },
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "checkpoint": {
            "path": "raw_eeg_cnn_multilabel_outer_fold0_checkpoint.pt",
            "sha256": fold_metrics["checkpoint_sha256"],
            "best_validation_checkpoint_retained": "raw_eeg_cnn_multilabel_best_validation.pt",
        },
        "bootstrap": {
            "unit": "patient",
            "replicates": int(10 if smoke else bootstrap),
        },
        "overfit_guard": {
            "selection_metric": "validation macro AUROC",
            "selection_partition": "validation patients only",
            "test_touched_during_training": False,
            "test_touched_during_threshold_selection": False,
            "test_evaluation_count_per_round": 1,
        },
        "fold_metrics": fold_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    _BASE.write_progress(
        out_dir / "progress.json",
        {
            "dataset": dataset,
            "stage": "complete",
            "status": manifest["status"],
            "outer_fold": OUTER_FOLD,
            "outer_folds_total": N_OUTER_SPLITS,
            "rounds_completed": 1,
            "test_evaluation_count": 1,
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("chbmit", "tusz"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    default_name = f"h3-raw-eeg-cnn-128hz-10s-multilabel-{args.dataset}-v1"
    out_dir = args.output_dir or (ROOT / "outputs/reports" / default_name)
    run_dataset(
        args.dataset,
        Path(out_dir),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        patience=max(1, int(args.patience)),
        bootstrap=max(1, int(args.bootstrap)),
        limit_windows=max(0, int(args.limit_windows)),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
