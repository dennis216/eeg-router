"""Train the balanced, pure-EEG, FP-union H3 classifier.

The builder in ``163_build_h3_fp_union_10s_balanced_dataset.py`` creates one
row per complete, non-overlapping 10-second window and one multi-hot target per
strict-overlap union event.  This script trains on every such window while
making each event and each patient contribute the same total loss.  The three
detector labels are then given equal positive/negative mass in the BCE loss.

This file is an implementation of the frozen protocol only; invoking it is a
separate action.  It never reads detector probabilities as CNN inputs, and it
refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
ANALYSIS_ID = "h3-fp-union-10s-balanced-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL_COLUMNS = tuple(f"label_{name}" for name in MODELS)
CHANNELS = 16
TARGET_HZ = 128.0
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = 1280
N_OUTER_SPLITS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("analysis_id") != ANALYSIS_ID:
        raise RuntimeError("config analysis_id does not match the frozen H3 protocol")
    return config, hashlib.sha256(config_bytes).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_windows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    windows = pd.read_parquet(path).copy()
    required = {
        "window_index",
        "window_id",
        "event_id",
        "patient_id",
        "relative_path",
        "window_start_s",
        "window_end_s",
        "windows_per_event",
        *LABEL_COLUMNS,
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise RuntimeError(f"window table missing columns: {missing}")
    if windows["window_id"].duplicated().any():
        raise RuntimeError("window_id is not unique")
    if windows["event_id"].isna().any() or windows["patient_id"].isna().any():
        raise RuntimeError("event_id/patient_id contains null values")
    windows["patient_id"] = windows["patient_id"].astype(str)
    windows["event_id"] = windows["event_id"].astype(str)
    windows["relative_path"] = windows["relative_path"].astype(str)
    for column in LABEL_COLUMNS:
        windows[column] = pd.to_numeric(windows[column], errors="raise").astype(np.float32)
        if not windows[column].isin((0.0, 1.0)).all():
            raise RuntimeError(f"{column} must be binary")
    event_sizes = windows.groupby("event_id", sort=False).size()
    declared_sizes = windows.groupby("event_id", sort=False)["windows_per_event"].first()
    if not np.array_equal(event_sizes.sort_index().to_numpy(), declared_sizes.sort_index().to_numpy()):
        raise RuntimeError("windows_per_event does not match actual event windows")
    if (windows["window_end_s"] - windows["window_start_s"] < WINDOW_SECONDS - 1e-6).any():
        raise RuntimeError("a window is shorter than 10 seconds")
    return windows.sort_values("window_index", kind="mergesort").reset_index(drop=True)


def patient_split(
    windows: pd.DataFrame,
    *,
    outer_fold: int,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if not 0 <= int(outer_fold) < N_OUTER_SPLITS:
        raise ValueError(f"outer_fold must be in [0,{N_OUTER_SPLITS - 1}]")
    indices = np.arange(len(windows), dtype=np.int64)
    groups = windows["patient_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    outer_splits = list(outer.split(indices, groups=groups))
    development, test = outer_splits[int(outer_fold)]
    development = np.asarray(development, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    train: np.ndarray | None = None
    validation: np.ndarray | None = None
    for attempt in range(100):
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=float(validation_fraction),
            random_state=int(seed + attempt),
        )
        local_train, local_validation = next(
            inner.split(
                development,
                groups=windows.iloc[development]["patient_id"].astype(str).to_numpy(),
            )
        )
        candidate_train = development[np.asarray(local_train, dtype=np.int64)]
        candidate_validation = development[np.asarray(local_validation, dtype=np.int64)]
        train_presence = windows.iloc[candidate_train][list(LABEL_COLUMNS)].sum(axis=0)
        if bool((train_presence > 0).all()):
            train = candidate_train
            validation = candidate_validation
            break
    if train is None or validation is None:
        raise RuntimeError("unable to create a patient-level train/validation split with all labels")

    patient_to_outer: dict[str, int] = {}
    for fold, (_development, fold_test) in enumerate(outer_splits):
        for patient in windows.iloc[fold_test]["patient_id"].astype(str).unique():
            patient_to_outer[str(patient)] = int(fold)
    assignment = pd.DataFrame({"patient_id": sorted(patient_to_outer)})
    assignment["outer_fold"] = assignment["patient_id"].map(patient_to_outer).astype(int)
    assignment["partition"] = "unassigned"
    train_patients = set(windows.iloc[train]["patient_id"].astype(str))
    validation_patients = set(windows.iloc[validation]["patient_id"].astype(str))
    test_patients = set(windows.iloc[test]["patient_id"].astype(str))
    assignment.loc[assignment["patient_id"].isin(train_patients), "partition"] = "train"
    assignment.loc[assignment["patient_id"].isin(validation_patients), "partition"] = "validation"
    assignment.loc[assignment["patient_id"].isin(test_patients), "partition"] = "test"
    if assignment["partition"].eq("unassigned").any():
        raise RuntimeError("some patients were not assigned")
    if train_patients & validation_patients or train_patients & test_patients or validation_patients & test_patients:
        raise RuntimeError("patient leakage detected")
    return train, validation, test, assignment


def base_window_weights(windows: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = windows.iloc[indices]
    event_counts = subset.groupby("patient_id", sort=False)["event_id"].nunique().to_dict()
    event_window_counts = subset.groupby("event_id", sort=False).size().to_dict()
    weights = np.asarray(
        [
            1.0
            / (
                float(event_counts[str(row.patient_id)])
                * float(event_window_counts[str(row.event_id)])
            )
            for row in subset.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("invalid patient/event-normalized weights")
    return weights


def balanced_mass(windows: pd.DataFrame, indices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = windows.iloc[indices][list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    positive = np.asarray([(weights * labels[:, i]).sum() for i in range(len(MODELS))], dtype=np.float64)
    negative = np.asarray([(weights * (1.0 - labels[:, i])).sum() for i in range(len(MODELS))], dtype=np.float64)
    if (positive <= 0).any() or (negative <= 0).any():
        raise RuntimeError(f"a detector has no positive or negative training mass: pos={positive}, neg={negative}")
    return positive, negative


def extract_window(raw128: np.ndarray, start_s: float) -> np.ndarray:
    if raw128.ndim != 2 or raw128.shape[0] != CHANNELS:
        raise ValueError(f"expected raw shape (16, samples), got {raw128.shape}")
    start_sample = int(round(float(start_s) * TARGET_HZ))
    stop_sample = start_sample + WINDOW_SAMPLES
    left_pad = max(0, -start_sample)
    right_pad = max(0, stop_sample - int(raw128.shape[1]))
    lo = max(0, start_sample)
    hi = min(int(raw128.shape[1]), stop_sample)
    if hi <= lo:
        edge = np.asarray(raw128[:, :1], dtype=np.float32)
        window = np.repeat(edge, WINDOW_SAMPLES, axis=1)
    else:
        window = np.asarray(raw128[:, lo:hi], dtype=np.float32)
        if left_pad or right_pad:
            window = np.pad(window, ((0, 0), (left_pad, right_pad)), mode="edge")
        if window.shape[1] < WINDOW_SAMPLES:
            window = np.pad(window, ((0, 0), (0, WINDOW_SAMPLES - window.shape[1])), mode="edge")
        elif window.shape[1] > WINDOW_SAMPLES:
            window = window[:, :WINDOW_SAMPLES]
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    window = window - window.mean(axis=1, keepdims=True)
    return (window / 100.0).astype(np.float32)


class RawWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        windows: pd.DataFrame,
        indices: np.ndarray,
        dataset: str,
        raw_cache_root: Path,
        weights: np.ndarray | None,
    ) -> None:
        self.windows = windows.iloc[indices].reset_index(drop=True)
        self.dataset = dataset
        self.raw_cache_root = raw_cache_root
        self.weights = np.ones(len(self.windows), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        if len(self.weights) != len(self.windows):
            raise ValueError("weight length does not match dataset")
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_limit = 2

    def __len__(self) -> int:
        return len(self.windows)

    def _raw(self, relative_path: str) -> np.ndarray:
        key = str(relative_path)
        if key in self.cache:
            raw = self.cache.pop(key)
            self.cache[key] = raw
            return raw
        path = self.raw_cache_root / self.dataset / Path(key).with_suffix(".npy")
        if not path.exists():
            raise FileNotFoundError(f"missing 128-Hz raw cache: {path}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != CHANNELS:
            raise RuntimeError(f"invalid raw cache shape {raw.shape}: {path}")
        self.cache[key] = raw
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return raw

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.windows.iloc[int(index)]
        signal = extract_window(self._raw(str(row.relative_path)), float(row.window_start_s))
        labels = row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        return (
            torch.from_numpy(signal),
            torch.from_numpy(labels),
            torch.tensor(float(self.weights[int(index)]), dtype=torch.float32),
            int(row.window_index),
        )


class EventAwareBatchSampler(BatchSampler):
    """Use every window once while spreading windows from one event across batches."""

    def __init__(self, windows: pd.DataFrame, batch_size: int, shuffle: bool, seed: int) -> None:
        self.windows = windows.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = [
            np.asarray(list(positions), dtype=np.int64)
            for _, positions in self.windows.groupby("event_id", sort=True).groups.items()
        ]

    def __len__(self) -> int:
        # The sampler deliberately interleaves events one round at a time.
        # Therefore its number of batches is the sum over rounds, rather than
        # ceil(total_windows / batch_size).
        max_length = max((len(group) for group in self.groups), default=0)
        return int(
            sum(
                math.ceil(sum(len(group) > offset for group in self.groups) / max(1, self.batch_size))
                for offset in range(max_length)
            )
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = [group.copy() for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        rounds: list[list[int]] = []
        max_length = max((len(group) for group in groups), default=0)
        for offset in range(max_length):
            pool = [int(group[offset]) for group in groups if offset < len(group)]
            if self.shuffle:
                rng.shuffle(pool)
            rounds.append(pool)
        for pool in rounds:
            for start in range(0, len(pool), self.batch_size):
                yield pool[start : start + self.batch_size]


def make_loader(
    windows: pd.DataFrame,
    indices: np.ndarray,
    dataset: str,
    raw_cache_root: Path,
    *,
    weights: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = windows.iloc[indices].reset_index(drop=True)
    data = RawWindowDataset(subset, np.arange(len(subset)), dataset, raw_cache_root, weights)
    sampler = EventAwareBatchSampler(subset, batch_size, shuffle, seed)
    return DataLoader(
        data,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class ConvGN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDilated(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class MultiScaleDetectorCNN(nn.Module):
    """Pure 10-second EEG encoder with no event/detector metadata inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvGN(CHANNELS, 64, 7, stride=2)
        self.branches = nn.ModuleList([ConvGN(64, 32, kernel) for kernel in (3, 7, 15)])
        self.project = ConvGN(96, 64, 1)
        self.dilated = nn.Sequential(*(ResidualDilated(64, dilation) for dilation in (1, 2, 4, 8)))
        self.to_128 = ConvGN(64, 128, 1)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(MODELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        multi = torch.cat([branch(stem) for branch in self.branches], dim=1)
        x = self.project(multi) + stem
        x = self.dilated(x)
        x = self.to_128(x)
        mean = x.mean(dim=-1)
        std = x.var(dim=-1, unbiased=False).clamp_min(1e-8).sqrt()
        return self.head(torch.cat([mean, std], dim=1))


class BalancedMultiLabelBCE(nn.Module):
    """Patient/event-normalized BCE with equal positive and negative mass per detector."""

    def __init__(self, positive_mass: np.ndarray, negative_mass: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("positive_mass", torch.as_tensor(positive_mass, dtype=torch.float32))
        self.register_buffer("negative_mass", torch.as_tensor(negative_mass, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for index in range(len(MODELS)):
            positive = weights * targets[:, index] * F.softplus(-logits[:, index])
            negative = weights * (1.0 - targets[:, index]) * F.softplus(logits[:, index])
            losses.append(
                0.5 * (
                    positive.sum() / self.positive_mass[index].clamp_min(1e-8)
                    + negative.sum() / self.negative_mass[index].clamp_min(1e-8)
                )
            )
        return torch.stack(losses).mean()


def collect_window_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[pd.DataFrame] = []
    losses: list[float] = []
    with torch.no_grad():
        for inputs, targets, _weights, window_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            target_array = targets.numpy().astype(np.int64)
            frame = pd.DataFrame({"window_index": np.asarray(window_indices).astype(int)})
            for index, name in enumerate(MODELS):
                frame[f"y_{name}"] = target_array[:, index]
                frame[f"logit_{name}"] = logits[:, index].cpu().numpy()
                frame[f"prob_{name}"] = probabilities[:, index]
            rows.append(frame)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return output, float(np.mean(losses)) if losses else float("nan")


def event_predictions(window_predictions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if window_predictions.empty:
        return pd.DataFrame()
    metadata = windows[
        ["window_index", "event_id", "patient_id", "relative_path", *LABEL_COLUMNS]
    ].copy()
    merged = window_predictions.merge(metadata, on="window_index", how="left", validate="one_to_one")
    if merged["event_id"].isna().any():
        raise RuntimeError("prediction contains an unknown window")
    grouped_rows: list[dict[str, Any]] = []
    for event_id, group in merged.groupby("event_id", sort=True):
        labels = group[list(LABEL_COLUMNS)].to_numpy(dtype=int)
        if not np.all(labels == labels[0]):
            raise RuntimeError(f"label vector changed inside event {event_id}")
        row: dict[str, Any] = {
            "event_id": str(event_id),
            "patient_id": str(group["patient_id"].iloc[0]),
            "relative_path": str(group["relative_path"].iloc[0]),
            "window_count": int(len(group)),
        }
        for index, name in enumerate(MODELS):
            row[f"y_{name}"] = int(labels[0, index])
            mean_logit = float(group[f"logit_{name}"].mean())
            row[f"mean_logit_{name}"] = mean_logit
            row[f"prob_{name}"] = float(1.0 / (1.0 + math.exp(-mean_logit)))
        grouped_rows.append(row)
    output = pd.DataFrame(grouped_rows)
    event_counts = output.groupby("patient_id", sort=False)["event_id"].transform("count").astype(float)
    output["patient_event_weight"] = 1.0 / event_counts
    return output


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])) if np.any(mask) else float("nan")


def event_metrics(events: pd.DataFrame, thresholds: np.ndarray) -> dict[str, float]:
    if events.empty:
        return {}
    weights = events["patient_event_weight"].to_numpy(dtype=float)
    y = events[[f"y_{name}" for name in MODELS]].to_numpy(dtype=int)
    probabilities = events[[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float)
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    result: dict[str, float] = {
        "events": float(len(events)),
        "patients": float(events["patient_id"].nunique()),
        "exact_match_accuracy": _weighted_mean(np.all(predictions == y, axis=1), weights),
        "label_accuracy": _weighted_mean((predictions == y).mean(axis=1), weights),
        "observed_label_cardinality": _weighted_mean(y.sum(axis=1), weights),
        "predicted_label_cardinality": _weighted_mean(predictions.sum(axis=1), weights),
    }
    aurocs: list[float] = []
    auprcs: list[float] = []
    balanced: list[float] = []
    f1_values: list[float] = []
    sensitivities: list[float] = []
    precisions: list[float] = []
    for index, name in enumerate(MODELS):
        truth = y[:, index]
        pred = predictions[:, index]
        prob = probabilities[:, index]
        if len(np.unique(truth)) < 2:
            auc = float("nan")
            ap = float("nan")
            bal = float("nan")
        else:
            auc = float(roc_auc_score(truth, prob, sample_weight=weights))
            ap = float(average_precision_score(truth, prob, sample_weight=weights))
            bal = float(balanced_accuracy_score(truth, pred, sample_weight=weights))
        f1 = float(f1_score(truth, pred, sample_weight=weights, zero_division=0))
        sensitivity = float(recall_score(truth, pred, sample_weight=weights, zero_division=0))
        precision = float(precision_score(truth, pred, sample_weight=weights, zero_division=0))
        result[f"{name}_auroc"] = auc
        result[f"{name}_auprc"] = ap
        result[f"{name}_balanced_accuracy"] = bal
        result[f"{name}_f1"] = f1
        result[f"{name}_sensitivity"] = sensitivity
        result[f"{name}_precision"] = precision
        aurocs.append(auc)
        auprcs.append(ap)
        balanced.append(bal)
        f1_values.append(f1)
        sensitivities.append(sensitivity)
        precisions.append(precision)
    result["macro_auroc"] = float(np.nanmean(aurocs))
    result["macro_auprc"] = float(np.nanmean(auprcs))
    result["balanced_accuracy"] = float(np.nanmean(balanced))
    result["macro_f1"] = float(np.nanmean(f1_values))
    result["macro_sensitivity"] = float(np.nanmean(sensitivities))
    result["macro_precision"] = float(np.nanmean(precisions))
    return result


def choose_validation_thresholds(events: pd.DataFrame) -> np.ndarray:
    thresholds = np.full(len(MODELS), 0.5, dtype=float)
    if events.empty:
        return thresholds
    weights = events["patient_event_weight"].to_numpy(dtype=float)
    candidates = np.linspace(0.05, 0.95, 91)
    for index, name in enumerate(MODELS):
        truth = events[f"y_{name}"].to_numpy(dtype=int)
        probabilities = events[f"prob_{name}"].to_numpy(dtype=float)
        if len(np.unique(truth)) < 2:
            continue
        scores: list[tuple[float, float, float]] = []
        for threshold in candidates:
            prediction = (probabilities >= threshold).astype(int)
            f1 = float(f1_score(truth, prediction, sample_weight=weights, zero_division=0))
            balanced = float(balanced_accuracy_score(truth, prediction, sample_weight=weights))
            scores.append((f1, balanced, -abs(float(threshold) - 0.5)))
        # Lexicographic selection makes the prespecified tie-break explicit:
        # maximize F1, then balanced accuracy, then choose the threshold closest
        # to 0.5.  ``argmax`` on a 2-D score array would silently ignore the
        # latter two criteria.
        best_index = max(range(len(candidates)), key=lambda position: scores[position])
        thresholds[index] = float(candidates[best_index])
    return thresholds


def train_round(
    windows: pd.DataFrame,
    *,
    dataset: str,
    config_sha256: str,
    raw_cache_root: Path,
    output_dir: Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)
    train_weights = base_window_weights(windows, train_indices)
    positive_mass, negative_mass = balanced_mass(windows, train_indices, train_weights)
    train_loader = make_loader(
        windows, train_indices, dataset, raw_cache_root,
        weights=train_weights, batch_size=batch_size, shuffle=True, seed=seed,
    )
    validation_loader = make_loader(
        windows, validation_indices, dataset, raw_cache_root,
        weights=None, batch_size=batch_size, shuffle=False, seed=seed + 1,
    )
    test_loader = make_loader(
        windows, test_indices, dataset, raw_cache_root,
        weights=None, batch_size=batch_size, shuffle=False, seed=seed + 2,
    )
    model = MultiScaleDetectorCNN().to(device)
    criterion = BalancedMultiLabelBCE(positive_mass, negative_mass).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets, weights, _window_indices in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        validation_window_predictions, _ = collect_window_predictions(model, validation_loader, device)
        validation_events = event_predictions(validation_window_predictions, windows)
        validation_metrics = event_metrics(validation_events, np.full(len(MODELS), 0.5))
        score = validation_metrics.get("macro_auroc", float("nan"))
        if not np.isfinite(score):
            score = -total_loss / max(1, batches)
        improved = bool(score > best_score + 1e-7)
        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save(
                {
                    "analysis_id": ANALYSIS_ID,
                    "config_sha256": config_sha256,
                    "architecture": "MultiScale/Dilated 1D CNN + GroupNorm + mean/std pooling + sigmoid heads",
                    "state_dict": best_state,
                    "dataset": dataset,
                    "input_shape": [CHANNELS, WINDOW_SAMPLES],
                    "input_is_raw_eeg_only": True,
                    "output_activation": "sigmoid",
                    "loss": "patient_event_normalized_equal_detector_positive_negative_BCE",
                    "positive_mass": positive_mass.tolist(),
                    "negative_mass": negative_mass.tolist(),
                    "test_used_for_selection": False,
                },
                output_dir / "best_validation_checkpoint.pt",
            )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": total_loss / max(1, batches),
                "validation_macro_auroc_at_0.5": validation_metrics.get("macro_auroc", float("nan")),
                "validation_macro_auprc_at_0.5": validation_metrics.get("macro_auprc", float("nan")),
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "improved": improved,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)) if use_amp else 0,
            }
        )
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "analysis_id": ANALYSIS_ID,
                    "dataset": dataset,
                    "stage": "training",
                    "epoch": int(epoch + 1),
                    "epoch_total": int(epochs),
                    "best_epoch": int(best_epoch),
                    "test_used_for_selection": False,
                },
                indent=2,
            )
            + "\n"
        )
        if epoch - best_epoch >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    validation_window_predictions, _ = collect_window_predictions(model, validation_loader, device)
    validation_events = event_predictions(validation_window_predictions, windows)
    thresholds = choose_validation_thresholds(validation_events)
    validation_metrics = event_metrics(validation_events, thresholds)
    test_window_predictions, _ = collect_window_predictions(model, test_loader, device)
    test_events = event_predictions(test_window_predictions, windows)
    test_metrics = event_metrics(test_events, thresholds)
    test_window_predictions.to_parquet(output_dir / "test_window_predictions.parquet", index=False)
    test_events.to_parquet(output_dir / "test_event_predictions.parquet", index=False)
    final_checkpoint = output_dir / "outer_fold_checkpoint.pt"
    torch.save(
        {
            "analysis_id": ANALYSIS_ID,
            "config_sha256": config_sha256,
            "architecture": "MultiScale/Dilated 1D CNN + GroupNorm + mean/std pooling + sigmoid heads",
            "state_dict": best_state,
            "dataset": dataset,
            "input_shape": [CHANNELS, WINDOW_SAMPLES],
            "input_is_raw_eeg_only": True,
            "model_derived_features": False,
            "thresholds_from_validation_only": thresholds.tolist(),
            "positive_mass": positive_mass.tolist(),
            "negative_mass": negative_mass.tolist(),
            "test_used_for_selection": False,
        },
        final_checkpoint,
    )
    fold_metrics: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "dataset": dataset,
        "status": "complete",
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history)),
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(validation_indices)),
        "test_windows": int(len(test_indices)),
        "train_events": int(windows.iloc[train_indices]["event_id"].nunique()),
        "validation_events": int(validation_events["event_id"].nunique()),
        "test_events": int(test_events["event_id"].nunique()),
        "train_patients": int(windows.iloc[train_indices]["patient_id"].nunique()),
        "validation_patients": int(windows.iloc[validation_indices]["patient_id"].nunique()),
        "test_patients": int(windows.iloc[test_indices]["patient_id"].nunique()),
        "validation_thresholds": thresholds.tolist(),
        "threshold_source": "validation-only",
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "checkpoint_sha256": sha256(final_checkpoint),
    }
    for prefix, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            fold_metrics[f"{prefix}_{key}"] = value
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame([fold_metrics]).to_csv(output_dir / "fold_metrics.csv", index=False)
    return fold_metrics, validation_events, test_events


def patient_bootstrap(events: pd.DataFrame, thresholds: np.ndarray, *, replicates: int, seed: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    patients = np.asarray(sorted(events["patient_id"].astype(str).unique()), dtype=object)
    by_patient = {
        patient: events[events["patient_id"].astype(str).eq(patient)].copy()
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        # Each bootstrap draw is a patient-level unit.  Keep duplicate draws as
        # separate contributions instead of grouping them back by the original
        # patient_id (which would underweight patients sampled more than once).
        sampled_parts: list[pd.DataFrame] = []
        for draw_index, patient in enumerate(sampled):
            part = by_patient[str(patient)].copy()
            part["patient_event_weight"] = 1.0 / max(1, len(part))
            part["_bootstrap_draw_patient"] = f"{draw_index}:{patient}"
            sampled_parts.append(part)
        selected = pd.concat(sampled_parts, ignore_index=True)
        metrics = event_metrics(selected, thresholds)
        for metric in (
            "macro_auroc",
            "macro_auprc",
            "exact_match_accuracy",
            "label_accuracy",
            "balanced_accuracy",
            "macro_f1",
        ):
            rows.append(
                {
                    "replicate": int(replicate),
                    "metric": metric,
                    "value": metrics.get(metric, float("nan")),
                    "bootstrap_unit": "patient",
                    "patient_count": int(len(patients)),
                }
            )
    return pd.DataFrame(rows)


def run(
    windows_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    config_path: Path,
    raw_cache_root: Path,
    outer_fold: int,
    seed: int,
    validation_fraction: float,
    epochs: int,
    batch_size: int,
    patience: int,
    bootstrap_replicates: int,
) -> None:
    _config, config_sha256 = load_config(config_path)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    windows = load_windows(windows_path)
    train_indices, validation_indices, test_indices, assignment = patient_split(
        windows,
        outer_fold=outer_fold,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    assignment.to_csv(output_dir / "patient_split.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics, _validation_events, test_events = train_round(
        windows,
        dataset=dataset,
        config_sha256=config_sha256,
        raw_cache_root=raw_cache_root,
        output_dir=output_dir,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
    )
    thresholds = np.asarray(fold_metrics["validation_thresholds"], dtype=float)
    bootstrap = patient_bootstrap(
        test_events,
        thresholds,
        replicates=bootstrap_replicates,
        seed=seed + 9000,
    )
    bootstrap.to_csv(output_dir / "patient_bootstrap.csv", index=False)
    summary = (
        bootstrap.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
        if not bootstrap.empty
        else pd.DataFrame(columns=["metric", "mean", "q025", "q975"])
    )
    summary.to_csv(output_dir / "patient_bootstrap_summary.csv", index=False)
    windows_manifest = windows_path.with_name("dataset_manifest.json")
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "dataset": dataset,
        "status": "complete",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "data_table": str(windows_path),
        "data_table_sha256": sha256(windows_path),
        "data_manifest_sha256": sha256(windows_manifest) if windows_manifest.exists() else None,
        "input_shape": [CHANNELS, WINDOW_SAMPLES],
        "raw_input": "16-channel 128-Hz raw EEG only; per-channel de-mean and fixed physical divisor 100",
        "label_columns": list(LABEL_COLUMNS),
        "window_rule": "all complete event-relative non-overlapping 10-second windows; incomplete tail dropped",
        "event_rule": "strict positive-time interval-overlap connected components within recording",
        "loss": {
            "window_base_weight": "1/(events_per_patient * windows_per_event)",
            "detector_loss_weight": "equal 1/3",
            "positive_negative_mass": "one_half_each",
            "oversampling": "none",
            "undersampling": "none",
        },
        "model": "MultiScale/Dilated 1D CNN + GroupNorm + temporal mean/std pooling + Linear(128,3) logits",
        "output_activation": "independent sigmoid",
        "split": {
            "outer": "GroupKFold",
            "outer_splits": N_OUTER_SPLITS,
            "outer_fold_evaluated": int(outer_fold),
            "inner_validation": "GroupShuffleSplit on development patients",
            "group": "canonical patient_id",
            "test_evaluation_count": 1,
            "test_used_for_selection": False,
        },
        "metrics": {
            "primary": "patient-weighted event-level macro-AUROC",
            "secondary": ["event-level macro-AUPRC", "per-detector AUROC/AUPRC", "exact-match accuracy", "label accuracy", "balanced accuracy", "macro-F1"],
            "event_aggregation": "mean logits within event then sigmoid",
            "bootstrap_unit": "patient",
            "bootstrap_replicates": int(bootstrap_replicates),
        },
        "fold_metrics": fold_metrics,
        "elapsed_seconds": float(time.monotonic() - started),
        # Include the two manifest files that are written immediately after
        # this object is assembled; this keeps the release inventory complete.
        "output_files": sorted(
            {path.name for path in output_dir.iterdir() if path.is_file()}
            | {"manifest.json", "progress.json"}
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "dataset": dataset,
                "stage": "complete",
                "config_sha256": config_sha256,
                "outer_fold": int(outer_fold),
                "test_evaluation_count": 1,
                "test_used_for_selection": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chbmit", "tusz"), required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--raw-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run(
        args.windows,
        args.output_dir,
        dataset=args.dataset,
        config_path=args.config,
        raw_cache_root=args.raw_cache_root,
        outer_fold=args.outer_fold,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        epochs=max(1, args.epochs),
        batch_size=max(1, args.batch_size),
        patience=max(1, args.patience),
        bootstrap_replicates=max(1, args.bootstrap),
    )


if __name__ == "__main__":
    main()
