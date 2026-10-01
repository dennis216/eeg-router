"""Train the specified pure-EEG detector-source classifier on CHB-MIT and TUSZ.

The input is always a 10-s window (16 channels x 1280 samples at 128 Hz)
centered on a false-positive event midpoint.  The classifier target is the
detector that produced the event: SingLEM, CBraMod, or TCN-GAT.  Detector
probabilities, event duration, QC features, event metadata, HMM states, and
all other model-derived quantities are excluded from the neural-network
input.  Event start/end are used only to locate the raw EEG window.

Architecture (kept exactly at the requested high level):

    Conv1D(16 -> 32) + BatchNorm + ReLU
    Conv1D(32 -> 64) + BatchNorm + ReLU
    Conv1D(64 -> 128) + BatchNorm + ReLU
    GlobalAvgPool
    Linear(128 -> 3)

All events are retained.  Splits are made with ordinary patient-level
GroupKFold; a patient-level GroupShuffleSplit inside each development fold
provides validation for early stopping.  The test fold is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv  # noqa: E402


SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: index for index, name in enumerate(MODELS)}
TARGET_HZ = 128.0
WINDOW_S = 10.0
WINDOW_SAMPLES = 1280
CHANNELS = 16
N_SPLITS = 5
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 128
DEFAULT_PATIENCE = 3
DEFAULT_BOOTSTRAP = 2000
BASE_SEED = 20260828
RAW_CACHE_ROOT = Path("/mnt/d/EEGData/bfa_h3_raw128_cnn_v1")

CHB_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "h2h3_events.parquet"
CHB_RAW_ROOT = Path(
    os.environ.get("BFA_CHB_RAW_ROOT", "/mnt/d/EEGData/bfa_biot_prest_input_v1")
)
TUSZ_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
TUSZ_COHORT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "tusz_groupkfold_external_manifest.json"


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


def normalize_events(dataset: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    if dataset == "chbmit":
        source_path = CHB_EVENT_PATH
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        events = pd.read_parquet(source_path)
        events = events[events["event_type"].astype(str) == "fp"].copy()
        events = events.rename(columns={"patient": "patient_id"})
        events["relative_path"] = events["relative_path"].astype(str)
        dataset_root = Path("/mnt/d/EEGData/chbmit-1.0.0")
    else:
        source_path = TUSZ_EVENT_PATH
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        events = pd.read_parquet(source_path)
        events = events[~events["is_true_positive"].astype(bool)].copy()
        events["patient_id"] = events["patient_id"].astype(str)
        events["relative_path"] = events["recording"].astype(str)
        if not TUSZ_COHORT_PATH.exists():
            raise FileNotFoundError(TUSZ_COHORT_PATH)
        cohort = json.loads(TUSZ_COHORT_PATH.read_text())
        dataset_root = Path(str(cohort["dataset_root"]))

    required = {"patient_id", "recording", "relative_path", "model", "start_s", "end_s"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise RuntimeError(f"{dataset} event table missing columns: {missing}")
    events = events.reset_index(drop=False).rename(columns={"index": "source_row"})
    events["patient_id"] = events["patient_id"].astype(str)
    events["model"] = events["model"].astype(str)
    events["recording"] = events["recording"].astype(str)
    events["relative_path"] = events["relative_path"].astype(str)
    events["start_s"] = pd.to_numeric(events["start_s"], errors="coerce")
    events["end_s"] = pd.to_numeric(events["end_s"], errors="coerce")
    events = events[np.isfinite(events["start_s"]) & np.isfinite(events["end_s"])].copy()
    events = events[events["end_s"] >= events["start_s"]].copy()
    events = events[events["model"].isin(MODELS)].copy()
    events["event_id"] = [
        sha256_bytes(
            f"{dataset}|{row.patient_id}|{row.recording}|{row.model}|"
            f"{row.start_s:.6f}|{row.end_s:.6f}|{int(row.source_row)}".encode()
        )[:20]
        for row in events.itertuples(index=False)
    ]
    events = events.sort_values(
        ["patient_id", "recording", "relative_path", "start_s", "end_s", "model", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    meta = {
        "dataset": dataset,
        "source_event_table": str(source_path.relative_to(ROOT)),
        "source_event_table_sha256": sha256(source_path),
        "source_rows_after_fp_filter": int(len(events)),
        "patients": int(events["patient_id"].nunique()),
        "recordings": int(events["recording"].nunique()),
        "models": sorted(events["model"].unique().tolist()),
        "dataset_root": str(dataset_root),
    }
    return events, dataset_root, meta


def cache_path(dataset: str, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    return RAW_CACHE_ROOT / dataset / relative.with_suffix(".npy")


def load_source_raw(dataset: str, relative_path: str, dataset_root: Path) -> tuple[np.ndarray, float, str]:
    if dataset == "chbmit":
        source = CHB_RAW_ROOT / Path(relative_path).with_suffix(".npy")
        if not source.exists():
            raise FileNotFoundError(source)
        raw = np.load(source, mmap_mode="r", allow_pickle=False)
        sampling_hz = 200.0
        source_description = str(source)
    else:
        source = dataset_root / relative_path
        if not source.exists():
            raise FileNotFoundError(source)
        raw, sampling_hz, _ = load_tuh_canonical_raw_uv(source)
        source_description = str(source)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != CHANNELS or raw.shape[1] < 2:
        raise ValueError(f"raw input must have shape (16, samples), got {raw.shape}: {source_description}")
    return raw, float(sampling_hz), source_description


def ensure_raw128_cache(dataset: str, events: pd.DataFrame, dataset_root: Path, progress_path: Path) -> dict[str, Any]:
    paths = sorted(events["relative_path"].astype(str).unique().tolist())
    cache_root = RAW_CACHE_ROOT / dataset
    cache_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, relative_path in enumerate(paths, start=1):
        destination = cache_path(dataset, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_description = ""
        if destination.exists():
            try:
                cached = np.load(destination, mmap_mode="r", allow_pickle=False)
                if cached.ndim == 2 and cached.shape[0] == CHANNELS:
                    records.append(
                        {
                            "relative_path": relative_path,
                            "cache_path": str(destination),
                            "shape": [int(cached.shape[0]), int(cached.shape[1])],
                            "sampling_hz": TARGET_HZ,
                            "status": "reused",
                        }
                    )
                    if index % 25 == 0 or index == len(paths):
                        progress_path.write_text(
                            json.dumps(
                                {
                                    "stage": "raw128_cache",
                                    "dataset": dataset,
                                    "recordings_completed": index,
                                    "recordings_total": len(paths),
                                    "status": "running",
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    continue
            except Exception:
                destination.unlink(missing_ok=True)
        raw, sampling_hz, source_description = load_source_raw(dataset, relative_path, dataset_root)
        ratio = Fraction(TARGET_HZ / float(sampling_hz)).limit_denominator(10_000)
        if ratio.numerator == ratio.denominator == 1:
            raw128 = raw.astype(np.float32, copy=False)
        else:
            raw128 = resample_poly(
                raw.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1
            ).astype(np.float32)
        if raw128.ndim != 2 or raw128.shape[0] != CHANNELS or not np.isfinite(raw128).all():
            raise ValueError(f"invalid 128-Hz cache result for {source_description}: {raw128.shape}")
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, raw128, allow_pickle=False)
        temporary.replace(destination)
        records.append(
            {
                "relative_path": relative_path,
                "cache_path": str(destination),
                "shape": [int(raw128.shape[0]), int(raw128.shape[1])],
                "source_sampling_hz": float(sampling_hz),
                "sampling_hz": TARGET_HZ,
                "status": "created",
            }
        )
        progress_path.write_text(
            json.dumps(
                {
                    "stage": "raw128_cache",
                    "dataset": dataset,
                    "recordings_completed": index,
                    "recordings_total": len(paths),
                    "status": "running",
                },
                indent=2,
            )
            + "\n"
        )
    manifest = {
        "dataset": dataset,
        "cache_root": str(cache_root),
        "recordings": len(records),
        "channels": CHANNELS,
        "sampling_hz": TARGET_HZ,
        "preprocessing": "canonical bipolar raw EEG in microvolts; resampling only; no model-specific filtering",
        "records": records,
    }
    manifest_path = cache_root / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "cache_root": str(cache_root),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256(manifest_path),
        "recordings": len(records),
    }


def extract_window(raw128: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    center_s = (float(start_s) + float(end_s)) / 2.0
    start_sample = int(round((center_s - WINDOW_S / 2.0) * TARGET_HZ))
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
            window = np.pad(
                window,
                ((0, 0), (0, WINDOW_SAMPLES - window.shape[1])),
                mode="edge",
            )
        elif window.shape[1] > WINDOW_SAMPLES:
            window = window[:, :WINDOW_SAMPLES]
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # A fixed physical scale keeps the input in a stable numerical range. It
    # is not fitted on test data and is not a detector-derived feature.
    window = window - window.mean(axis=1, keepdims=True)
    return (window / 100.0).astype(np.float32)


class RawEventDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Raw event windows with a one-recording LRU cache.

    The batch sampler groups adjacent indices by recording, so a raw recording
    is normally loaded once for a contiguous set of events.
    """

    def __init__(self, events: pd.DataFrame, dataset: str, dataset_root: Path, indices: Iterable[int]):
        self.events = events.iloc[list(indices)].reset_index(drop=True)
        self.dataset = dataset
        self.dataset_root = dataset_root
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_limit = 2

    def __len__(self) -> int:
        return len(self.events)

    def _raw(self, relative_path: str) -> np.ndarray:
        key = str(relative_path)
        if key in self._cache:
            raw = self._cache.pop(key)
            self._cache[key] = raw
            return raw
        path = cache_path(self.dataset, key)
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        raw = np.asarray(raw, dtype=np.float32)
        self._cache[key] = raw
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return raw

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.events.iloc[int(index)]
        raw = self._raw(str(row.relative_path))
        window = extract_window(raw, float(row.start_s), float(row.end_s))
        label = LABELS[str(row.model)]
        return torch.from_numpy(window), int(label), int(row.source_row)


class RecordingBatchSampler(BatchSampler):
    """Yield batches grouped by recording, with deterministic epoch shuffles."""

    def __init__(
        self,
        events: pd.DataFrame,
        indices: Iterable[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.events = events.iloc[list(indices)].reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: list[np.ndarray] = []
        for _, positions in self.events.groupby("relative_path", sort=True).groups.items():
            self.groups.append(np.asarray(list(positions), dtype=np.int64))

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
            group = self.groups[int(group_index)].copy()
            if self.shuffle:
                rng.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                yield group[start : start + self.batch_size].tolist()


class RawEEGSourceCNN(nn.Module):
    """The fixed 1D CNN requested by the user."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(CHANNELS, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, len(MODELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def metrics_from_arrays(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.argmax(probabilities, axis=1)
    result: dict[str, float] = {
        "events": float(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, labels=list(range(len(MODELS))), average="macro", zero_division=0)
        ),
    }
    try:
        result["macro_auroc"] = float(
            roc_auc_score(
                y,
                probabilities,
                labels=list(range(len(MODELS))),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        result["macro_auroc"] = float("nan")
    for index, model in enumerate(MODELS):
        try:
            result[f"{model}_auroc"] = float(
                roc_auc_score((y == index).astype(np.int64), probabilities[:, index])
            )
        except ValueError:
            result[f"{model}_auroc"] = float("nan")
    return result


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    model.eval()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    source_rows: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for inputs, targets, row_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            if criterion is not None:
                losses.append(float(criterion(logits, targets).detach().cpu()))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities.append(probs)
            labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
            source_rows.extend(np.asarray(row_ids).astype(int).tolist())
    y = np.asarray(labels, dtype=np.int64)
    probs = np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, len(MODELS)))
    output = pd.DataFrame({"source_row": source_rows, "y": y})
    for index, model_name in enumerate(MODELS):
        output[f"prob_{model_name}"] = probs[:, index]
    output["prediction"] = np.argmax(probs, axis=1) if len(probs) else np.empty((0,), dtype=int)
    metrics = metrics_from_arrays(y, probs) if len(y) else {}
    return metrics, output, float(np.mean(losses)) if losses else float("nan")


def make_split(
    events: pd.DataFrame,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = GroupKFold(n_splits=N_SPLITS)
    all_indices = np.arange(len(events), dtype=np.int64)
    splits = list(outer.split(all_indices, groups=events["patient_id"].astype(str)))
    development, test = splits[fold]
    development = np.asarray(development, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    for attempt in range(20):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(seed + attempt),
        )
        train_local, validation_local = next(
            splitter.split(development, groups=events.iloc[development]["patient_id"].astype(str))
        )
        train = development[np.asarray(train_local, dtype=np.int64)]
        validation = development[np.asarray(validation_local, dtype=np.int64)]
        train_classes = set(events.iloc[train]["model"].astype(str).tolist())
        if train_classes == set(MODELS):
            return train, validation, test
    raise RuntimeError(f"could not construct all-class train split for fold {fold}")


def class_weights(events: pd.DataFrame, train_indices: np.ndarray) -> torch.Tensor:
    counts = events.iloc[train_indices]["model"].astype(str).map(LABELS).value_counts()
    values = np.asarray([float(counts.get(index, 0)) for index in range(len(MODELS))], dtype=float)
    if np.any(values <= 0):
        raise RuntimeError(f"training split lacks detector class: {values.tolist()}")
    weights = values.sum() / (len(MODELS) * values)
    return torch.tensor(weights, dtype=torch.float32)


def dataloader_for(
    events: pd.DataFrame,
    indices: np.ndarray,
    dataset: str,
    dataset_root: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    subset = events.iloc[indices].reset_index(drop=True)
    dataset_object = RawEventDataset(subset, dataset, dataset_root, np.arange(len(subset)))
    sampler = RecordingBatchSampler(subset, np.arange(len(subset)), batch_size, shuffle, seed)
    return DataLoader(
        dataset_object,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_one_fold(
    events: pd.DataFrame,
    dataset: str,
    dataset_root: Path,
    out_dir: Path,
    fold: int,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_everything(seed)
    train_idx, validation_idx, test_idx = make_split(events, fold, seed)
    train_loader = dataloader_for(events, train_idx, dataset, dataset_root, batch_size, True, seed)
    validation_loader = dataloader_for(events, validation_idx, dataset, dataset_root, batch_size, False, seed + 1)
    test_loader = dataloader_for(events, test_idx, dataset, dataset_root, batch_size, False, seed + 2)
    model = RawEEGSourceCNN().to(device)
    weights = class_weights(events, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    fold_start = time.monotonic()
    for epoch in range(epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for inputs, targets, _row_ids in train_loader:
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
            batch_count += 1
        validation_metrics, _validation_predictions, validation_loss = evaluate_loader(
            model, validation_loader, device, criterion
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
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss_sum / max(1, batch_count),
                "validation_loss": validation_loss,
                "validation_macro_auroc": validation_metrics.get("macro_auroc", float("nan")),
                "validation_accuracy": validation_metrics.get("accuracy", float("nan")),
                "validation_balanced_accuracy": validation_metrics.get("balanced_accuracy", float("nan")),
                "validation_macro_f1": validation_metrics.get("macro_f1", float("nan")),
                "best_score": best_score,
                "improved": improved,
            }
        )
        write_progress(
            out_dir / "progress.json",
            {
                "dataset": dataset,
                "stage": "training",
                "fold": fold,
                "fold_total": N_SPLITS,
                "epoch": epoch + 1,
                "epoch_total": epochs,
                "best_epoch": best_epoch,
                "train_events": int(len(train_idx)),
                "validation_events": int(len(validation_idx)),
                "test_events": int(len(test_idx)),
            },
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "train_loss": history[-1]["train_loss"],
                    "validation_macro_auroc": history[-1]["validation_macro_auroc"],
                    "best_epoch": best_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"fold {fold} never produced a valid checkpoint")
    model.load_state_dict(best_state)
    test_metrics, test_predictions, test_loss = evaluate_loader(model, test_loader, device, criterion)
    test_predictions = test_predictions.merge(
        events.iloc[test_idx][
            ["source_row", "event_id", "patient_id", "recording", "relative_path", "model", "start_s", "end_s"]
        ],
        on="source_row",
        how="left",
        validate="one_to_one",
    )
    test_predictions["fold"] = int(fold)
    test_predictions["centering"] = "raw_waveform"
    test_predictions.to_parquet(out_dir / f"raw_eeg_cnn_fold{fold}_test_predictions.parquet", index=False)
    checkpoint = {
        "architecture": "Conv1D(16,32)+BN+ReLU -> Conv1D(32,64)+BN+ReLU -> Conv1D(64,128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
        "state_dict": best_state,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_patients": sorted(events.iloc[train_idx]["patient_id"].astype(str).unique().tolist()),
        "validation_patients": sorted(events.iloc[validation_idx]["patient_id"].astype(str).unique().tolist()),
        "test_patients": sorted(events.iloc[test_idx]["patient_id"].astype(str).unique().tolist()),
        "input_shape": [CHANNELS, WINDOW_SAMPLES],
        "input_units": "scaled microvolts (fixed divide by 100 after per-channel window de-mean)",
        "model_derived_features": False,
        "test_used_for_selection": False,
    }
    torch.save(checkpoint, out_dir / f"raw_eeg_cnn_fold{fold}.pt")
    fold_record = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "status": "complete",
        "train_events": int(len(train_idx)),
        "validation_events": int(len(validation_idx)),
        "test_events": int(len(test_idx)),
        "train_patients": int(events.iloc[train_idx]["patient_id"].nunique()),
        "validation_patients": int(events.iloc[validation_idx]["patient_id"].nunique()),
        "test_patients": int(events.iloc[test_idx]["patient_id"].nunique()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_macro_auroc": history[best_epoch]["validation_macro_auroc"],
        "validation_accuracy": history[best_epoch]["validation_accuracy"],
        "validation_balanced_accuracy": history[best_epoch]["validation_balanced_accuracy"],
        "validation_macro_f1": history[best_epoch]["validation_macro_f1"],
        "test_loss": test_loss,
        "test_evaluation_count": 1,
        "test_used_for_selection": False,
        "classifier": "raw_eeg_1d_cnn_fixed_10s_128hz",
        "test_macro_auroc": test_metrics.get("macro_auroc", float("nan")),
        "test_accuracy": test_metrics.get("accuracy", float("nan")),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", float("nan")),
        "test_macro_f1": test_metrics.get("macro_f1", float("nan")),
    }
    for model_name in MODELS:
        fold_record[f"test_{model_name}_auroc"] = test_metrics.get(f"{model_name}_auroc", float("nan"))
    pd.DataFrame(history).to_csv(out_dir / f"raw_eeg_cnn_fold{fold}_training_history.csv", index=False)
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": dataset,
            "stage": "fold_complete",
            "fold": fold,
            "fold_total": N_SPLITS,
            "best_epoch": best_epoch,
            "test_macro_auroc": fold_record["test_macro_auroc"],
        },
    )
    return fold_record, test_predictions


def patient_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    patients = np.asarray(sorted(predictions["patient_id"].astype(str).unique()), dtype=object)
    patient_indices = {
        patient: np.flatnonzero(predictions["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(n_bootstrap)):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_indices[patient] for patient in selected])
        y = predictions.iloc[indices]["y"].to_numpy(dtype=np.int64)
        probs = predictions.iloc[indices][[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float)
        metrics = metrics_from_arrays(y, probs)
        for metric_name in ("macro_auroc", "accuracy", "balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric_name,
                    "value": metrics.get(metric_name, float("nan")),
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
    limit_events: int,
    smoke: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not smoke and any(
        (out_dir / name).exists()
        for name in (
            "manifest.json",
            "raw_eeg_cnn_summary_metrics.csv",
            "raw_eeg_cnn_patient_bootstrap.csv",
        )
    ):
        raise SystemExit(f"refusing to overwrite formal raw EEG CNN output: {out_dir}")
    started = time.monotonic()
    events, dataset_root, event_meta = normalize_events(dataset)
    if limit_events > 0:
        # A smoke subset must still contain several patients and all three
        # detector labels; taking the first rows would usually select only
        # one patient because the formal table is patient-sorted.
        selected_parts: list[pd.DataFrame] = []
        smoke_patients = sorted(events["patient_id"].astype(str).unique().tolist())[:5]
        per_class = max(1, int(limit_events // max(1, len(smoke_patients) * len(MODELS))))
        for patient in smoke_patients:
            for model_name in MODELS:
                part = events[
                    (events["patient_id"].astype(str) == patient)
                    & (events["model"].astype(str) == model_name)
                ].head(per_class)
                selected_parts.append(part)
        events = pd.concat(selected_parts, ignore_index=True).copy()
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": dataset,
            "stage": "preparing",
            "events": int(len(events)),
            "patients": int(events["patient_id"].nunique()),
            "recordings": int(events["recording"].nunique()),
        },
    )
    cache_meta = ensure_raw128_cache(dataset, events, dataset_root, out_dir / "progress.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "dataset": dataset,
                "events": len(events),
                "patients": int(events["patient_id"].nunique()),
                "recordings": int(events["recording"].nunique()),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "architecture": "Conv1D(16->32)->BN->ReLU->Conv1D(32->64)->BN->ReLU->Conv1D(64->128)->BN->ReLU->GlobalAvgPool->Linear(128,3)",
                "input_shape": [CHANNELS, WINDOW_SAMPLES],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    fold_records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    actual_splits = N_SPLITS if not smoke else min(2, N_SPLITS)
    for fold in range(actual_splits):
        record, fold_predictions = train_one_fold(
            events,
            dataset,
            dataset_root,
            out_dir,
            fold,
            epochs=1 if smoke else epochs,
            batch_size=min(batch_size, 16) if smoke else batch_size,
            patience=1 if smoke else patience,
            seed=BASE_SEED + fold,
            device=device,
        )
        fold_records.append(record)
        predictions.append(fold_predictions)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(out_dir / "raw_eeg_cnn_oof_predictions.parquet", index=False)
    pooled_metrics = metrics_from_arrays(
        all_predictions["y"].to_numpy(dtype=np.int64),
        all_predictions[[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float),
    )
    summary_rows = [
        {
            "dataset": dataset,
            "evaluation": "pooled_outer_oof",
            "folds": actual_splits,
            "events": int(len(all_predictions)),
            "patients": int(all_predictions["patient_id"].nunique()),
            "chance_macro_auroc": 0.5,
            **pooled_metrics,
        }
    ]
    pd.DataFrame(fold_records).to_csv(out_dir / "raw_eeg_cnn_fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "raw_eeg_cnn_summary_metrics.csv", index=False)
    bootstrap_frame = patient_bootstrap(
        all_predictions,
        n_bootstrap=10 if smoke else bootstrap,
        seed=BASE_SEED + 9000,
    )
    bootstrap_frame.to_csv(out_dir / "raw_eeg_cnn_patient_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap_frame.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(out_dir / "raw_eeg_cnn_patient_bootstrap_summary.csv", index=False)
    formal = not smoke and limit_events <= 0
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": dataset,
        "status": "complete" if formal else "smoke",
        "formal": formal,
        "source_run": SOURCE_RUN,
        "source_event_table": event_meta["source_event_table"],
        "source_event_table_sha256": event_meta["source_event_table_sha256"],
        "events": int(len(events)),
        "patients": int(events["patient_id"].nunique()),
        "recordings": int(events["recording"].nunique()),
        "models": list(MODELS),
        "target": "detector_source_label_only",
        "input": {
            "shape": [CHANNELS, WINDOW_SAMPLES],
            "sampling_hz": TARGET_HZ,
            "window_seconds": WINDOW_S,
            "alignment": "event midpoint, center +/- 5 seconds",
            "channels": CHANNELS,
            "units_before_fixed_scale": "microvolts",
            "normalization": "per-channel window de-mean followed by fixed divide by 100",
            "resampling": "polyphase to 128 Hz",
            "model_specific_filtering": False,
            "model_derived_features": False,
            "event_duration_as_classifier_feature": False,
            "peak_probability_as_classifier_feature": False,
            "qc_features_as_classifier_features": False,
            "hmm_features_as_classifier_features": False,
        },
        "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
        "loss": "class-weighted cross-entropy; weights fit on training events only",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "patient_split": "ordinary GroupKFold(n_splits=5) with GroupShuffleSplit validation inside development fold",
        "grouping_unit": "canonical patient_id",
        "test_used_for_selection": False,
        "test_evaluation_count_per_outer_fold": 1,
        "metrics": ["macro_auroc", "accuracy", "balanced_accuracy", "macro_f1", "per_class_auroc"],
        "bootstrap": {
            "unit": "patient",
            "replicates": int(10 if smoke else bootstrap),
        },
        "raw_cache": cache_meta,
        "fold_metrics": fold_records,
        "pooled_oof_metrics": pooled_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": dataset,
            "stage": "complete",
            "status": manifest["status"],
            "folds_completed": actual_splits,
            "folds_total": actual_splits,
            "events": int(len(events)),
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
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    default_name = f"h3-raw-eeg-cnn-128hz-10s-{args.dataset}-v1"
    out_dir = args.output_dir or (ROOT / "outputs/reports" / default_name)
    run_dataset(
        args.dataset,
        Path(out_dir),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        patience=max(1, int(args.patience)),
        bootstrap=max(1, int(args.bootstrap)),
        limit_events=max(0, int(args.limit_events)),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
