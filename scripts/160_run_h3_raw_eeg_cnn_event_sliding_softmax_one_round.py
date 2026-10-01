"""One-round CHB-MIT event-level sliding-window raw-EEG source classifier.

This experiment keeps the same target and protocol as the fixed-midpoint H3
softmax run, but represents each complete false-positive event by all of its
10-s raw EEG windows.  Windows start at the event boundary and advance with
the detector's 2-s step (from the frozen experiment configuration); the final
window is forced to end at the event boundary so the complete event is
covered.  A short event uses one midpoint-centred window.

Every window from one event passes through the same 1-D CNN encoder.  The
resulting window embeddings z_t are averaged within that event, then a single
Linear(128, 3) softmax head predicts the detector-source label.  No detector
probability, duration, QC, HMM, or other metadata enters the network.  Source
event rows are retained as separate single-label samples and overlap is
explicitly ignored, as requested.

Only outer fold 0 of ordinary patient-level GroupKFold(5) is run.  A
patient-level GroupShuffleSplit inside development patients provides
validation for early stopping; the outer test is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.checkpoint import checkpoint


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
BASE_PATH = Path(__file__).resolve().parent / "157_run_h3_raw_eeg_cnn_fixed10s.py"
SPEC = importlib.util.spec_from_file_location("bfa_raw_eeg_base_160", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import raw-EEG utilities from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: index for index, name in enumerate(MODELS)}
TARGET_HZ = 128.0
WINDOW_S = 10.0
WINDOW_SAMPLES = 1280
CHANNELS = 16
WINDOW_STRIDE_S = 2.0  # frozen detector/inference stride in configs/base.yaml
N_SPLITS = 5
OUTER_FOLD = 0
BASE_SEED = 20260828
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 256
DEFAULT_PATIENCE = 3
DEFAULT_BOOTSTRAP = 2000
DEFAULT_MAX_WINDOWS_PER_BATCH = 256
DEFAULT_CACHE_ROOT = "/mnt/d/EEGData/bfa_h3_raw128_cnn_v1"
DEFAULT_ENCODER_WINDOWS = 512
DEFAULT_NUM_WORKERS = 2


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


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def window_starts(start_s: float, end_s: float) -> np.ndarray:
    """Return starts that cover [start_s, end_s] with 10-s windows and 2-s hop."""
    start = float(start_s)
    end = max(start, float(end_s))
    duration = end - start
    if duration <= WINDOW_S:
        center = (start + end) / 2.0
        return np.asarray([center - WINDOW_S / 2.0], dtype=np.float64)
    last = end - WINDOW_S
    starts = list(np.arange(start, last + 1e-9, WINDOW_STRIDE_S, dtype=np.float64))
    if not starts or starts[-1] < last - 1e-7:
        starts.append(last)
    return np.asarray(starts, dtype=np.float64)


def add_window_counts(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["event_duration_s"] = (
        result["end_s"].astype(float) - result["start_s"].astype(float)
    ).clip(lower=0.0)
    result["window_count"] = [
        int(len(window_starts(row.start_s, row.end_s)))
        for row in result[["start_s", "end_s"]].itertuples(index=False)
    ]
    return result


def select_smoke(events: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Keep five patients and all labels, preferring short events for a quick smoke."""
    patients = sorted(events["patient_id"].astype(str).unique().tolist())[:5]
    pieces: list[pd.DataFrame] = []
    per_class = max(1, int(limit // max(1, len(patients) * len(MODELS))))
    for patient in patients:
        for model in MODELS:
            group = events[
                (events["patient_id"].astype(str) == patient)
                & (events["model"].astype(str) == model)
            ].sort_values(["window_count", "event_duration_s", "source_row"])
            # Include both short and long events in a smoke whenever possible,
            # so that the z_t aggregation path is exercised rather than only
            # the one-window fallback.
            short_n = max(1, per_class // 2)
            long_n = max(0, per_class - short_n)
            selected = pd.concat(
                [group.head(short_n), group.tail(long_n) if long_n else group.head(0)],
                ignore_index=True,
            ).drop_duplicates(subset=["event_id"], keep="first")
            pieces.append(selected)
    selected = pd.concat(pieces, ignore_index=True) if pieces else events.head(0).copy()
    selected = selected.head(int(limit)).reset_index(drop=True)
    if selected["patient_id"].nunique() < min(5, events["patient_id"].nunique()):
        raise RuntimeError("smoke selection did not retain five patient groups")
    if set(selected["model"].astype(str)) != set(MODELS):
        raise RuntimeError("smoke selection did not retain all three detector labels")
    return selected


def make_split(
    events: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    indices = np.arange(len(events), dtype=np.int64)
    groups = events["patient_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=N_SPLITS)
    development, test = list(outer.split(indices, groups=groups))[OUTER_FOLD]
    development = np.asarray(development, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    train: np.ndarray | None = None
    validation: np.ndarray | None = None
    for attempt in range(50):
        inner = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=int(seed + attempt)
        )
        train_local, validation_local = next(
            inner.split(
                development,
                groups=events.iloc[development]["patient_id"].astype(str).to_numpy(),
            )
        )
        candidate_train = development[np.asarray(train_local, dtype=np.int64)]
        candidate_validation = development[np.asarray(validation_local, dtype=np.int64)]
        if set(events.iloc[candidate_train]["model"].astype(str)) == set(MODELS):
            train, validation = candidate_train, candidate_validation
            break
    if train is None or validation is None:
        raise RuntimeError("could not construct all-class patient-level train split")
    split_rows: list[dict[str, Any]] = []
    for name, split_indices in (("train", train), ("validation", validation), ("test", test)):
        subset = events.iloc[split_indices]
        for patient in sorted(subset["patient_id"].astype(str).unique()):
            patient_rows = subset[subset["patient_id"].astype(str) == patient]
            split_rows.append(
                {
                    "split": name,
                    "patient_id": patient,
                    "event_count": int(len(patient_rows)),
                    "window_count": int(patient_rows["window_count"].sum()),
                }
            )
    return train, validation, test, pd.DataFrame(split_rows)


class SlidingEventDataset(Dataset):
    def __init__(self, events: pd.DataFrame, dataset_root: Path, cache_root: Path):
        self.events = events.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.cache_root = Path(cache_root)
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
        path = self.cache_root / "chbmit" / Path(key).with_suffix(".npy")
        raw = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        self._cache[key] = raw
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return raw

    def __getitem__(self, index: int):
        row = self.events.iloc[int(index)]
        raw = self._raw(str(row.relative_path))
        starts = window_starts(float(row.start_s), float(row.end_s))
        # Vectorized sliding extraction.  The earlier prototype called
        # extract_window once per window, which made long events needlessly
        # CPU-bound.  Padding and per-channel de-meaning match the base
        # utility exactly, while advanced indexing keeps all z_t windows in
        # one contiguous event tensor.
        start_samples = np.rint(starts * TARGET_HZ).astype(np.int64)
        min_start = int(start_samples.min()) if len(start_samples) else 0
        max_stop = int(start_samples.max()) + WINDOW_SAMPLES if len(start_samples) else WINDOW_SAMPLES
        left_pad = max(0, -min_start)
        right_pad = max(0, max_stop - int(raw.shape[1]))
        if left_pad or right_pad:
            padded = np.pad(raw, ((0, 0), (left_pad, right_pad)), mode="edge")
        else:
            padded = raw
        starts_padded = start_samples + left_pad
        view = np.lib.stride_tricks.sliding_window_view(
            np.asarray(padded, dtype=np.float32), WINDOW_SAMPLES, axis=-1
        )
        windows = np.ascontiguousarray(view[:, starts_padded, :].transpose(1, 0, 2))
        windows = windows - windows.mean(axis=2, keepdims=True)
        windows = (windows / 100.0).astype(np.float32, copy=False)
        label = LABELS[str(row.model)]
        return (
            torch.from_numpy(np.ascontiguousarray(windows)),
            int(label),
            int(row.source_row),
            str(row.event_id),
            str(row.patient_id),
            int(len(starts)),
        )


class EventBatchSampler(BatchSampler):
    """Batch events while bounding the total number of sliding windows."""

    def __init__(
        self,
        window_counts: Iterable[int],
        max_windows: int,
        shuffle: bool,
        seed: int,
        recording_keys: Iterable[str] | None = None,
    ):
        self.window_counts = np.asarray(list(window_counts), dtype=np.int64)
        self.max_windows = max(1, int(max_windows))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.recording_keys = (
            np.asarray(list(recording_keys), dtype=object)
            if recording_keys is not None
            else np.asarray([""] * len(self.window_counts), dtype=object)
        )
        if len(self.recording_keys) != len(self.window_counts):
            raise ValueError("recording_keys and window_counts must have equal length")
        self.epoch = 0

    def __len__(self) -> int:
        batches = 0
        total = 0
        for count in self.window_counts:
            count = int(count)
            if total and total + count > self.max_windows:
                batches += 1
                total = 0
            total += count
        return batches + int(total > 0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Keep events from the same EDF adjacent so the memory-mapped raw EEG
        # remains hot in the dataset cache.  Shuffle EDFs and events within an
        # EDF each epoch; this preserves stochastic training while avoiding
        # tens of thousands of unnecessary mmap switches.
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(self.recording_keys):
            groups.setdefault(str(key), []).append(index)
        keys = list(groups)
        if self.shuffle:
            rng.shuffle(keys)
            for key in keys:
                rng.shuffle(groups[key])
        else:
            keys.sort()
        order = np.asarray([index for key in keys for index in groups[key]], dtype=np.int64)
        batch: list[int] = []
        total = 0
        for raw_index in order:
            index = int(raw_index)
            count = int(self.window_counts[index])
            if batch and total + count > self.max_windows:
                yield batch
                batch, total = [], 0
            batch.append(index)
            total += count
        if batch:
            yield batch


def collate_events(samples):
    windows, labels, source_rows, event_ids, patients, counts = zip(*samples)
    event_index = torch.cat(
        [torch.full((int(count),), event_number, dtype=torch.long) for event_number, count in enumerate(counts)]
    )
    return (
        torch.cat(list(windows), dim=0),
        event_index,
        torch.tensor(labels, dtype=torch.long),
        list(source_rows),
        list(event_ids),
        list(patients),
        list(counts),
    )


def dataloader_for(
    events: pd.DataFrame,
    indices: np.ndarray,
    dataset_root: Path,
    max_windows_per_batch: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    subset = events.iloc[indices].reset_index(drop=True)
    dataset = SlidingEventDataset(subset, dataset_root, Path(BASE.RAW_CACHE_ROOT))
    sampler = EventBatchSampler(
        subset["window_count"].to_numpy(),
        max_windows_per_batch,
        shuffle,
        seed,
        recording_keys=subset["relative_path"].astype(str).to_numpy(),
    )
    loader_kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_events,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


class SlidingEventCNN(nn.Module):
    """Shared 1-D CNN encoder plus mean event aggregation and softmax head."""

    def __init__(self, encoder_windows: int = DEFAULT_ENCODER_WINDOWS) -> None:
        super().__init__()
        self.encoder_windows = max(1, int(encoder_windows))
        self.encoder = nn.Sequential(
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x)).squeeze(-1)

    def _encode_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch while exposing only pooled embeddings to autograd.

        Complete events can expand to thousands of overlapping windows.  A
        regular forward pass retains every convolution activation until the
        event-level loss is backpropagated, which can exhaust GPU memory.  The
        non-reentrant checkpoint keeps the same network and batch statistics,
        but recomputes the encoder during backward and retains only the pooled
        embeddings.
        """
        return self.pool(self.encoder(x)).squeeze(-1)

    def forward(
        self, windows: torch.Tensor, event_index: torch.Tensor, n_events: int
    ) -> torch.Tensor:
        # A single complete event can contain thousands of overlapping
        # windows.  Encode those windows in bounded chunks, then accumulate
        # their embeddings before applying the event-level classifier.  This
        # preserves the exact arithmetic mean over all windows while avoiding
        # a giant convolution activation tensor for long recordings.
        aggregate: torch.Tensor | None = None
        for start in range(0, int(windows.shape[0]), self.encoder_windows):
            stop = min(int(windows.shape[0]), start + self.encoder_windows)
            chunk = windows[start:stop]
            if self.training and torch.is_grad_enabled():
                z_chunk = checkpoint(self._encode_pooled, chunk, use_reentrant=False)
            else:
                z_chunk = self._encode_pooled(chunk)
            if aggregate is None:
                aggregate = torch.zeros(
                    (n_events, z_chunk.shape[1]), dtype=z_chunk.dtype, device=z_chunk.device
                )
            aggregate = aggregate.index_add(0, event_index[start:stop], z_chunk)
        if aggregate is None:
            raise RuntimeError("cannot classify an empty sliding-window batch")
        counts = torch.bincount(event_index, minlength=n_events).to(aggregate.dtype).clamp_min(1.0)
        event_embedding = aggregate / counts.unsqueeze(1)
        return self.classifier(event_embedding)


def metrics_from_arrays(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return BASE.metrics_from_arrays(y, probabilities)


def evaluate(model, loader, device, criterion):
    model.eval()
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    source_rows: list[int] = []
    event_ids: list[str] = []
    patients: list[str] = []
    losses: list[float] = []
    with torch.no_grad():
        for windows, event_index, targets, row_ids, ids, batch_patients, _counts in loader:
            windows = windows.to(device, non_blocking=True)
            event_index = event_index.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(windows, event_index, len(targets))
            if criterion is not None:
                losses.append(float(criterion(logits, targets).detach().cpu()))
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
            source_rows.extend(row_ids)
            event_ids.extend(ids)
            patients.extend(batch_patients)
    y = np.asarray(labels, dtype=np.int64)
    probs = np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, 3))
    output = pd.DataFrame(
        {
            "source_row": source_rows,
            "event_id": event_ids,
            "patient_id": patients,
            "y": y,
        }
    )
    for index, name in enumerate(MODELS):
        output[f"prob_{name}"] = probs[:, index]
    output["prediction"] = np.argmax(probs, axis=1) if len(probs) else np.empty((0,), dtype=int)
    return (metrics_from_arrays(y, probs) if len(y) else {}), output, float(np.mean(losses)) if losses else float("nan")


def bootstrap_patients(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    patients = np.asarray(sorted(predictions["patient_id"].astype(str).unique()), dtype=object)
    by_patient = {
        patient: np.flatnonzero(predictions["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        selected = np.concatenate([by_patient[patient] for patient in sampled])
        y = predictions.iloc[selected]["y"].to_numpy(dtype=np.int64)
        probs = predictions.iloc[selected][[f"prob_{name}" for name in MODELS]].to_numpy(float)
        metrics = metrics_from_arrays(y, probs)
        for metric in ("macro_auroc", "accuracy", "balanced_accuracy", "macro_f1"):
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


def train_one(
    events: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    epochs: int,
    max_windows_per_batch: int,
    patience: int,
    seed: int,
    device: torch.device,
    encoder_windows: int,
    num_workers: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    seed_everything(seed)
    train_idx, validation_idx, test_idx, split_frame = make_split(events, seed)
    split_frame.to_csv(out_dir / "raw_eeg_cnn_event_sliding_patient_split.csv", index=False)
    train_loader = dataloader_for(events, train_idx, dataset_root, max_windows_per_batch, True, seed, num_workers)
    validation_loader = dataloader_for(events, validation_idx, dataset_root, max_windows_per_batch, False, seed + 1, num_workers)
    test_loader = dataloader_for(events, test_idx, dataset_root, max_windows_per_batch, False, seed + 2, num_workers)
    model = SlidingEventCNN(encoder_windows=encoder_windows).to(device)
    counts = events.iloc[train_idx]["model"].astype(str).map(LABELS).value_counts()
    count_array = np.asarray([float(counts.get(i, 0)) for i in range(3)], dtype=float)
    if np.any(count_array <= 0):
        raise RuntimeError(f"training labels missing a class: {count_array.tolist()}")
    class_weight_values = count_array.sum() / (3.0 * count_array)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
        batch_count = 0
        epoch_started = time.monotonic()
        windows_seen = 0
        for windows, event_index, targets, _row_ids, _ids, _patients, _counts in train_loader:
            windows = windows.to(device, non_blocking=True)
            event_index = event_index.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(windows, event_index, len(targets))
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
            windows_seen += int(windows.shape[0])
            if batch_count % 250 == 0:
                elapsed = max(1e-6, time.monotonic() - epoch_started)
                print(json.dumps({
                    "stage": "train_batch",
                    "epoch": epoch + 1,
                    "batch": batch_count,
                    "windows_seen": windows_seen,
                    "windows_per_second": windows_seen / elapsed,
                    "gpu_memory_gb": (
                        torch.cuda.memory_allocated() / (1024 ** 3)
                        if device.type == "cuda" else 0.0
                    ),
                }, sort_keys=True), flush=True)
        validation_metrics, _val_predictions, validation_loss = evaluate(
            model, validation_loader, device, criterion
        )
        score = validation_metrics.get("macro_auroc", float("nan"))
        if not np.isfinite(score):
            score = -validation_loss if np.isfinite(validation_loss) else -float("inf")
        improved = bool(score > best_score + 1e-6)
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save(
                {
                    "architecture": "shared Conv1D(16->32->64->128)+BN+ReLU+GlobalAvgPool encoder; mean event aggregation; Linear(128,3)",
                    "head": "three-class softmax at inference",
                    "loss": "class-weighted CrossEntropyLoss",
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "fold": OUTER_FOLD,
                    "seed": seed,
                    "input_shape": [16, 1280],
                    "window_stride_seconds": WINDOW_STRIDE_S,
                    "aggregation": "mean of all within-event z_t",
                    "model_derived_features": False,
                },
                out_dir / "raw_eeg_cnn_event_sliding_softmax_best_validation.pt",
            )
        history.append(
            {
                "fold": OUTER_FOLD,
                "epoch": epoch,
                "train_loss": total_loss / max(1, batch_count),
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
                "dataset": "chbmit",
                "stage": "training",
                "fold": OUTER_FOLD,
                "fold_total": N_SPLITS,
                "epoch": epoch + 1,
                "epoch_total": epochs,
                "best_epoch": best_epoch,
                "train_events": int(len(train_idx)),
                "train_windows": int(events.iloc[train_idx]["window_count"].sum()),
                "validation_events": int(len(validation_idx)),
                "validation_windows": int(events.iloc[validation_idx]["window_count"].sum()),
                "test_events": int(len(test_idx)),
                "test_windows": int(events.iloc[test_idx]["window_count"].sum()),
                "activation": "softmax",
                "aggregation": "mean z_t per event",
            },
        )
        print(
            json.dumps(
                {
                    "stage": "epoch",
                    "fold": OUTER_FOLD,
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
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    test_metrics, test_predictions, test_loss = evaluate(model, test_loader, device, criterion)
    test_predictions = test_predictions.merge(
        events.iloc[test_idx][
            ["source_row", "event_id", "patient_id", "recording", "relative_path", "model", "start_s", "end_s", "event_duration_s", "window_count"]
        ],
        on=["source_row", "event_id", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    test_predictions["fold"] = OUTER_FOLD
    test_predictions["activation"] = "softmax"
    test_predictions.to_parquet(out_dir / "raw_eeg_cnn_event_sliding_fold0_test_predictions.parquet", index=False)
    checkpoint_payload = {
        "architecture": "shared Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> mean event aggregation -> Linear(128,3)",
        "head": "three-class softmax",
        "loss": "class-weighted CrossEntropyLoss",
        "state_dict": best_state,
        "fold": OUTER_FOLD,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_patients": sorted(events.iloc[train_idx]["patient_id"].astype(str).unique().tolist()),
        "validation_patients": sorted(events.iloc[validation_idx]["patient_id"].astype(str).unique().tolist()),
        "test_patients": sorted(events.iloc[test_idx]["patient_id"].astype(str).unique().tolist()),
        "input_shape": [16, 1280],
        "window_stride_seconds": WINDOW_STRIDE_S,
        "aggregation": "mean of all within-event embeddings z_t",
        "model_derived_features": False,
        "test_used_for_selection": False,
        "overlap_handling": "ignored; each source FP row retained as an independent single-label event",
    }
    outer_checkpoint = out_dir / "raw_eeg_cnn_event_sliding_outer_fold0_checkpoint.pt"
    torch.save(checkpoint_payload, outer_checkpoint)
    pd.DataFrame(history).to_csv(out_dir / "raw_eeg_cnn_event_sliding_training_history.csv", index=False)
    fold_record: dict[str, Any] = {
        "dataset": "chbmit",
        "fold": OUTER_FOLD,
        "seed": seed,
        "status": "complete",
        "train_events": int(len(train_idx)),
        "train_windows": int(events.iloc[train_idx]["window_count"].sum()),
        "validation_events": int(len(validation_idx)),
        "validation_windows": int(events.iloc[validation_idx]["window_count"].sum()),
        "test_events": int(len(test_idx)),
        "test_windows": int(events.iloc[test_idx]["window_count"].sum()),
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
        "activation": "softmax",
        "aggregation": "mean of all within-event z_t",
        "loss": "class-weighted CrossEntropyLoss",
        "checkpoint": outer_checkpoint.name,
        "checkpoint_sha256": sha256(outer_checkpoint),
    }
    for metric in ("macro_auroc", "accuracy", "balanced_accuracy", "macro_f1"):
        fold_record[f"test_{metric}"] = test_metrics.get(metric, float("nan"))
    for model_name in MODELS:
        fold_record[f"test_{model_name}_auroc"] = test_metrics.get(f"{model_name}_auroc", float("nan"))
    return fold_record, test_predictions, {"class_weights": class_weight_values.tolist()}


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    existing = [
        path for path in out_dir.iterdir() if path.name != "no_duplicate_overlap_preflight.json"
    ] if out_dir.exists() else []
    if existing:
        raise SystemExit(f"refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # The default cache is on a Windows-mounted drive.  For the formal run we
    # allow an identical copy on WSL's local filesystem, which avoids 9p
    # random-read latency without changing any waveform or preprocessing.
    BASE.RAW_CACHE_ROOT = Path(args.cache_root)
    started = time.monotonic()
    events, dataset_root, event_meta = BASE.normalize_events("chbmit")
    events = add_window_counts(events)
    original_rows = len(events)
    if args.limit_events > 0:
        events = select_smoke(events, args.limit_events)
    events.to_parquet(out_dir / "raw_eeg_cnn_event_sliding_events_used.parquet", index=False)
    total_windows = int(events["window_count"].sum())
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": "chbmit",
            "stage": "preparing",
            "events": int(len(events)),
            "source_rows_after_fp_filter": int(original_rows),
            "patients": int(events["patient_id"].nunique()),
            "recordings": int(events["recording"].nunique()),
            "sliding_windows": total_windows,
            "window_seconds": WINDOW_S,
            "window_stride_seconds": WINDOW_STRIDE_S,
            "overlap_handling": "ignored; no event merging, deduplication, or exclusion",
        },
    )
    cache_meta = BASE.ensure_raw128_cache("chbmit", events, dataset_root, out_dir / "progress.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "stage": "start",
                "dataset": "chbmit",
                "events": len(events),
                "source_rows_after_fp_filter": original_rows,
                "sliding_windows": total_windows,
                "patients": int(events["patient_id"].nunique()),
                "recordings": int(events["recording"].nunique()),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "input_shape": [16, 1280],
                "window_stride_seconds": WINDOW_STRIDE_S,
                "aggregation": "mean of all within-event z_t",
                "activation": "softmax",
                "overlap_handling": "ignored",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    fold_record, predictions, split_info = train_one(
        events,
        dataset_root,
        out_dir,
        epochs=1 if args.smoke else args.epochs,
        max_windows_per_batch=max(1, args.max_windows_per_batch),
        patience=1 if args.smoke else args.patience,
        seed=BASE_SEED,
        device=device,
        encoder_windows=max(1, args.encoder_windows),
        num_workers=max(0, args.num_workers),
    )
    pooled = metrics_from_arrays(
        predictions["y"].to_numpy(dtype=np.int64),
        predictions[[f"prob_{name}" for name in MODELS]].to_numpy(float),
    )
    pd.DataFrame([fold_record]).to_csv(out_dir / "raw_eeg_cnn_event_sliding_fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "chbmit",
                "evaluation": "outer_fold0_test",
                "folds": 1,
                "events": int(len(predictions)),
                "sliding_windows": int(events.iloc[make_split(events, BASE_SEED)[2]]["window_count"].sum()),
                "patients": int(predictions["patient_id"].nunique()),
                "chance_macro_auroc": 0.5,
                **pooled,
            }
        ]
    ).to_csv(out_dir / "raw_eeg_cnn_event_sliding_summary_metrics.csv", index=False)
    bootstrap = bootstrap_patients(predictions, 10 if args.smoke else args.bootstrap, BASE_SEED + 9000)
    bootstrap.to_csv(out_dir / "raw_eeg_cnn_event_sliding_patient_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(out_dir / "raw_eeg_cnn_event_sliding_patient_bootstrap_summary.csv", index=False)
    formal = not args.smoke and args.limit_events <= 0
    checkpoint_path = out_dir / "raw_eeg_cnn_event_sliding_outer_fold0_checkpoint.pt"
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": "chbmit",
        "status": "complete" if formal else "smoke",
        "formal": formal,
        "source_run": "v3-groupkfold-confirmatory-v1",
        "source_event_table": event_meta["source_event_table"],
        "source_event_table_sha256": event_meta["source_event_table_sha256"],
        "source_rows_after_fp_filter": original_rows,
        "events_used": int(len(events)),
        "sliding_windows_used": total_windows,
        "patients": int(events["patient_id"].nunique()),
        "recordings": int(events["recording"].nunique()),
        "labels": {name: int(LABELS[name]) for name in MODELS},
        "target": "single detector-source label per retained FP event",
        "input": {
            "shape": [16, WINDOW_SAMPLES],
            "sampling_hz": TARGET_HZ,
            "window_seconds": WINDOW_S,
            "window_stride_seconds": WINDOW_STRIDE_S,
            "window_coverage": "all windows covering each complete event; endpoint window forced",
            "short_event_rule": "one midpoint-centred 10-s window",
            "channels": CHANNELS,
            "raw_eeg_only": True,
            "normalization": "per-channel window de-mean followed by fixed divide by 100",
            "detector_probability_feature": False,
            "event_duration_feature": False,
            "qc_feature": False,
            "hmm_feature": False,
        },
        "architecture": "shared Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> mean event aggregation -> Linear(128,3)",
        "aggregation": "arithmetic mean of all within-event encoder embeddings z_t",
        "output_activation": "softmax",
        "loss": "class-weighted CrossEntropyLoss; weights fit on train events only",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "runtime_optimization": {
            "max_windows_per_optimizer_batch": int(args.max_windows_per_batch),
            "encoder_chunk_windows": int(args.encoder_windows),
            "data_loader_workers": int(args.num_workers),
            "recording_local_batch_order": True,
            "activation_checkpointing": True,
            "mixed_precision": bool(torch.cuda.is_available()),
        },
        "overlap_handling": "ignored; every source FP row retained as an independent single-label sample; no overlap-based filtering or merging",
        "patient_split": "ordinary GroupKFold(n_splits=5), outer fold 0 only; GroupShuffleSplit validation inside development patients",
        "grouping_unit": "canonical patient_id",
        "outer_fold_run": OUTER_FOLD,
        "outer_folds_total": N_SPLITS,
        "test_used_for_selection": False,
        "threshold_selection": "none; softmax argmax",
        "test_evaluation_count": 1,
        "metrics": ["macro_auroc", "per_class_auroc", "accuracy", "balanced_accuracy", "macro_f1"],
        "bootstrap": {"unit": "patient", "replicates": int(10 if args.smoke else args.bootstrap)},
        "class_weight_values_train": split_info["class_weights"],
        "checkpoint": {
            "best_validation": "raw_eeg_cnn_event_sliding_softmax_best_validation.pt",
            "outer_fold0": checkpoint_path.name,
            "outer_fold0_sha256": sha256(checkpoint_path),
        },
        "raw_cache": cache_meta,
        "pooled_test_metrics": pooled,
        "fold_record": fold_record,
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": "chbmit",
            "stage": "complete",
            "status": manifest["status"],
            "folds_completed": 1,
            "folds_total": N_SPLITS,
            "events": int(len(events)),
            "sliding_windows": total_windows,
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/h3-raw-eeg-cnn-event-sliding-softmax-singlelabel-chb-v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-windows-per-batch", type=int, default=DEFAULT_MAX_WINDOWS_PER_BATCH)
    parser.add_argument("--encoder-windows", type=int, default=DEFAULT_ENCODER_WINDOWS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.max_windows_per_batch < 1
        or args.encoder_windows < 1
        or args.num_workers < 0
        or args.patience < 1
        or args.bootstrap < 1
    ):
        raise SystemExit("epochs, batch/chunk sizes, patience and bootstrap must be positive; workers cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
