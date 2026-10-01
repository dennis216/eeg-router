from __future__ import annotations

"""Run the compact-v3 conventional baselines.

This runner intentionally uses the same frozen CHB-MIT window manifest, patient
splits, eventization, validation-only threshold selection, and single-use test
evaluation as the v3 unified runs.  It adds two auditable comparators:

* ``eegnet``: a small EEGNet-style convolutional classifier trained on the
  current 10-second, 16-channel window.
* ``psd_catboost``: deterministic band-power/PSD features followed by CatBoost.

The script is deliberately self-contained rather than changing the existing
v3 runner.  That keeps the baseline addendum independent and makes it possible
to audit its input hashes and test access separately.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.evaluation.metrics import ThresholdScore, select_threshold
from bfa.training.sampler import PatientBalancedSampler


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "experiment" / "v3_unified.yaml"
RATE = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = RATE * WINDOW_SECONDS
CHANNELS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only keeps the runner usable across CUDA/cuDNN versions while still
    # recording all seeds in the run manifest.
    torch.use_deterministic_algorithms(True, warn_only=True)


def indexes_for_split(
    windows: pd.DataFrame, split: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = windows[
        windows.patient.isin(split["train"]) & windows.train_eligible
    ].reset_index(drop=True)
    validation = windows[
        windows.patient.isin(split["validation"]) & ~windows.warmup
    ].reset_index(drop=True)
    test = windows[windows.patient.isin(split["test"]) & ~windows.warmup].reset_index(drop=True)
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("all split partitions must have non-empty indexes")
    if set(train.patient) & set(validation.patient):
        raise ValueError("patient leakage from train to validation")
    if set(train.patient) & set(test.patient):
        raise ValueError("patient leakage from train to test")
    if set(validation.patient) & set(test.patient):
        raise ValueError("patient leakage from validation to test")
    if train.label.isna().any():
        raise ValueError("training index contains unlabeled rows")
    return train, validation, test


def balanced_smoke(index: pd.DataFrame, n_each: int = 32) -> pd.DataFrame:
    positive = index[index.label == 1].head(n_each)
    negative = index[index.label == 0].head(n_each)
    output = pd.concat([positive, negative], ignore_index=True)
    if output.label.nunique() != 2:
        raise RuntimeError("smoke validation subset must contain both classes")
    return output


class RawWindowDataset(Dataset[dict[str, object]]):
    """Read the current ten-second window from the frozen TCN signal cache."""

    def __init__(
        self,
        index: pd.DataFrame,
        cache_root: Path,
        *,
        normalize: bool,
        max_open_recordings: int = 32,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.cache_root = Path(cache_root) / "tcn_gat"
        self.normalize = normalize
        self.max_open_recordings = max_open_recordings
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, relative_path: str) -> np.ndarray:
        if relative_path not in self._arrays:
            path = self.cache_root / Path(relative_path).with_suffix(".npy")
            self._arrays[relative_path] = np.load(path, mmap_mode="r", allow_pickle=False)
            if len(self._arrays) > self.max_open_recordings:
                self._arrays.pop(next(iter(self._arrays)))
        return self._arrays[relative_path]

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.index.iloc[item]
        relative_path = str(row.relative_path)
        signal = self._array(relative_path)
        start = int(round(float(row.start) * RATE))
        end = start + WINDOW_SAMPLES
        segment = np.asarray(signal[:, start:end], dtype=np.float32)
        if segment.shape != (CHANNELS, WINDOW_SAMPLES):
            padded = np.zeros((CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
            copy_len = min(WINDOW_SAMPLES, segment.shape[-1]) if segment.ndim == 2 else 0
            if segment.ndim == 2 and segment.shape[0] == CHANNELS and copy_len:
                padded[:, :copy_len] = segment[:, :copy_len]
            segment = padded
        if self.normalize:
            mean = segment.mean(axis=-1, keepdims=True)
            std = segment.std(axis=-1, keepdims=True)
            segment = (segment - mean) / (std + 1e-6)
            segment = np.clip(segment, -8.0, 8.0)
        label = -1.0 if pd.isna(row.label) else float(row.label)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(segment)),
            "y": torch.tensor(label, dtype=torch.float32),
            "row_id": int(item),
        }


class EEGNet(nn.Module):
    """Compact EEGNet-like network for a 16-channel ten-second window."""

    def __init__(self, channels: int = CHANNELS) -> None:
        super().__init__()
        f1, depth, f2 = 8, 2, 16
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, kernel_size=(channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(
                f1 * depth,
                f1 * depth,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth,
                bias=False,
            ),
            nn.Conv2d(f1 * depth, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.unsqueeze(1))
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(-1)


def feature_names() -> list[str]:
    names: list[str] = []
    for channel in range(CHANNELS):
        names.extend([f"ch{channel:02d}_log_std", f"ch{channel:02d}_log_rms"])
        names.extend(
            f"ch{channel:02d}_rel_band_{band}"
            for band in ("delta", "theta", "alpha", "beta", "gamma")
        )
    names.extend(f"global_mean_rel_band_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma"))
    names.extend(f"global_std_rel_band_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma"))
    return names


FEATURE_NAMES = feature_names()


def psd_features(batch: np.ndarray) -> np.ndarray:
    """Return channel-level log amplitude and relative 0.5--45 Hz band power."""
    x = np.asarray(batch, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    z = (x - mean) / (std + 1e-6)
    spectrum = np.abs(np.fft.rfft(z, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[-1], d=1.0 / RATE)
    bands = (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0))
    total_mask = (frequencies >= 0.5) & (frequencies <= 45.0)
    total = spectrum[..., total_mask].sum(axis=-1) + 1e-8
    band_values = []
    for _name, low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        band_values.append(spectrum[..., mask].sum(axis=-1) / total)
    rel = np.stack(band_values, axis=-1)
    log_std = np.log(np.std(x, axis=-1) + 1e-5)
    log_rms = np.log(np.sqrt(np.mean(x * x, axis=-1)) + 1e-5)
    channel_features = np.concatenate([log_std[..., None], log_rms[..., None], rel], axis=-1)
    global_mean = rel.mean(axis=1)
    global_std = rel.std(axis=1)
    return np.concatenate([channel_features.reshape(len(x), -1), global_mean, global_std], axis=-1).astype(np.float32)


def index_signature(index: pd.DataFrame) -> str:
    columns = ["patient", "recording", "start", "end", "label", "warmup", "relative_path"]
    payload = index[columns].to_csv(index=False, lineterminator="\n")
    return sha256_text(payload)


def extract_features(
    index: pd.DataFrame,
    cache_root: Path,
    cache_dir: Path,
    partition: str,
    *,
    workers: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"{partition}.npy"
    index_path = cache_dir / f"{partition}.parquet"
    meta_path = cache_dir / f"{partition}.json"
    signature = index_signature(index)
    if feature_path.exists() and index_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("index_sha256") == signature and meta.get("feature_names") == FEATURE_NAMES:
            return np.load(feature_path, mmap_mode="r"), pd.read_parquet(index_path)
    dataset = RawWindowDataset(index, cache_root, normalize=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    chunks: list[np.ndarray] = []
    started = time.monotonic()
    for batch_number, batch in enumerate(loader, start=1):
        chunks.append(psd_features(batch["x"].numpy()))
        if batch_number % 100 == 0 or batch_number == len(loader):
            print(
                f"BASELINE_FEATURE_PROGRESS partition={partition} batches={batch_number}/{len(loader)} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    features = np.concatenate(chunks, axis=0)
    np.save(feature_path, features)
    index.to_parquet(index_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "partition": partition,
                "rows": len(index),
                "feature_dim": int(features.shape[1]),
                "feature_names": FEATURE_NAMES,
                "index_sha256": signature,
                "feature_sha256": sha256_file(feature_path),
                "sampling_hz": RATE,
                "window_seconds": WINDOW_SECONDS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return features, index


def probability_table(index: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if len(index) != len(probabilities):
        raise ValueError("probabilities do not align with the frozen index")
    output = index[["patient", "recording", "start", "end", "label", "warmup", "relative_path"]].copy()
    output["probability"] = np.asarray(probabilities, dtype=np.float32)
    return output


def truth_intervals(seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def score_probabilities(
    table: pd.DataFrame,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int | None]:
    true_positives = 0
    false_alarms = 0
    truth_events = 0
    nonseizure_seconds = 0.0
    delays: list[float] = []
    for recording_id, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        evaluation_start_s = float(ordered.end.iloc[0])
        recording_rows = recordings[recordings.recording_id == recording_id]
        if recording_rows.empty:
            continue
        duration_s = float(recording_rows.duration_s.iloc[0])
        truths = truth_intervals(seizures, str(recording_id), evaluation_start_s)
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=threshold,
        )
        matched = match_events(predictions, truths)
        true_positives += len(matched.pairs)
        false_alarms += len(matched.unmatched_predictions)
        truth_events += len(truths)
        for pair in matched.pairs:
            delays.append(
                float(predictions[pair.prediction_index].start_s - truths[pair.truth_index][0])
            )
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        nonseizure_seconds += max(0.0, duration_s - evaluation_start_s - seizure_seconds)
    sensitivity = true_positives / truth_events if truth_events else 1.0
    nonseizure_hours = nonseizure_seconds / 3600.0
    return {
        "threshold": float(threshold),
        "true_positive_events": int(true_positives),
        "false_alarm_events": int(false_alarms),
        "truth_events": int(truth_events),
        "event_sensitivity": float(sensitivity),
        "nonseizure_hours": float(nonseizure_hours),
        "fa_per_24h": float(false_alarms * 24.0 / nonseizure_hours) if nonseizure_hours else float("nan"),
        "median_delay_s": float(np.median(delays)) if delays else None,
    }


def threshold_sweep(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [score_probabilities(table, seizures, recordings, threshold) for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2)]
    selected = select_threshold(
        [
            ThresholdScore(
                threshold=float(row["threshold"]),
                event_sensitivity=float(row["event_sensitivity"]),
                fa_per_24h=float(row["fa_per_24h"]),
            )
            for row in rows
        ],
        target_sensitivity=0.80,
    )
    selected_row = next(row for row in rows if row["threshold"] == selected.threshold)
    return pd.DataFrame(rows), selected_row


def evaluate_validation(
    table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, run_dir: Path
) -> dict[str, Any]:
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    probability_path = validation_dir / "probabilities.parquet"
    table.to_parquet(probability_path, index=False)
    labeled = table.label.notna()
    labels = table.loc[labeled, "label"].astype(int)
    probabilities = table.loc[labeled, "probability"]
    if labels.nunique() != 2:
        raise RuntimeError("validation labels must include both classes")
    sweep, selected = threshold_sweep(table, seizures, recordings)
    sweep_path = validation_dir / "threshold_sweep.parquet"
    sweep.to_parquet(sweep_path, index=False)
    metrics = {
        "window_auroc": float(roc_auc_score(labels, probabilities)),
        "window_average_precision": float(average_precision_score(labels, probabilities)),
        "selected_event_operating_point": selected,
        "probability_sha256": sha256_file(probability_path),
        "threshold_sweep_sha256": sha256_file(sweep_path),
    }
    (validation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def evaluate_test(
    table: pd.DataFrame,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    run_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    probability_path = test_dir / "probabilities.parquet"
    if probability_path.exists():
        raise RuntimeError("refusing to evaluate frozen test twice")
    table.to_parquet(probability_path, index=False)
    metrics = {
        "test_evaluation_count": 1,
        "threshold_source": "validation_only",
        "score": score_probabilities(table, seizures, recordings, threshold),
        "probability_sha256": sha256_file(probability_path),
    }
    (test_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_eegnet(
    train_index: pd.DataFrame,
    validation_index: pd.DataFrame,
    test_index: pd.DataFrame,
    cache_root: Path,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    run_dir: Path,
    *,
    split_seed: int,
    model_seed: int,
    updates: int,
    workers: int,
    smoke: bool,
    manifest: dict[str, Any],
) -> None:
    configure_seed(model_seed)
    model = EEGNet().cuda()
    train_dataset = RawWindowDataset(train_index, cache_root, normalize=True)
    sampler = PatientBalancedSampler(
        train_index,
        batch_size=16,
        positive_fraction=0.30,
        seed=model_seed * 1_000_000 + split_seed * 1_000,
        epoch_size=updates * 4 * 16,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    positives = float((train_index.label == 1).sum())
    negatives = float((train_index.label == 0).sum())
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-5)
    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    accumulation = 4
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for _ in range(accumulation):
            batch = next(iterator)
            x = batch["x"].cuda(non_blocking=True)
            y = batch["y"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                raw_loss = functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
                (raw_loss / accumulation).backward()
            losses.append(float(raw_loss.detach().cpu()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite EEGNet gradient norm")
        optimizer.step()
        scheduler.step()
        if update % 25 == 0 or update == updates:
            row = {
                "update": update,
                "loss": float(np.mean(losses)),
                "elapsed_seconds": time.monotonic() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            history.append(row)
            print("BASELINE_EEGNET_PROGRESS " + json.dumps(row, sort_keys=True), flush=True)
    checkpoint = run_dir / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "updates": updates, "manifest": manifest}, checkpoint)
    (checkpoint.with_suffix(".pt.sha256")).write_text(sha256_file(checkpoint) + "\n")
    save_json(run_dir / "training_history.json", {"checkpoints": history})
    model.eval()

    def predict(index: pd.DataFrame) -> np.ndarray:
        dataset = RawWindowDataset(index, cache_root, normalize=True)
        eval_loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for batch_number, batch in enumerate(eval_loader, start=1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs.append(torch.sigmoid(model(batch["x"].cuda(non_blocking=True))).float().cpu().numpy())
                if batch_number % 500 == 0 or batch_number == len(eval_loader):
                    print(f"BASELINE_EEGNET_EVAL_PROGRESS batches={batch_number}/{len(eval_loader)}", flush=True)
        return np.concatenate(outputs)

    val_table = probability_table(validation_index, predict(validation_index))
    val_metrics = evaluate_validation(val_table, seizures, recordings, run_dir)
    threshold = float(val_metrics["selected_event_operating_point"]["threshold"])
    threshold_lock = {
        "source": "validation_only",
        "threshold": threshold,
        "selected_event_operating_point": val_metrics["selected_event_operating_point"],
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_probability_sha256": val_metrics["probability_sha256"],
    }
    save_json(run_dir / "threshold_lock.json", threshold_lock)
    if smoke:
        save_json(run_dir / "completed.json", {"smoke": True, "validation": val_metrics})
        print(f"BASELINE_SMOKE_OK model=eegnet updates={updates}", flush=True)
        return
    manifest["test_accessed"] = True
    manifest["validation_threshold"] = threshold
    save_json(run_dir / "run_manifest.json", manifest)
    test_metrics = evaluate_test(
        probability_table(test_index, predict(test_index)), seizures, recordings, run_dir, threshold
    )
    save_json(run_dir / "completed.json", {"validation": val_metrics, "test": test_metrics, "checkpoint_sha256": sha256_file(checkpoint)})
    print("BASELINE_COMPLETE " + json.dumps({"model": "eegnet", "test": test_metrics}, sort_keys=True), flush=True)


def choose_train_sample(index: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    sampler = PatientBalancedSampler(index, batch_size=1, positive_fraction=0.30, seed=seed, epoch_size=rows)
    positions = list(iter(sampler))
    return index.iloc[positions].reset_index(drop=True)


def train_catboost(
    train_index: pd.DataFrame,
    validation_index: pd.DataFrame,
    test_index: pd.DataFrame,
    cache_root: Path,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    run_dir: Path,
    feature_cache_dir: Path,
    *,
    split_seed: int,
    model_seed: int,
    train_feature_rows: int,
    iterations: int,
    workers: int,
    smoke: bool,
    manifest: dict[str, Any],
) -> None:
    configure_seed(model_seed)
    train_sample = choose_train_sample(train_index, min(train_feature_rows, len(train_index)), split_seed * 1000 + 17)
    x_train, _ = extract_features(train_sample, cache_root, feature_cache_dir, "train_sample", workers=workers)
    x_val, val_index_cached = extract_features(validation_index, cache_root, feature_cache_dir, "validation", workers=workers)
    if len(val_index_cached) != len(validation_index) or index_signature(val_index_cached) != index_signature(validation_index):
        raise RuntimeError("validation feature cache does not match frozen index")
    y_train = train_sample.label.astype(int).to_numpy()
    val_labeled = validation_index.label.notna().to_numpy()
    x_val_labeled = np.asarray(x_val)[val_labeled]
    y_val = validation_index.loc[val_labeled, "label"].astype(int).to_numpy()
    model = CatBoostClassifier(
        iterations=iterations,
        depth=7,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=model_seed,
        thread_count=max(1, workers),
        verbose=50,
        allow_writing_files=False,
        random_strength=1.0,
        l2_leaf_reg=5.0,
    )
    started = time.monotonic()
    model.fit(x_train, y_train, eval_set=(x_val_labeled, y_val), early_stopping_rounds=50)
    print(f"BASELINE_CATBOOST_FIT_COMPLETE seconds={time.monotonic() - started:.1f} best_iteration={model.get_best_iteration()}", flush=True)
    model_path = run_dir / "model.cbm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    (model_path.with_suffix(".cbm.sha256")).write_text(sha256_file(model_path) + "\n")
    save_json(run_dir / "feature_manifest.json", {"feature_names": FEATURE_NAMES, "feature_dim": len(FEATURE_NAMES), "train_sample_rows": len(train_sample), "train_sample_index_sha256": index_signature(train_sample), "cache_dir": str(feature_cache_dir)})
    val_probabilities = model.predict_proba(np.asarray(x_val))[:, 1]
    val_table = probability_table(validation_index, val_probabilities)
    val_metrics = evaluate_validation(val_table, seizures, recordings, run_dir)
    threshold = float(val_metrics["selected_event_operating_point"]["threshold"])
    save_json(
        run_dir / "threshold_lock.json",
        {
            "source": "validation_only",
            "threshold": threshold,
            "selected_event_operating_point": val_metrics["selected_event_operating_point"],
            "model": str(model_path.relative_to(run_dir)),
            "model_sha256": sha256_file(model_path),
            "validation_probability_sha256": val_metrics["probability_sha256"],
        },
    )
    if smoke:
        save_json(run_dir / "completed.json", {"smoke": True, "validation": val_metrics})
        print(f"BASELINE_SMOKE_OK model=psd_catboost iterations={iterations}", flush=True)
        return
    manifest["test_accessed"] = True
    manifest["validation_threshold"] = threshold
    save_json(run_dir / "run_manifest.json", manifest)
    x_test, test_index_cached = extract_features(test_index, cache_root, feature_cache_dir, "test", workers=workers)
    if len(test_index_cached) != len(test_index) or index_signature(test_index_cached) != index_signature(test_index):
        raise RuntimeError("test feature cache does not match frozen index")
    test_probabilities = model.predict_proba(np.asarray(x_test))[:, 1]
    test_metrics = evaluate_test(probability_table(test_index, test_probabilities), seizures, recordings, run_dir, threshold)
    save_json(run_dir / "completed.json", {"validation": val_metrics, "test": test_metrics, "model_sha256": sha256_file(model_path)})
    print("BASELINE_COMPLETE " + json.dumps({"model": "psd_catboost", "test": test_metrics}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["eegnet", "psd_catboost"], required=True)
    parser.add_argument("--split-seed", type=int, choices=[17, 42], required=True)
    parser.add_argument("--model-seed", type=int, choices=[17, 42, 3407], required=True)
    parser.add_argument("--output-namespace", default="v3-compact-baselines")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--catboost-iterations", type=int, default=400)
    parser.add_argument("--train-feature-rows", type=int, default=80000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.model == "eegnet":
        raise RuntimeError("EEGNet runner requires CUDA")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    split_path = ROOT / "manifests" / "splits" / f"split_{args.split_seed}.json"
    split = json.loads(split_path.read_text())
    windows = pd.read_parquet(ROOT / config["inputs"]["window_manifest"])
    recordings = pd.read_parquet(ROOT / config["inputs"]["recordings_manifest"])
    seizures = pd.read_parquet(ROOT / config["inputs"]["seizures_manifest"])
    train_index, validation_index, test_index = indexes_for_split(windows, split)
    smoke = args.smoke_updates > 0
    effective_updates = args.smoke_updates if smoke else args.updates
    if smoke:
        validation_index = balanced_smoke(validation_index)
    config_sha = sha256_file(CONFIG_PATH)
    split_sha = sha256_file(split_path)
    run_id = f"{args.model}_split{args.split_seed}_seed{args.model_seed}"
    namespace = args.output_namespace + ("-smoke" if smoke and not args.output_namespace.endswith("-smoke") else "")
    run_dir = ROOT / "runs" / "baselines" / namespace / args.model / run_id
    if (run_dir / "completed.json").exists():
        print(f"BASELINE_RUN_ALREADY_COMPLETE run_id={run_id}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory exists and is not complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(config["inputs"]["tcn_cache_root"])
    first_relative = Path(str(train_index.relative_path.iloc[0])).with_suffix(".npy")
    required_cache = cache_root / "tcn_gat" / first_relative
    if not required_cache.is_file():
        raise FileNotFoundError(f"frozen TCN cache is absent: {required_cache}")
    manifest: dict[str, Any] = {
        "protocol": config["protocol"],
        "addendum": "v3_compact_baselines",
        "run_id": run_id,
        "model": args.model,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "split_manifest": str(split_path.relative_to(ROOT)),
        "split_sha256": split_sha,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": config_sha,
        "code": str(Path(__file__).relative_to(ROOT)),
        "code_sha256": sha256_file(Path(__file__)),
        "code_commit": git_commit(),
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "updates": effective_updates if args.model == "eegnet" else None,
        "catboost_iterations": args.catboost_iterations if args.model == "psd_catboost" else None,
        "feature_names": FEATURE_NAMES if args.model == "psd_catboost" else None,
        "test_accessed": False,
        "smoke": smoke,
        "started_at_unix": time.time(),
        "window_rate_hz": RATE,
        "window_seconds": WINDOW_SECONDS,
        "threshold_policy": "validation-only target sensitivity 0.80; eventize unchanged from v3",
    }
    save_json(run_dir / "run_manifest.json", manifest)
    print("BASELINE_START " + json.dumps({"run_dir": str(run_dir), "model": args.model, "split_seed": args.split_seed, "model_seed": args.model_seed, "train_rows": len(train_index), "validation_rows": len(validation_index), "test_rows": len(test_index), "updates": effective_updates}, sort_keys=True), flush=True)
    if args.model == "eegnet":
        train_eegnet(train_index, validation_index, test_index, cache_root, seizures, recordings, run_dir, split_seed=args.split_seed, model_seed=args.model_seed, updates=effective_updates, workers=args.workers, smoke=smoke, manifest=manifest)
    else:
        feature_cache_dir = ROOT / "runs" / "baselines" / namespace / "feature_cache" / f"split_{args.split_seed}"
        train_catboost(train_index, validation_index, test_index, cache_root, seizures, recordings, run_dir, feature_cache_dir, split_seed=args.split_seed, model_seed=args.model_seed, train_feature_rows=args.train_feature_rows, iterations=(max(20, args.catboost_iterations // 4) if smoke else args.catboost_iterations), workers=args.workers, smoke=smoke, manifest=manifest)


if __name__ == "__main__":
    main()
