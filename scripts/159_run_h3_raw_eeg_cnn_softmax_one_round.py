"""One-round CHB-MIT raw-EEG detector-source classifier (softmax version).

This is a fresh, explicitly single-label rerun of the raw-EEG H3 experiment.
Each detector false-positive row is retained as an independent sample and is
assigned one label (``singlem``, ``cbramod`` or ``tcn_gat``).  Overlapping
events are intentionally *not* merged, deduplicated, or excluded in this
analysis; the overlap decision is recorded in the manifest and is not an
input to the classifier.

The classifier sees only a 10-s, 16-channel, 128-Hz waveform (16 x 1280),
with the fixed per-window de-mean and scale used by the raw-EEG utility.  The
head is a conventional three-class softmax head trained with class-weighted
cross entropy.  No detector probabilities, duration, QC, HMM, or event
metadata are fed to the network.

Only outer fold 0 of an ordinary patient-level GroupKFold(5) is run.  A
patient-level GroupShuffleSplit within the development patients supplies
validation for early stopping.  The outer test is evaluated exactly once
after loading the best validation checkpoint.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
BASE_PATH = Path(__file__).resolve().parent / "157_run_h3_raw_eeg_cnn_fixed10s.py"
SPEC = importlib.util.spec_from_file_location("bfa_raw_eeg_base_159", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import raw-EEG utilities from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: index for index, name in enumerate(MODELS)}
N_SPLITS = 5
OUTER_FOLD = 0
BASE_SEED = 20260828
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 128
DEFAULT_PATIENCE = 3
DEFAULT_BOOTSTRAP = 2000


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


def select_smoke(events: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select a small multi-patient, all-class smoke set without changing formal data."""
    patients = sorted(events["patient_id"].astype(str).unique().tolist())[:5]
    per_class = max(1, int(limit // max(1, len(patients) * len(MODELS))))
    pieces: list[pd.DataFrame] = []
    for patient in patients:
        for model in MODELS:
            pieces.append(
                events[
                    (events["patient_id"].astype(str) == patient)
                    & (events["model"].astype(str) == model)
                ].head(per_class)
            )
    selected = pd.concat(pieces, ignore_index=True) if pieces else events.head(0).copy()
    if len(selected) > limit:
        selected = selected.head(int(limit)).copy()
    if selected["patient_id"].nunique() < min(5, events["patient_id"].nunique()):
        raise RuntimeError("smoke selection did not retain five patient groups")
    if set(selected["model"].astype(str)) != set(MODELS):
        raise RuntimeError("smoke selection did not retain all three detector labels")
    return selected.reset_index(drop=True)


def make_split(
    events: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    indices = np.arange(len(events), dtype=np.int64)
    groups = events["patient_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=N_SPLITS)
    outer_splits = list(outer.split(indices, groups=groups))
    development, test = outer_splits[OUTER_FOLD]
    development = np.asarray(development, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    train: np.ndarray | None = None
    validation: np.ndarray | None = None
    for attempt in range(50):
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(seed + attempt),
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
    rows: list[dict[str, Any]] = []
    for split_name, split_indices in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for patient in sorted(events.iloc[split_indices]["patient_id"].astype(str).unique()):
            rows.append(
                {
                    "split": split_name,
                    "patient_id": patient,
                    "event_count": int(
                        (events.iloc[split_indices]["patient_id"].astype(str) == patient).sum()
                    ),
                }
            )
    split_frame = pd.DataFrame(rows)
    return train, validation, test, split_frame


def metrics_from_arrays(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return BASE.metrics_from_arrays(y, probabilities)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    return BASE.evaluate_loader(model, loader, device, criterion)


def bootstrap_patients(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
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
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
    bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    seed_everything(seed)
    train_idx, validation_idx, test_idx, split_frame = make_split(events, seed)
    split_frame.to_csv(out_dir / "raw_eeg_cnn_softmax_patient_split.csv", index=False)
    train_loader = BASE.dataloader_for(events, train_idx, "chbmit", dataset_root, batch_size, True, seed)
    validation_loader = BASE.dataloader_for(
        events, validation_idx, "chbmit", dataset_root, batch_size, False, seed + 1
    )
    test_loader = BASE.dataloader_for(events, test_idx, "chbmit", dataset_root, batch_size, False, seed + 2)
    model = BASE.RawEEGSourceCNN().to(device)
    counts = events.iloc[train_idx]["model"].astype(str).map(LABELS).value_counts()
    count_array = np.asarray([float(counts.get(i, 0)) for i in range(3)], dtype=float)
    if np.any(count_array <= 0):
        raise RuntimeError(f"training labels missing a class: {count_array.tolist()}")
    class_weight_values = count_array.sum() / (3.0 * count_array)
    class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        batches = 0
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
            total_loss += float(loss.detach().cpu())
            batches += 1
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
                    "architecture": "Conv1D(16->32)->BN->ReLU->Conv1D(32->64)->BN->ReLU->Conv1D(64->128)->BN->ReLU->GlobalAvgPool->Linear(128,3)",
                    "head": "softmax at inference; class-weighted CrossEntropyLoss",
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "fold": OUTER_FOLD,
                    "seed": seed,
                    "input_shape": [16, 1280],
                    "model_derived_features": False,
                },
                out_dir / "raw_eeg_cnn_softmax_singlelabel_best_validation.pt",
            )
        history.append(
            {
                "fold": OUTER_FOLD,
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
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
                "validation_events": int(len(validation_idx)),
                "test_events": int(len(test_idx)),
                "activation": "softmax",
                "loss": "class_weighted_cross_entropy",
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
            ["source_row", "event_id", "patient_id", "recording", "relative_path", "model", "start_s", "end_s"]
        ],
        on="source_row",
        how="left",
        validate="one_to_one",
    )
    test_predictions["fold"] = OUTER_FOLD
    test_predictions["activation"] = "softmax"
    test_predictions.to_parquet(out_dir / "raw_eeg_cnn_softmax_singlelabel_fold0_test_predictions.parquet", index=False)
    checkpoint_payload = {
        "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
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
        "model_derived_features": False,
        "test_used_for_selection": False,
        "overlap_handling": "ignored; each source FP row retained as an independent single-label sample",
    }
    outer_checkpoint = out_dir / "raw_eeg_cnn_softmax_singlelabel_outer_fold0_checkpoint.pt"
    torch.save(checkpoint_payload, outer_checkpoint)
    pd.DataFrame(history).to_csv(out_dir / "raw_eeg_cnn_softmax_singlelabel_training_history.csv", index=False)
    fold_record: dict[str, Any] = {
        "dataset": "chbmit",
        "fold": OUTER_FOLD,
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
        "activation": "softmax",
        "loss": "class-weighted CrossEntropyLoss",
        "checkpoint": outer_checkpoint.name,
        "checkpoint_sha256": sha256(outer_checkpoint),
    }
    for metric in ("macro_auroc", "accuracy", "balanced_accuracy", "macro_f1"):
        fold_record[f"test_{metric}"] = test_metrics.get(metric, float("nan"))
    for model_name in MODELS:
        fold_record[f"test_{model_name}_auroc"] = test_metrics.get(f"{model_name}_auroc", float("nan"))
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": "chbmit",
            "stage": "fold_complete",
            "fold": OUTER_FOLD,
            "fold_total": N_SPLITS,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "test_macro_auroc": fold_record["test_macro_auroc"],
        },
    )
    return fold_record, test_predictions, {
        "train_indices": train_idx,
        "validation_indices": validation_idx,
        "test_indices": test_idx,
        "class_weights": class_weight_values.tolist(),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    existing = [
        path
        for path in out_dir.iterdir()
        if path.name != "no_duplicate_overlap_preflight.json"
    ] if out_dir.exists() else []
    if existing:
        raise SystemExit(f"refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    events, dataset_root, event_meta = BASE.normalize_events("chbmit")
    original_rows = len(events)
    if args.limit_events > 0:
        events = select_smoke(events, args.limit_events)
    events.to_parquet(out_dir / "raw_eeg_cnn_softmax_singlelabel_events_used.parquet", index=False)
    write_progress(
        out_dir / "progress.json",
        {
            "dataset": "chbmit",
            "stage": "preparing",
            "events": int(len(events)),
            "source_rows_after_fp_filter": int(original_rows),
            "patients": int(events["patient_id"].nunique()),
            "recordings": int(events["recording"].nunique()),
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
                "patients": int(events["patient_id"].nunique()),
                "recordings": int(events["recording"].nunique()),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "input_shape": [16, 1280],
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
        batch_size=min(args.batch_size, 16) if args.smoke else args.batch_size,
        patience=1 if args.smoke else args.patience,
        seed=BASE_SEED,
        device=device,
        bootstrap=10 if args.smoke else args.bootstrap,
    )
    pooled = metrics_from_arrays(
        predictions["y"].to_numpy(dtype=np.int64),
        predictions[[f"prob_{name}" for name in MODELS]].to_numpy(float),
    )
    pd.DataFrame([fold_record]).to_csv(out_dir / "raw_eeg_cnn_softmax_singlelabel_fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "chbmit",
                "evaluation": "outer_fold0_test",
                "folds": 1,
                "events": int(len(predictions)),
                "patients": int(predictions["patient_id"].nunique()),
                # For one-vs-rest AUROC, the chance level is 0.5 even in a
                # three-class macro average (1/3 is the random accuracy level,
                # not the AUROC reference).
                "chance_macro_auroc": 0.5,
                **pooled,
            }
        ]
    ).to_csv(out_dir / "raw_eeg_cnn_softmax_singlelabel_summary_metrics.csv", index=False)
    bootstrap = bootstrap_patients(predictions, 10 if args.smoke else args.bootstrap, BASE_SEED + 9000)
    bootstrap.to_csv(out_dir / "raw_eeg_cnn_softmax_singlelabel_patient_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(out_dir / "raw_eeg_cnn_softmax_singlelabel_patient_bootstrap_summary.csv", index=False)
    formal = not args.smoke and args.limit_events <= 0
    checkpoint_path = out_dir / "raw_eeg_cnn_softmax_singlelabel_outer_fold0_checkpoint.pt"
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
        "patients": int(events["patient_id"].nunique()),
        "recordings": int(events["recording"].nunique()),
        "labels": {name: int(LABELS[name]) for name in MODELS},
        "target": "single detector-source label per retained FP event",
        "input": {
            "shape": [16, 1280],
            "sampling_hz": 128,
            "window_seconds": 10,
            "alignment": "event midpoint centered in +/- 5 seconds",
            "channels": 16,
            "raw_eeg_only": True,
            "normalization": "per-channel window de-mean followed by fixed divide by 100",
            "detector_probability_feature": False,
            "event_duration_feature": False,
            "qc_feature": False,
            "hmm_feature": False,
        },
        "architecture": "Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> Linear(128,3)",
        "output_activation": "softmax",
        "loss": "class-weighted CrossEntropyLoss; weights fit on train events only",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
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
            "best_validation": "raw_eeg_cnn_softmax_singlelabel_best_validation.pt",
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
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/h3-raw-eeg-cnn-128hz-10s-softmax-singlelabel-chb-v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.bootstrap < 1:
        raise SystemExit("epochs, batch-size, patience and bootstrap must be positive")
    run(args)


if __name__ == "__main__":
    main()
