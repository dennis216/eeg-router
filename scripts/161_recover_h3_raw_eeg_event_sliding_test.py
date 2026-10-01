"""Recover the one-time outer-test evaluation after a completed training run.

The companion sliding-window experiment completed training and selected its
checkpoint using validation, but a DataLoader worker ran out of host memory
while reading one very long test event.  This script deliberately performs no
training: it loads that validation checkpoint, reconstructs the frozen patient
split, evaluates the test partition once with a single worker, and writes a
new immutable recovery namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = SCRIPT_DIR / "160_run_h3_raw_eeg_cnn_event_sliding_softmax_one_round.py"
SPEC = importlib.util.spec_from_file_location("h3_sliding_source_161", SOURCE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SOURCE_SCRIPT}")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--encoder-windows", type=int, default=2048)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite existing recovery directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    MOD.BASE.RAW_CACHE_ROOT = Path(args.cache_root)
    source_events = args.source_dir / "raw_eeg_cnn_event_sliding_events_used.parquet"
    source_split = args.source_dir / "raw_eeg_cnn_event_sliding_patient_split.csv"
    source_checkpoint = args.source_dir / "raw_eeg_cnn_event_sliding_softmax_best_validation.pt"
    if not source_events.exists() or not source_split.exists() or not source_checkpoint.exists():
        raise FileNotFoundError("source events, patient split, or validation checkpoint is missing")

    started = time.monotonic()
    events = pd.read_parquet(source_events).reset_index(drop=True)
    split_frame = pd.read_csv(source_split)
    test_patients = set(split_frame.loc[split_frame["split"] == "test", "patient_id"].astype(str))
    train_patients = set(split_frame.loc[split_frame["split"] == "train", "patient_id"].astype(str))
    validation_patients = set(split_frame.loc[split_frame["split"] == "validation", "patient_id"].astype(str))
    if not test_patients or train_patients & test_patients or validation_patients & test_patients:
        raise RuntimeError("patient split audit failed during recovery")
    test_idx = np.flatnonzero(events["patient_id"].astype(str).isin(test_patients).to_numpy())
    train_idx = np.flatnonzero(events["patient_id"].astype(str).isin(train_patients).to_numpy())
    validation_idx = np.flatnonzero(events["patient_id"].astype(str).isin(validation_patients).to_numpy())
    if len(test_idx) == 0 or len(train_idx) == 0 or len(validation_idx) == 0:
        raise RuntimeError("reconstructed patient split is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = MOD.SlidingEventCNN(encoder_windows=args.encoder_windows).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    counts = events.iloc[train_idx]["model"].astype(str).map(MOD.LABELS).value_counts()
    count_array = np.asarray([float(counts.get(i, 0)) for i in range(3)], dtype=float)
    class_weights = count_array.sum() / (3.0 * count_array)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    # Single worker is intentional: it prevents prefetching multiple giant
    # complete-event arrays while keeping the frozen input and test protocol.
    test_loader = MOD.dataloader_for(
        events,
        test_idx,
        MOD.BASE.RAW_CACHE_ROOT,
        max_windows_per_batch=2048,
        shuffle=False,
        seed=20260830,
        num_workers=0,
    )
    test_metrics, predictions, test_loss = MOD.evaluate(model, test_loader, device, criterion)
    metadata_cols = [
        "source_row", "event_id", "patient_id", "recording", "relative_path", "model",
        "start_s", "end_s", "event_duration_s", "window_count",
    ]
    predictions = predictions.merge(
        events.iloc[test_idx][metadata_cols],
        on=["source_row", "event_id", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    predictions["fold"] = MOD.OUTER_FOLD
    predictions["activation"] = "softmax"
    pred_path = args.output_dir / "raw_eeg_cnn_event_sliding_fold0_test_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)

    metrics_frame = pd.DataFrame([{
        "dataset": "chbmit", "fold": MOD.OUTER_FOLD, "seed": int(checkpoint.get("seed", MOD.BASE_SEED)),
        "status": "complete", "train_events": int(len(train_idx)),
        "validation_events": int(len(validation_idx)), "test_events": int(len(test_idx)),
        "train_windows": int(events.iloc[train_idx]["window_count"].sum()),
        "validation_windows": int(events.iloc[validation_idx]["window_count"].sum()),
        "test_windows": int(events.iloc[test_idx]["window_count"].sum()),
        "train_patients": int(len(train_patients)), "validation_patients": int(len(validation_patients)),
        "test_patients": int(len(test_patients)), "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "test_loss": float(test_loss), "test_evaluation_count": 1,
        "test_used_for_selection": False, "activation": "softmax",
        "aggregation": "mean of all within-event z_t", "loss": "class-weighted CrossEntropyLoss",
        "checkpoint": "raw_eeg_cnn_event_sliding_softmax_best_validation.pt",
        "checkpoint_sha256": sha256(source_checkpoint),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }])
    metrics_frame.to_csv(args.output_dir / "raw_eeg_cnn_event_sliding_fold_metrics.csv", index=False)
    pooled = MOD.metrics_from_arrays(
        predictions["y"].to_numpy(dtype=np.int64),
        predictions[[f"prob_{name}" for name in MOD.MODELS]].to_numpy(float),
    )
    pd.DataFrame([{"dataset": "chbmit", "evaluation": "outer_fold0_test", "folds": 1,
                   "events": int(len(predictions)), "sliding_windows": int(events.iloc[test_idx]["window_count"].sum()),
                   "patients": int(predictions["patient_id"].nunique()), "chance_macro_auroc": 0.5, **pooled}
                  ]).to_csv(args.output_dir / "raw_eeg_cnn_event_sliding_summary_metrics.csv", index=False)

    boot = MOD.bootstrap_patients(predictions, int(args.bootstrap), int(MOD.BASE_SEED) + 99)
    boot.to_csv(args.output_dir / "raw_eeg_cnn_event_sliding_patient_bootstrap.csv", index=False)
    summary = (
        boot.groupby("metric", as_index=False)["value"]
        .agg(mean="mean", q025=lambda values: float(np.nanquantile(values, 0.025)),
             q975=lambda values: float(np.nanquantile(values, 0.975)))
    )
    summary.to_csv(args.output_dir / "raw_eeg_cnn_event_sliding_patient_bootstrap_summary.csv", index=False)
    shutil.copy2(source_events, args.output_dir / source_events.name)
    shutil.copy2(source_split, args.output_dir / source_split.name)
    shutil.copy2(source_checkpoint, args.output_dir / "raw_eeg_cnn_event_sliding_softmax_best_validation.pt")
    history = args.source_dir / "raw_eeg_cnn_event_sliding_training_history.csv"
    if history.exists():
        shutil.copy2(history, args.output_dir / history.name)

    source_table = MOD.ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h2h3_events.parquet"
    source_table_hash = sha256(source_table) if source_table.exists() else None
    total_windows = int(events["window_count"].sum())
    manifest = {
        "analysis_id": args.output_dir.name,
        "dataset": "chbmit", "status": "complete", "formal": True,
        "recovery_only": True, "recovered_from": str(args.source_dir),
        "source_run": "v3-groupkfold-confirmatory-v1",
        "source_event_table": str(source_table), "source_event_table_sha256": source_table_hash,
        "source_rows_after_fp_filter": 174893, "events_used": int(len(events)),
        "sliding_windows_used": total_windows, "patients": int(events["patient_id"].nunique()),
        "recordings": int(events["recording"].nunique()),
        "labels": {name: int(MOD.LABELS[name]) for name in MOD.MODELS},
        "target": "single detector-source label per retained FP event",
        "input": {"shape": [16, MOD.WINDOW_SAMPLES], "sampling_hz": MOD.TARGET_HZ,
                   "window_seconds": MOD.WINDOW_S, "window_stride_seconds": MOD.WINDOW_STRIDE_S,
                   "window_coverage": "all windows covering each complete event; endpoint window forced",
                   "short_event_rule": "one midpoint-centred 10-s window", "channels": MOD.CHANNELS,
                   "raw_eeg_only": True,
                   "normalization": "per-channel window de-mean followed by fixed divide by 100",
                   "detector_probability_feature": False, "event_duration_feature": False,
                   "qc_feature": False, "hmm_feature": False},
        "architecture": "shared Conv1D(16->32)+BN+ReLU -> Conv1D(32->64)+BN+ReLU -> Conv1D(64->128)+BN+ReLU -> GlobalAvgPool -> mean event aggregation -> Linear(128,3)",
        "aggregation": "arithmetic mean of all within-event encoder embeddings z_t",
        "output_activation": "softmax", "loss": "class-weighted CrossEntropyLoss; weights fit on train events only",
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "runtime_optimization": {"max_windows_per_optimizer_batch": 2048, "encoder_chunk_windows": int(args.encoder_windows),
                                 "training_data_loader_workers": 2, "test_data_loader_workers": 0,
                                 "recording_local_batch_order": True, "activation_checkpointing": True,
                                 "mixed_precision": bool(torch.cuda.is_available())},
        "overlap_handling": "ignored; every source FP row retained as an independent single-label sample; no overlap-based filtering or merging",
        "patient_split": "ordinary GroupKFold(n_splits=5), outer fold 0 only; GroupShuffleSplit validation inside development patients",
        "grouping_unit": "canonical patient_id", "outer_fold_run": MOD.OUTER_FOLD, "outer_folds_total": MOD.N_SPLITS,
        "test_used_for_selection": False, "threshold_selection": "none; softmax argmax", "test_evaluation_count": 1,
        "metrics": ["macro_auroc", "per_class_auroc", "accuracy", "balanced_accuracy", "macro_f1"],
        "bootstrap": {"unit": "patient", "replicates": int(args.bootstrap)},
        "class_weight_values_train": class_weights.tolist(),
        "checkpoint": {"best_validation": "raw_eeg_cnn_event_sliding_softmax_best_validation.pt",
                        "best_validation_sha256": sha256(source_checkpoint)},
        "raw_cache": {"cache_root": str(MOD.BASE.RAW_CACHE_ROOT)},
        "pooled_test_metrics": pooled, "fold_record": metrics_frame.iloc[0].to_dict(),
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (args.output_dir / "progress.json").write_text(json.dumps({"dataset": "chbmit", "stage": "complete",
        "status": "complete", "recovery_only": True, "folds_completed": 1, "folds_total": MOD.N_SPLITS,
        "events": int(len(events)), "sliding_windows": total_windows,
        "test_evaluation_count": 1, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "status": "complete", "test_metrics": pooled,
                      "test_events": int(len(predictions)), "test_patients": int(len(test_patients)),
                      "bootstrap_replicates": int(args.bootstrap)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
