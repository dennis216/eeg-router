"""Run the requested raw-EEG full-window H3 source classifier.

This experiment is deliberately limited to one question: can the source
detector (SingLEM, CBraMod, or TCN-GAT) be identified from EEG alone when the
complete false-positive event interval is represented?  The event duration,
peak probability, model probabilities, QC variables, and any other
detector-derived value are excluded from the classifier input.

The raw signal is converted to one common 16-channel bipolar view at 200 Hz.
For each complete event interval, non-overlapping 2-s frames are summarized
with ten EEG-only statistics (two amplitude, two spatial, five relative-band,
and one high/low-ratio feature); the frame summaries are aggregated over the
whole event interval.  This fixed-size representation makes variable-length
events comparable while retaining the full interval rather than a 2-s event
center crop.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv  # noqa: E402


SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {model: index for index, model in enumerate(MODELS)}
FEATURES = (
    "rms_median_uv",
    "rms_iqr_uv",
    "spatial_rms_dispersion",
    "spatial_rms_max_min_log",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "high_low_ratio",
)
TARGET_HZ = 200.0
FRAME_S = 2.0
FRAME_SAMPLES = int(TARGET_HZ * FRAME_S)
FRAME_BATCH = 128
N_SPLITS = 5
DEFAULT_BOOTSTRAP = 2000
BASE_SEED = 20260828

CHB_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "h2h3_events.parquet"
CHB_RAW_ROOT = Path("/mnt/d/EEGData/bfa_biot_prest_input_v1")
TUSZ_EVENT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
TUSZ_COHORT_PATH = ROOT / "outputs/reports" / SOURCE_RUN / "tusz_groupkfold_external_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _band_masks() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frequencies = np.fft.rfftfreq(FRAME_SAMPLES, d=1.0 / TARGET_HZ)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    masks = {
        name: valid & (frequencies >= low) & (frequencies < high)
        for name, (low, high) in bands.items()
    }
    return frequencies, masks


def _frame_features(raw_200: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame starts, frame centers, and ten EEG-only frame features."""
    if raw_200.ndim != 2 or raw_200.shape[0] != 16:
        raise ValueError(f"expected raw shape (16, samples), got {raw_200.shape}")
    raw_200 = np.asarray(raw_200, dtype=np.float32)
    raw_200 = np.nan_to_num(raw_200, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = int(raw_200.shape[1])
    if n_samples < FRAME_SAMPLES:
        pad = FRAME_SAMPLES - n_samples
        raw_200 = np.pad(raw_200, ((0, 0), (0, pad)), mode="edge")
        n_samples = FRAME_SAMPLES
    starts = np.arange(0, n_samples - FRAME_SAMPLES + 1, FRAME_SAMPLES, dtype=np.int64)
    centers = (starts.astype(np.float64) + FRAME_SAMPLES / 2.0) / TARGET_HZ
    frequencies, masks = _band_masks()
    output = np.empty((len(starts), len(FEATURES)), dtype=np.float32)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)

    for offset in range(0, len(starts), FRAME_BATCH):
        batch_starts = starts[offset : offset + FRAME_BATCH]
        # [channel, frame, sample] -> [frame, channel, sample]
        indices = batch_starts[:, None] + np.arange(FRAME_SAMPLES, dtype=np.int64)[None, :]
        windows = np.asarray(raw_200[:, indices], dtype=np.float32).transpose(1, 0, 2)
        windows = windows - windows.mean(axis=2, keepdims=True)
        rms = np.sqrt(np.mean(windows * windows, axis=2) + 1e-8)
        rms_log = np.log1p(rms)
        spectrum = np.abs(np.fft.rfft(windows, axis=2)) ** 2
        total_power = spectrum[:, :, valid].sum(axis=2) + 1e-8
        band_means: dict[str, np.ndarray] = {}
        for name, mask in masks.items():
            band_power = spectrum[:, :, mask].sum(axis=2)
            band_means[name] = np.mean(band_power / total_power, axis=1)
        values = np.column_stack(
            [
                np.median(rms, axis=1),
                np.quantile(rms, 0.75, axis=1) - np.quantile(rms, 0.25, axis=1),
                np.std(rms_log, axis=1),
                np.max(rms_log, axis=1) - np.min(rms_log, axis=1),
                band_means["delta"],
                band_means["theta"],
                band_means["alpha"],
                band_means["beta"],
                band_means["gamma"],
                (band_means["beta"] + band_means["gamma"])
                / (band_means["delta"] + band_means["theta"] + 1e-8),
            ]
        )
        output[offset : offset + len(batch_starts)] = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
    return starts, centers, output


def _aggregate_interval(
    starts: np.ndarray,
    centers: np.ndarray,
    frame_values: np.ndarray,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Aggregate all frame summaries whose centers fall in the full event."""
    mask = (centers >= float(start_s)) & (centers <= float(end_s))
    if not bool(np.any(mask)):
        midpoint = (float(start_s) + float(end_s)) / 2.0
        nearest = int(np.argmin(np.abs(centers - midpoint)))
        mask[nearest] = True
    values = frame_values[mask]
    # Median for robust amplitude/spatial summaries, mean for spectral ratios.
    result = np.empty(len(FEATURES), dtype=np.float32)
    result[0:4] = np.median(values[:, 0:4], axis=0)
    result[4:9] = np.mean(values[:, 4:9], axis=0)
    result[9] = float((result[7] + result[8]) / (result[4] + result[5] + 1e-8))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _load_raw(dataset: str, relative_path: str, dataset_root: Path) -> np.ndarray:
    if dataset == "chbmit":
        path = CHB_RAW_ROOT / Path(relative_path).with_suffix(".npy")
        if not path.exists():
            raise FileNotFoundError(path)
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.shape[0] != 16:
            raise ValueError(f"CHB raw cache has unexpected shape {raw.shape}: {path}")
        # BIOT cache is canonical bipolar raw EEG at the common 200-Hz rate.
        return np.asarray(raw, dtype=np.float32)
    path = dataset_root / relative_path
    raw_uv, sampling_hz, _ = load_tuh_canonical_raw_uv(path)
    ratio = Fraction(TARGET_HZ / float(sampling_hz)).limit_denominator(10000)
    if ratio.numerator == ratio.denominator == 1:
        raw_200 = raw_uv
    else:
        raw_200 = resample_poly(
            raw_uv.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1
        ).astype(np.float32)
    if raw_200.shape[0] != 16:
        raise ValueError(f"TUSZ canonical raw has unexpected shape {raw_200.shape}: {path}")
    return raw_200


def _recording_features(job: tuple[str, str, str, list[dict[str, Any]], str]) -> dict[str, Any]:
    dataset, recording, relative_path, records, dataset_root_text = job
    try:
        raw = _load_raw(dataset, relative_path, Path(dataset_root_text))
        starts, centers, frame_values = _frame_features(raw)
        interval_cache: dict[tuple[float, float], np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        for record in records:
            key = (round(float(record["start_s"]), 6), round(float(record["end_s"]), 6))
            if key not in interval_cache:
                interval_cache[key] = _aggregate_interval(
                    starts, centers, frame_values, key[0], key[1]
                )
            values = interval_cache[key]
            base = dict(record)
            for index, feature in enumerate(FEATURES):
                base[feature] = float(values[index])
            base["raw_window_definition"] = "complete_event_interval"
            base["raw_sampling_hz"] = float(TARGET_HZ)
            base["frame_s"] = float(FRAME_S)
            base["frame_count"] = int(np.sum((centers >= key[0]) & (centers <= key[1])))
            rows.append(base)
        return {
            "recording": recording,
            "rows": rows,
            "status": "ok",
            "raw_samples": int(raw.shape[1]),
            "frames": int(len(starts)),
            "unique_intervals": int(len(interval_cache)),
        }
    except Exception as exc:  # retain a complete failure record
        return {
            "recording": recording,
            "rows": [],
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=10),
        }


def _prepare_jobs(dataset: str, events: pd.DataFrame) -> tuple[list[tuple[str, str, str, list[dict[str, Any]], str]], dict[str, Any]]:
    if dataset == "chbmit":
        work = events[events["event_type"].astype(str) == "fp"].copy()
        work = work.rename(columns={"patient": "patient_id"})
        work["relative_path"] = work["relative_path"].astype(str)
        dataset_root = Path("/mnt/d/EEGData/chbmit-1.0.0")
        group_columns = ["recording", "relative_path"]
    else:
        work = events[~events["is_true_positive"].astype(bool)].copy()
        work = work.rename(columns={"patient_id": "patient_id"})
        work["relative_path"] = work["recording"].astype(str)
        cohort = json.loads(TUSZ_COHORT_PATH.read_text())
        dataset_root = Path(str(cohort["dataset_root"]))
        group_columns = ["recording", "relative_path"]
    jobs: list[tuple[str, str, str, list[dict[str, Any]], str]] = []
    for group_key, frame in work.groupby(group_columns, sort=True):
        recording, relative_path = (str(group_key[0]), str(group_key[1]))
        if dataset == "tusz" and not (dataset_root / relative_path).exists():
            raise FileNotFoundError(dataset_root / relative_path)
        keep = frame.to_dict("records")
        jobs.append((dataset, recording, relative_path, keep, str(dataset_root)))
    return jobs, {
        "events": int(len(work)),
        "recordings": int(work["recording"].nunique()),
        "patients": int(work["patient_id"].nunique()),
        "dataset_root": str(dataset_root),
    }


def extract_features(dataset: str, out_dir: Path, limit_recordings: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_path = CHB_EVENT_PATH if dataset == "chbmit" else TUSZ_EVENT_PATH
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    events = pd.read_parquet(event_path)
    jobs, prep = _prepare_jobs(dataset, events)
    if limit_recordings > 0:
        jobs = jobs[:limit_recordings]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"dataset": dataset, "jobs": len(jobs), "workers": workers, **prep}, sort_keys=True), flush=True)
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_recording_features(job))
            if index % 10 == 0 or index == len(jobs):
                print(f"RAW_EEG_FEATURES_PROGRESS dataset={dataset} recordings={index}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_recording_features, job) for job in jobs]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                results.append(future.result())
                if index % 10 == 0 or index == len(jobs):
                    print(f"RAW_EEG_FEATURES_PROGRESS dataset={dataset} recordings={index}/{len(jobs)}", flush=True)
    failures = [item for item in results if item.get("status") != "ok"]
    if failures:
        (out_dir / "feature_extraction_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"feature extraction failed for {len(failures)} recordings")
    rows = [row for result in results for row in result.get("rows", [])]
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("raw EEG feature table is empty")
    table = table.sort_values(
        ["patient_id", "recording", "model", "fold", "seed", "start_s", "end_s"],
        kind="mergesort",
    ).reset_index(drop=True)
    feature_path = out_dir / "raw_eeg_full_window_features.parquet"
    table.to_parquet(feature_path, index=False)
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": dataset,
        "source_run": SOURCE_RUN,
        "source_event_table": str(event_path.relative_to(ROOT)),
        "source_event_table_sha256": sha256(event_path),
        "rows": int(len(table)),
        "recordings": int(table["recording"].nunique()),
        "patients": int(table["patient_id"].nunique()),
        "models": sorted(table["model"].astype(str).unique().tolist()),
        "folds": sorted(table["fold"].astype(int).unique().tolist()),
        "seeds": sorted(table["seed"].astype(int).unique().tolist()),
        "fp_only": True,
        "feature_set": "raw_eeg_full_window10",
        "feature_names": list(FEATURES),
        "feature_semantics": {
            "rms_median_uv": "median channel RMS per 2-s frame, then event-window median",
            "rms_iqr_uv": "channel RMS IQR per 2-s frame, then event-window median",
            "spatial_rms_dispersion": "SD of log1p channel RMS per frame, then event-window median",
            "spatial_rms_max_min_log": "log-RMS channel max-min per frame, then event-window median",
            "relative_band_power": "mean channel band/0.5-45-Hz power ratio, averaged over event frames",
            "high_low_ratio": "(beta+gamma)/(delta+theta) after event-window band aggregation",
        },
        "raw_input": {
            "channels": 16,
            "units": "microvolts",
            "common_sampling_hz": TARGET_HZ,
            "preprocessing": "canonical bipolar raw EEG; resampling to 200 Hz only; per-frame de-mean",
            "model_specific_filtering": False,
            "model_derived_features": False,
            "event_window": "complete [start_s, end_s] interval; frame centers within interval",
            "frame_s": FRAME_S,
            "frame_stride_s": FRAME_S,
        },
        "output_sha256": sha256(feature_path),
        "fit_or_selection_performed": False,
    }
    (out_dir / "raw_eeg_full_window_feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return table, manifest


def _center(frame: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    result = frame.copy()
    result.loc[:, FEATURES] = result.loc[:, FEATURES] - result.groupby(patient_col)[list(FEATURES)].transform("mean")
    return result


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, patient_col: str, centered: bool, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_frame = _center(train, patient_col) if centered else train.copy()
    test_frame = _center(test, patient_col) if centered else test.copy()
    train_values = train_frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    test_values = test_frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    test_values = test_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    test_values = test_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    y_train = train_frame["model"].astype(str).map(LABELS).to_numpy(dtype=int)
    y_test = test_frame["model"].astype(str).map(LABELS).to_numpy(dtype=int)
    if len(np.unique(y_train)) != len(MODELS):
        raise ValueError("training partition does not contain all detector classes")
    classifier = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed)
    classifier.fit(scaler.transform(train_values), y_train)
    probabilities = classifier.predict_proba(scaler.transform(test_values))
    predictions = classifier.predict(scaler.transform(test_values))
    output_cols = [patient_col, "recording", "model", "fold", "seed", "start_s", "end_s"]
    output = test_frame[output_cols].copy()
    output = output.rename(columns={patient_col: "patient_id"})
    output["y"] = y_test
    for index, model in enumerate(MODELS):
        output[f"prob_{model}"] = probabilities[:, index]
    output["prediction"] = predictions
    return output, {
        "train_events": int(len(train_frame)),
        "test_events": int(len(test_frame)),
        "classifier": "multinomial_logistic_C1_class_weight_balanced",
        "scaler_fit_partition": "train_only",
        "test_used_for_selection": False,
    }


def _metrics(pred: pd.DataFrame) -> dict[str, float]:
    y = pred["y"].to_numpy(dtype=int)
    probabilities = pred[[f"prob_{model}" for model in MODELS]].to_numpy(dtype=float)
    result: dict[str, float] = {
        "events": float(len(pred)),
        "patients": float(pred["patient_id"].nunique()),
        "accuracy": float(accuracy_score(y, pred["prediction"].to_numpy(dtype=int))),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred["prediction"].to_numpy(dtype=int))),
    }
    try:
        result["macro_auroc"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro", labels=list(range(len(MODELS)))))
    except ValueError:
        result["macro_auroc"] = float("nan")
    return result


def _bootstrap_patient_auc(pred: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    patients = np.asarray(sorted(pred["patient_id"].astype(str).unique()), dtype=object)
    by_patient = {patient: np.flatnonzero(pred["patient_id"].astype(str).to_numpy() == patient) for patient in patients}
    y = pred["y"].to_numpy(dtype=int)
    probabilities = pred[[f"prob_{model}" for model in MODELS]].to_numpy(dtype=float)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in selected])
        try:
            values.append(float(roc_auc_score(y[indices], probabilities[indices], multi_class="ovr", average="macro", labels=list(range(len(MODELS))))))
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), int(len(array))


def _write_progress(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def classify_chb(table: pd.DataFrame, out_dir: Path, n_bootstrap: int) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[pd.DataFrame]] = {"raw": [], "patient_centered": []}
    for fold in range(N_SPLITS):
        fold_frame = table[table["fold"].astype(int) == fold].copy()
        train = fold_frame[fold_frame["partition"].astype(str) == "train"].copy()
        validation = fold_frame[fold_frame["partition"].astype(str) == "validation"].copy()
        test = fold_frame[fold_frame["partition"].astype(str) == "test"].copy()
        if train.empty or validation.empty or test.empty:
            raise RuntimeError(f"CHB empty partition fold {fold}")
        for centering in ("raw", "patient_centered"):
            centered = centering == "patient_centered"
            val_pred, val_info = _fit_predict(train, validation, "patient_id", centered, BASE_SEED + fold)
            test_pred, test_info = _fit_predict(train, test, "patient_id", centered, BASE_SEED + fold + 100)
            val_metric = _metrics(val_pred)
            test_metric = _metrics(test_pred)
            predictions[centering].append(test_pred)
            fold_rows.append({
                "fold": fold,
                "centering": centering,
                "status": "ok",
                "test_used_for_selection": False,
                "chance_macro_auroc": 0.5,
                "validation_macro_auroc": val_metric["macro_auroc"],
                "validation_accuracy": val_metric["accuracy"],
                "validation_balanced_accuracy": val_metric["balanced_accuracy"],
                "test_macro_auroc": test_metric["macro_auroc"],
                "test_accuracy": test_metric["accuracy"],
                "test_balanced_accuracy": test_metric["balanced_accuracy"],
                "train_events": int(len(train)),
                "validation_events": int(len(validation)),
                "test_events": int(len(test)),
                "classifier": test_info["classifier"],
                "scaler_fit_partition": "train_only",
            })
        _write_progress(out_dir, {"dataset": "chbmit", "status": "classifying", "fold_completed": fold + 1, "fold_total": N_SPLITS})
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(out_dir / "raw_eeg_full_window_cv_fold_metrics.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for centering in ("raw", "patient_centered"):
        subset = fold_frame[fold_frame.centering == centering]
        oof = pd.concat(predictions[centering], ignore_index=True)
        metric = _metrics(oof)
        boot = _bootstrap_patient_auc(oof, n_bootstrap, np.random.default_rng(BASE_SEED + (1 if centering == "patient_centered" else 0)))
        summary_rows.append({"dataset": "CHB-MIT", "centering": centering, "folds": N_SPLITS, "events": len(oof), "patients": oof.patient_id.nunique(), **metric, "chance_macro_auroc": 0.5})
        bootstrap_rows.append({"dataset": "CHB-MIT", "centering": centering, "metric": "macro_auroc", "mean": boot[0], "q025": boot[1], "q975": boot[2], "bootstrap_replicates": n_bootstrap, "bootstrap_valid": boot[3], "bootstrap_unit": "patient"})
    pd.DataFrame(summary_rows).to_csv(out_dir / "raw_eeg_full_window_summary_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out_dir / "raw_eeg_full_window_patient_bootstrap.csv", index=False)
    pd.concat(predictions["patient_centered"], ignore_index=True).to_parquet(out_dir / "raw_eeg_full_window_oof_predictions_patient_centered.parquet", index=False)
    return {"folds": N_SPLITS, "summary": summary_rows, "bootstrap": bootstrap_rows}


def _patient_folds(patients: Iterable[str]) -> dict[str, int]:
    roster = pd.DataFrame({"patient_id": sorted({str(x) for x in patients})})
    assignment = np.full(len(roster), -1, dtype=int)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (_train, test) in enumerate(splitter.split(roster, groups=roster["patient_id"])):
        assignment[test] = fold
    if np.any(assignment < 0):
        raise RuntimeError("TUSZ patient GroupKFold assignment incomplete")
    return dict(zip(roster.patient_id.astype(str), assignment.tolist()))


def classify_tusz(table: pd.DataFrame, out_dir: Path, n_bootstrap: int) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    saved_centered: list[pd.DataFrame] = []
    checkpoint_keys = sorted(table[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    for unit_index, (external_fold, external_seed) in enumerate(checkpoint_keys, start=1):
        unit = table[(table.fold.astype(int) == int(external_fold)) & (table.seed.astype(int) == int(external_seed))].copy()
        lookup = _patient_folds(unit.patient_id.astype(str).unique())
        unit["cv_fold"] = unit.patient_id.astype(str).map(lookup).astype(int)
        for centering in ("raw", "patient_centered"):
            centered = centering == "patient_centered"
            oof_parts: list[pd.DataFrame] = []
            for cv_fold in range(N_SPLITS):
                train = unit[unit.cv_fold != cv_fold].copy()
                test = unit[unit.cv_fold == cv_fold].copy()
                if train.empty or test.empty:
                    continue
                pred, info = _fit_predict(train, test, "patient_id", centered, BASE_SEED + int(external_fold) * 10000 + int(external_seed) + cv_fold)
                metric = _metrics(pred)
                fold_rows.append({
                    "external_fold": int(external_fold),
                    "external_seed": int(external_seed),
                    "checkpoint_id": f"fold{int(external_fold)}_seed{int(external_seed)}",
                    "cv_fold": cv_fold,
                    "centering": centering,
                    "status": "ok",
                    "chance_macro_auroc": 0.5,
                    "test_used_for_selection": False,
                    **metric,
                    **info,
                })
                pred["external_fold"] = int(external_fold)
                pred["external_seed"] = int(external_seed)
                pred["cv_fold"] = cv_fold
                pred["centering"] = centering
                oof_parts.append(pred)
            if not oof_parts:
                continue
            oof = pd.concat(oof_parts, ignore_index=True)
            metric = _metrics(oof)
            boot = _bootstrap_patient_auc(oof, n_bootstrap, np.random.default_rng(BASE_SEED + int(external_fold) * 10000 + int(external_seed) + (1 if centered else 0)))
            summary_rows.append({
                "external_fold": int(external_fold),
                "external_seed": int(external_seed),
                "checkpoint_id": f"fold{int(external_fold)}_seed{int(external_seed)}",
                "centering": centering,
                "status": "ok",
                "chance_macro_auroc": 0.5,
                "test_used_for_selection": False,
                **metric,
            })
            bootstrap_rows.append({
                "external_fold": int(external_fold),
                "external_seed": int(external_seed),
                "checkpoint_id": f"fold{int(external_fold)}_seed{int(external_seed)}",
                "centering": centering,
                "metric": "macro_auroc",
                "point_macro_auroc": metric["macro_auroc"],
                "mean": boot[0],
                "q025": boot[1],
                "q975": boot[2],
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_valid": boot[3],
                "bootstrap_unit": "patient",
            })
            if centered:
                saved_centered.append(oof)
        _write_progress(out_dir, {"dataset": "tusz", "status": "classifying", "checkpoint_completed": unit_index, "checkpoint_total": len(checkpoint_keys), "checkpoint": f"fold{int(external_fold)}_seed{int(external_seed)}"})
    summary_frame = pd.DataFrame(summary_rows)
    if summary_frame.empty:
        raise RuntimeError("TUSZ raw EEG classifier produced no summary rows")
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(out_dir / "raw_eeg_full_window_cv_fold_metrics.csv", index=False)
    summary_frame.to_csv(out_dir / "raw_eeg_full_window_summary_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out_dir / "raw_eeg_full_window_patient_bootstrap.csv", index=False)
    if saved_centered:
        pd.concat(saved_centered, ignore_index=True).to_parquet(out_dir / "raw_eeg_full_window_oof_predictions_patient_centered.parquet", index=False)
    model_summary = (
        summary_frame.groupby("centering", as_index=False)
        .agg(units=("macro_auroc", "count"), mean_macro_auroc=("macro_auroc", "mean"), median_macro_auroc=("macro_auroc", "median"), sd_macro_auroc=("macro_auroc", "std"), min_macro_auroc=("macro_auroc", "min"), max_macro_auroc=("macro_auroc", "max"), mean_accuracy=("accuracy", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"))
    )
    model_summary["chance_macro_auroc"] = 0.5
    model_summary["fraction_units_above_chance"] = [float(np.mean(summary_frame[summary_frame.centering == row.centering].macro_auroc > 0.5)) for row in model_summary.itertuples(index=False)]
    model_summary.to_csv(out_dir / "raw_eeg_full_window_model_summary.csv", index=False)
    return {"checkpoint_units": len(checkpoint_keys), "summary": summary_rows, "bootstrap": bootstrap_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("chbmit", "tusz"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-recordings", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    default_name = f"h3-raw-eeg-full-window-{args.dataset}-v1"
    out_dir = args.output_dir or (ROOT / "outputs/reports" / default_name)
    out_dir = Path(out_dir)
    formal = not args.smoke and args.limit_recordings <= 0
    if formal and any((out_dir / name).exists() for name in ("raw_eeg_full_window_features.parquet", "raw_eeg_full_window_feature_manifest.json", "raw_eeg_full_window_summary_metrics.csv")):
        raise SystemExit(f"refusing to overwrite existing formal raw EEG output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    table, feature_manifest = extract_features(args.dataset, out_dir, args.limit_recordings, max(1, args.workers))
    if args.dataset == "chbmit":
        analysis = classify_chb(table, out_dir, args.bootstrap)
    else:
        analysis = classify_tusz(table, out_dir, args.bootstrap)
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": args.dataset,
        "status": "complete" if formal else "smoke",
        "formal": formal,
        "source_run": SOURCE_RUN,
        "feature_set": "raw_eeg_full_window10",
        "feature_manifest": str((out_dir / "raw_eeg_full_window_feature_manifest.json").relative_to(ROOT)) if (out_dir / "raw_eeg_full_window_feature_manifest.json").is_relative_to(ROOT) else str(out_dir / "raw_eeg_full_window_feature_manifest.json"),
        "feature_manifest_sha256": sha256(out_dir / "raw_eeg_full_window_feature_manifest.json"),
        "rows": int(len(table)),
        "patients": int(table.patient_id.nunique()),
        "recordings": int(table.recording.nunique()),
        "models": sorted(table.model.astype(str).unique().tolist()),
        "feature_names": list(FEATURES),
        "model_derived_features": False,
        "event_duration_used_as_classifier_feature": False,
        "peak_probability_used_as_classifier_feature": False,
        "raw_input_preprocessing": "common canonical 16-channel bipolar raw EEG, 200-Hz resampling, per-frame de-mean; no model-specific filtering or model outputs",
        "classifier": "multinomial_logistic_C1_class_weight_balanced",
        "chance_macro_auroc": 0.5,
        "chance_accuracy": 1.0 / len(MODELS),
        "patient_centering": "label-free within-partition patient mean sensitivity analysis",
        "bootstrap": {"unit": "patient", "replicates": int(args.bootstrap)},
        "test_used_for_selection": False,
        "analysis": analysis,
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_progress(out_dir, {"dataset": args.dataset, "status": manifest["status"], "rows": len(table), "completed": True, "elapsed_seconds": manifest["elapsed_seconds"]})
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
