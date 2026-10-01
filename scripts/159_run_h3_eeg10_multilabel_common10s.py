"""Matched EEG10 multi-label baseline for the raw-waveform H3 experiment.

This run is deliberately matched to ``158_run_h3_raw_eeg_cnn_multilabel_one_round.py``:

* the same common 10-s midpoint-bin windows;
* the same three independent detector-source labels (multi-hot targets);
* the same frozen patient assignment and outer fold 0;
* the same validation-only per-label thresholds;
* the same four primary metrics and patient bootstrap.

Only the representation changes.  Each 10-s, 128-Hz, 16-channel EEG window is
converted into ten EEG-only statistics using five non-overlapping 2-s frames:
two amplitude, two spatial, five relative-band-power, and one high/low-ratio
feature.  No detector probability, event duration, QC, HMM, or other
model-derived input is used.  Three independent balanced logistic classifiers
provide the multi-label analogue of sigmoid heads; test is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import only deterministic window/split/metric utilities.  The imported file
# has no training side effect at import time.
_BASE_PATH = Path(__file__).resolve().parent / "158_run_h3_raw_eeg_cnn_multilabel_one_round.py"
_SPEC = importlib.util.spec_from_file_location("bfa_multilabel_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import matched baseline utilities: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_RAW = _BASE._BASE


SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL_COLUMNS = tuple(f"label_{name}" for name in MODELS)
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
TARGET_HZ = 128.0
WINDOW_S = 10.0
WINDOW_SAMPLES = 1280
FRAME_S = 2.0
FRAME_SAMPLES = 256
CHANNELS = 16
N_OUTER_SPLITS = 5
OUTER_FOLD = 0
BASE_SEED = 20260828
DEFAULT_BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def output_default(dataset: str) -> Path:
    name = "chb" if dataset == "chbmit" else dataset
    return ROOT / "outputs" / "reports" / f"h3-eeg10-multilabel-common10s-{name}-v1"


def raw_reference_dir(dataset: str) -> Path:
    name = "chb" if dataset == "chbmit" else dataset
    return ROOT / "outputs" / "reports" / f"h3-raw-eeg-cnn-128hz-10s-multilabel-{name}-v1"


def frame_feature_matrix(window_uv: np.ndarray) -> np.ndarray:
    """Compute the ten fixed EEG-only statistics for one 10-s window."""
    window_uv = np.asarray(window_uv, dtype=np.float32)
    if window_uv.shape != (CHANNELS, WINDOW_SAMPLES):
        raise ValueError(f"expected {(CHANNELS, WINDOW_SAMPLES)}, got {window_uv.shape}")
    # Five non-overlapping 2-s frames.  This is the same feature definition as
    # the prior EEG10 experiment, but the frame/sample rate is the matched
    # 128-Hz common input used by the raw CNN.
    frames = window_uv.reshape(CHANNELS, 5, FRAME_SAMPLES).transpose(1, 0, 2)
    frames = frames - frames.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=2) + 1e-8)
    rms_log = np.log1p(rms)
    frequencies = np.fft.rfftfreq(FRAME_SAMPLES, d=1.0 / TARGET_HZ)
    valid = (frequencies >= 0.5) & (frequencies <= 45.0)
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    spectrum = np.abs(np.fft.rfft(frames, axis=2)) ** 2
    total_power = spectrum[:, :, valid].sum(axis=2) + 1e-8
    band_means: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = valid & (frequencies >= low) & (frequencies < high)
        band_power = spectrum[:, :, mask].sum(axis=2)
        # Mean across the 16 canonical channels, then retain one value/frame.
        band_means[name] = np.mean(band_power / total_power, axis=1)
    frame_values = np.column_stack(
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
    # Aggregate across the five frames exactly as the earlier full-event
    # EEG10 feature definition: robust medians for amplitude/spatial values,
    # means for spectral values, then recompute the ratio.
    result = np.empty(len(FEATURES), dtype=np.float32)
    result[0:4] = np.median(frame_values[:, 0:4], axis=0)
    result[4:9] = np.mean(frame_values[:, 4:9], axis=0)
    result[9] = float((result[7] + result[8]) / (result[4] + result[5] + 1e-8))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_features(
    dataset: str,
    windows: pd.DataFrame,
    progress_path: Path,
) -> pd.DataFrame:
    """Read each cached recording once and compute matched EEG10 features."""
    rows: list[dict[str, Any]] = []
    groups = list(windows.groupby("relative_path", sort=True))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    for recording_index, (relative_path, group) in enumerate(groups, start=1):
        cache = _RAW.cache_path(dataset, str(relative_path))
        if not cache.exists():
            raise FileNotFoundError(cache)
        raw = np.asarray(np.load(cache, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        if raw.shape[0] != CHANNELS:
            raise ValueError(f"raw cache has wrong channel count: {cache}: {raw.shape}")
        for row in group.itertuples(index=False):
            # The base extractor applies the same canonical de-mean and fixed
            # divide-by-100 used by the raw CNN.  Restore microvolt units for
            # the interpretable RMS features; the classifier itself fits a
            # train-only StandardScaler below.
            signal = _RAW.extract_window(
                raw,
                float(row.window_start_s),
                float(row.window_end_s),
            ) * 100.0
            features = frame_feature_matrix(signal)
            record = {
                "window_index": int(row.window_index),
                "window_id": str(row.window_id),
                "patient_id": str(row.patient_id),
                "recording": str(row.recording),
                "relative_path": str(row.relative_path),
                "window_bin": int(row.window_bin),
                "window_start_s": float(row.window_start_s),
                "window_end_s": float(row.window_end_s),
                "window_center_s": float(row.window_center_s),
            }
            for label_column in LABEL_COLUMNS:
                record[label_column] = int(getattr(row, label_column))
            for feature, value in zip(FEATURES, features):
                record[feature] = float(value)
            rows.append(record)
        if recording_index % 25 == 0 or recording_index == len(groups):
            _RAW.write_progress(
                progress_path,
                {
                    "stage": "feature_extraction",
                    "dataset": dataset,
                    "recordings_completed": recording_index,
                    "recordings_total": len(groups),
                    "windows_completed": len(rows),
                    "status": "running",
                },
            )
            print(
                f"EEG10_FEATURE_PROGRESS dataset={dataset} recordings={recording_index}/{len(groups)} windows={len(rows)}",
                flush=True,
            )
    result = pd.DataFrame(rows).sort_values("window_index", kind="mergesort").reset_index(drop=True)
    if len(result) != len(windows):
        raise RuntimeError(f"feature row mismatch: {len(result)} != {len(windows)}")
    return result


def load_frozen_split(dataset: str, windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    reference = raw_reference_dir(dataset) / "raw_eeg_cnn_multilabel_patient_split.csv"
    if not reference.exists():
        raise FileNotFoundError(reference)
    assignment = pd.read_csv(reference, dtype={"patient_id": str})
    required = {"patient_id", "outer_fold", "partition"}
    if not required.issubset(assignment.columns):
        raise RuntimeError(f"frozen split missing columns: {sorted(required - set(assignment.columns))}")
    assignment = assignment[["patient_id", "outer_fold", "partition"]].copy()
    assignment["patient_id"] = assignment["patient_id"].astype(str)
    if assignment["patient_id"].duplicated().any():
        raise RuntimeError("frozen split contains duplicate patients")
    expected = set(windows["patient_id"].astype(str).unique())
    observed = set(assignment["patient_id"])
    if expected != observed:
        raise RuntimeError(f"frozen split roster mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    if not bool((assignment["outer_fold"].astype(int) == OUTER_FOLD).any()):
        raise RuntimeError("frozen split does not contain evaluated outer fold")
    lookup = assignment.set_index("patient_id")["partition"].to_dict()
    partitions = windows["patient_id"].astype(str).map(lookup)
    if partitions.isna().any():
        raise RuntimeError("window patient missing frozen partition")
    train = np.flatnonzero(partitions.to_numpy() == "train").astype(np.int64)
    validation = np.flatnonzero(partitions.to_numpy() == "validation").astype(np.int64)
    test = np.flatnonzero(partitions.to_numpy() == "test").astype(np.int64)
    if not len(train) or not len(validation) or not len(test):
        raise RuntimeError("frozen split has an empty partition")
    return train, validation, test, assignment


def fit_preprocess(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_values = train.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    target_values = target.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = train_values.median().fillna(0.0)
    train_values = train_values.fillna(medians)
    target_values = target_values.fillna(medians)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    target_values = target_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    info = {
        "imputation": "train-feature medians",
        "clipping": "train 1%-99% quantiles",
        "scaler": "train-fitted StandardScaler",
        "medians": {key: float(value) for key, value in medians.items()},
        "lower": {key: float(value) for key, value in lower.items()},
        "upper": {key: float(value) for key, value in upper.items()},
        "scaler_mean": {key: float(value) for key, value in zip(FEATURES, scaler.mean_)},
        "scaler_scale": {key: float(value) for key, value in zip(FEATURES, scaler.scale_)},
    }
    return scaler.transform(train_values), scaler.transform(target_values), {"scaler": scaler, **info}


def fit_predict(
    train: pd.DataFrame,
    target: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    train_values, target_values, preprocess = fit_preprocess(train, target)
    y_train = train.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
    probabilities = np.empty((len(target), len(MODELS)), dtype=float)
    classifiers: list[LogisticRegression] = []
    for index in range(len(MODELS)):
        y = y_train[:, index]
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"training label {MODELS[index]} lacks both classes")
        classifier = LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",
            random_state=int(seed + index),
        )
        classifier.fit(train_values, y)
        probabilities[:, index] = classifier.predict_proba(target_values)[:, 1]
        classifiers.append(classifier)
    artifact = {
        "classifiers": classifiers,
        "preprocess": preprocess,
        "features": list(FEATURES),
        "models": list(MODELS),
        "classifier": "three independent balanced LogisticRegression(C=1) heads",
    }
    return probabilities, artifact, preprocess


def patient_bootstrap(
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
    n_bootstrap: int,
    seed: int,
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
        y = predictions.iloc[indices][list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
        probabilities = predictions.iloc[indices][[f"prob_{name}" for name in MODELS]].to_numpy(dtype=float)
        metrics = _BASE.multilabel_metrics(y, probabilities, thresholds)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"):
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
            "eeg10_multilabel_fold_metrics.csv",
            "eeg10_multilabel_outer_test_predictions.parquet",
        )
    ):
        raise SystemExit(f"refusing to overwrite formal EEG10 output: {out_dir}")
    started = time.monotonic()
    reference_dir = raw_reference_dir(dataset)
    windows_path = reference_dir / "raw_eeg_multilabel_common_windows.parquet"
    input_manifest_path = reference_dir / "raw_eeg_multilabel_input_manifest.json"
    if not windows_path.exists() or not input_manifest_path.exists():
        raise FileNotFoundError(f"matched raw-window input missing: {windows_path}")
    windows = pd.read_parquet(windows_path)
    original_windows = len(windows)
    if limit_windows > 0:
        windows = _BASE.select_smoke_windows(windows, limit_windows)
    if not len(windows):
        raise RuntimeError("no windows selected")
    source_input_manifest = json.loads(input_manifest_path.read_text())
    features = extract_features(dataset, windows, out_dir / "progress.json")
    feature_path = out_dir / "eeg10_multilabel_common10s_features.parquet"
    features.to_parquet(feature_path, index=False)
    feature_manifest = {
        "dataset": dataset,
        "feature_set": "matched_eeg_only10_common10s",
        "feature_names": list(FEATURES),
        "window_definition": "same common 10-s midpoint-bin windows as raw CNN",
        "window_seconds": WINDOW_S,
        "sampling_hz": TARGET_HZ,
        "channels": CHANNELS,
        "frame_seconds": FRAME_S,
        "frame_stride_seconds": FRAME_S,
        "feature_semantics": {
            "rms_median_uv": "median channel RMS per 2-s frame, then median over five frames",
            "rms_iqr_uv": "channel RMS IQR per 2-s frame, then median over five frames",
            "spatial_rms_dispersion": "SD of log1p channel RMS per frame, then median over five frames",
            "spatial_rms_max_min_log": "log-RMS channel max-min per frame, then median over five frames",
            "relative_band_power": "mean channel 0.5-45-Hz band/total power ratio, averaged over five frames",
            "high_low_ratio": "(beta+gamma)/(delta+theta) after five-frame aggregation",
        },
        "classifier_input_columns": list(FEATURES),
        "model_derived_features": False,
        "detector_probabilities_as_input": False,
        "event_duration_as_input": False,
        "qc_features_as_input": False,
        "hmm_features_as_input": False,
        "source_common_windows": str(windows_path.relative_to(ROOT)),
        "source_common_windows_sha256": sha256(windows_path),
        "source_raw_input_manifest": str(input_manifest_path.relative_to(ROOT)),
        "source_raw_input_manifest_sha256": sha256(input_manifest_path),
        "windows_used": int(len(windows)),
        "windows_available": int(original_windows),
        "output_feature_sha256": sha256(feature_path),
    }
    (out_dir / "eeg10_multilabel_feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n"
    )

    if formal:
        train_indices, validation_indices, test_indices, assignment = load_frozen_split(dataset, windows)
        split_source = raw_reference_dir(dataset) / "raw_eeg_cnn_multilabel_patient_split.csv"
    else:
        train_indices, validation_indices, test_indices, assignment = _BASE.split_outer_and_validation(
            windows,
            BASE_SEED,
        )
        split_source = None
    assignment.to_csv(out_dir / "eeg10_multilabel_patient_split.csv", index=False)
    train = features.iloc[train_indices].copy()
    validation = features.iloc[validation_indices].copy()
    test = features.iloc[test_indices].copy()
    validation_probabilities, _validation_artifact, _ = fit_predict(train, validation, BASE_SEED)
    validation_y = validation.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
    thresholds = _BASE.choose_validation_thresholds(validation_y, validation_probabilities)
    test_probabilities, artifact, _ = fit_predict(train, test, BASE_SEED)
    test_y = test.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
    test_metrics = _BASE.multilabel_metrics(test_y, test_probabilities, thresholds)
    validation_metrics = _BASE.multilabel_metrics(validation_y, validation_probabilities, thresholds)
    predictions = test[
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
    ].copy()
    for index, name in enumerate(MODELS):
        predictions[f"prob_{name}"] = test_probabilities[:, index]
        predictions[f"pred_{name}"] = (test_probabilities[:, index] >= thresholds[index]).astype(np.int64)
    predictions.to_parquet(out_dir / "eeg10_multilabel_outer_test_predictions.parquet", index=False)
    bootstrap_frame = patient_bootstrap(
        predictions,
        thresholds,
        10 if smoke else bootstrap,
        BASE_SEED + 9000,
    )
    bootstrap_frame.to_csv(out_dir / "eeg10_multilabel_patient_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap_frame.groupby("metric")["value"]
        .agg(
            mean="mean",
            q025=lambda values: float(np.nanquantile(values, 0.025)),
            q975=lambda values: float(np.nanquantile(values, 0.975)),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(out_dir / "eeg10_multilabel_patient_bootstrap_summary.csv", index=False)
    joblib.dump(
        {
            "classifiers": artifact["classifiers"],
            "scaler": artifact["preprocess"]["scaler"],
            "features": list(FEATURES),
            "models": list(MODELS),
            "thresholds_from_validation_only": thresholds.tolist(),
            "outer_fold": OUTER_FOLD,
            "test_used_for_selection": False,
        },
        out_dir / "eeg10_multilabel_outer_fold0.joblib",
    )
    joblib.dump(
        {
            "classifiers": artifact["classifiers"],
            "scaler": artifact["preprocess"]["scaler"],
            "features": list(FEATURES),
            "models": list(MODELS),
            "thresholds_from_validation_only": thresholds.tolist(),
            "outer_fold": OUTER_FOLD,
            "test_used_for_selection": False,
        },
        out_dir / "eeg10_multilabel_best_validation.joblib",
    )
    checkpoint_hash = sha256(out_dir / "eeg10_multilabel_outer_fold0.joblib")
    fold_metrics = {
        "dataset": dataset,
        "outer_fold": OUTER_FOLD,
        "outer_folds_total": N_OUTER_SPLITS,
        "rounds_completed": 1,
        "status": "complete" if formal else "smoke",
        "train_windows": int(len(train)),
        "validation_windows": int(len(validation)),
        "test_windows": int(len(test)),
        "train_patients": int(train.patient_id.nunique()),
        "validation_patients": int(validation.patient_id.nunique()),
        "test_patients": int(test.patient_id.nunique()),
        "validation_thresholds": thresholds.tolist(),
        "threshold_source": "validation-only",
        "test_evaluation_count": 1,
        "test_used_for_selection": False,
        "classifier": "three independent balanced LogisticRegression(C=1) heads",
        "checkpoint": "eeg10_multilabel_outer_fold0.joblib",
        "checkpoint_sha256": checkpoint_hash,
    }
    for prefix, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"):
            fold_metrics[f"{prefix}_{metric_name}"] = metrics.get(metric_name, float("nan"))
    pd.DataFrame([fold_metrics]).to_csv(out_dir / "eeg10_multilabel_fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "evaluation": "outer_fold0_test_once",
                "accuracy": test_metrics.get("accuracy", float("nan")),
                "balanced_accuracy": test_metrics.get("balanced_accuracy", float("nan")),
                "macro_f1": test_metrics.get("macro_f1", float("nan")),
                "macro_auroc": test_metrics.get("macro_auroc", float("nan")),
                "threshold_source": "validation-only",
                "test_used_for_selection": False,
                "test_evaluation_count": 1,
            }
        ]
    ).to_csv(out_dir / "eeg10_multilabel_summary_metrics.csv", index=False)
    manifest = {
        "analysis_id": out_dir.name,
        "dataset": dataset,
        "status": "complete" if formal else "smoke",
        "formal": formal,
        "source_run": SOURCE_RUN,
        "matched_raw_analysis": raw_reference_dir(dataset).name,
        "matched_common_windows": str(windows_path.relative_to(ROOT)),
        "matched_common_windows_sha256": sha256(windows_path),
        "input": feature_manifest,
        "feature_table": feature_path.name,
        "feature_table_sha256": sha256(feature_path),
        "feature_set": "matched_eeg_only10_common10s",
        "feature_names": list(FEATURES),
        "architecture": "three independent balanced LogisticRegression(C=1) heads",
        "output_activation": "independent logistic probabilities",
        "loss": "per-label logistic loss through independent one-vs-rest heads",
        "patient_split": {
            "outer": "same frozen ordinary GroupKFold(n_splits=5), canonical patient_id groups",
            "outer_fold_evaluated": OUTER_FOLD,
            "outer_folds_total": N_OUTER_SPLITS,
            "inner_validation": "same frozen patient-level validation partition as raw CNN fold 0",
            "split_source": str(split_source.relative_to(ROOT)) if split_source else "smoke GroupShuffleSplit",
            "split_source_sha256": sha256(split_source) if split_source else None,
            "rounds_completed": 1,
        },
        "thresholds": {
            "source": "validation-only",
            "method": "same per-detector validation F1 maximization as raw CNN",
            "values": thresholds.tolist(),
        },
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "bootstrap": {"unit": "patient", "replicates": int(10 if smoke else bootstrap)},
        "overfit_guard": {
            "selection_partition": "validation patients only",
            "test_touched_during_feature_preprocessing": False,
            "test_touched_during_threshold_selection": False,
            "test_evaluation_count_per_round": 1,
        },
        "fold_metrics": fold_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "output_files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    _RAW.write_progress(
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
    print(json.dumps({"dataset": dataset, **fold_metrics}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("chbmit", "tusz"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset
    out_dir = args.output_dir or output_default(dataset)
    run_dataset(
        dataset,
        Path(out_dir),
        bootstrap=max(1, int(args.bootstrap)),
        limit_windows=max(0, int(args.limit_windows)),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
