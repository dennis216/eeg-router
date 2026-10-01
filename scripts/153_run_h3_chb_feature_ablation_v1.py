"""CHB-MIT H3 feature-ablation matrix.

This script supplements the frozen Full15 CHB GroupKFold H3 result.  It
fits the same detector-source classifier on the three missing feature sets:
EEG/QC-only-13, duration+peak-2, and strict EEG-only-10.  The existing
Full15 result is deliberately not rerun.  All rows come from the frozen
v3-groupkfold-confirmatory-v1 feature table and use its patient-level
outer-fold assignments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUN_ID = "h3-chb-feature-ablations-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
FEATURE_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "h2h3_fp_features.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: index for index, name in enumerate(MODELS)}
FEATURE_SETS = {
    "eeg_qc_only13": (
        "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
        "spatial_rms_max_min_log", "line_noise_ratio", "clipping_fraction",
        "flat_fraction", "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power", "relative_gamma_power",
        "high_low_ratio",
    ),
    "duration_peak2": ("event_duration_s", "peak_probability"),
    "strict_eeg_only10": (
        "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
        "spatial_rms_max_min_log", "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power", "relative_gamma_power",
        "high_low_ratio",
    ),
}
N_BOOTSTRAP = 2000
RNG_SEED = 20260828


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def center(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    cols = list(columns)
    out.loc[:, cols] = out.loc[:, cols] - out.groupby("patient")[cols].transform("mean")
    return out


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, columns: tuple[str, ...], centered: bool, fold: int) -> tuple[dict[str, object], pd.DataFrame]:
    train_frame = center(train, columns) if centered else train.copy()
    target_frame = center(target, columns) if centered else target.copy()
    train_values = train_frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    target_values = target_frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    target_values = target_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    target_values = target_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    y_train = train_frame.model.map(LABELS).astype(int).to_numpy()
    y_target = target_frame.model.map(LABELS).astype(int).to_numpy()
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=RNG_SEED + fold)
    if len(np.unique(y_train)) != len(MODELS):
        raise RuntimeError(f"fold {fold} training lacks a detector source class")
    clf.fit(scaler.transform(train_values), y_train)
    probabilities = clf.predict_proba(scaler.transform(target_values))
    prediction = np.argmax(probabilities, axis=1)
    metrics = {
        "fold": fold,
        "patient_centered": centered,
        "train_events": int(len(train_frame)),
        "validation_events": int(len(target_frame)),
        "accuracy": float(accuracy_score(y_target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_target, prediction)),
        "macro_f1": float(f1_score(y_target, prediction, labels=list(range(len(MODELS))), average="macro", zero_division=0)),
        "chance_accuracy": 1.0 / len(MODELS),
        "chance_macro_auroc": 0.5,
    }
    try:
        metrics["macro_auroc"] = float(roc_auc_score(y_target, probabilities, labels=list(range(len(MODELS))), multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_auroc"] = math.nan
    pred = target_frame[["patient", "recording", "model", "start_s", "end_s"]].copy()
    pred["true_model"] = pred.model.astype(str)
    pred["prediction"] = [MODELS[i] for i in prediction]
    for i, model in enumerate(MODELS):
        pred[f"probability_{model}"] = probabilities[:, i]
    return metrics, pred.drop(columns=["model"])


def patient_bootstrap(pred: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float, int]:
    rows = []
    for patient, group in pred.groupby("patient", sort=True):
        y = group.true_model.map(LABELS).astype(int).to_numpy()
        probs = group[[f"probability_{m}" for m in MODELS]].to_numpy(float)
        if len(np.unique(y)) != len(MODELS):
            continue
        rows.append(float(roc_auc_score(y, probs, labels=list(range(len(MODELS))), multi_class="ovr", average="macro")))
    values = np.asarray(rows, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan, math.nan, 0
    sampled = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        sampled[i] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return float(np.mean(values)), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), int(len(values))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    features = pd.read_parquet(FEATURE_PATH)
    required = {"partition", "split", "patient", "recording", "model", "start_s", "end_s"} | set(sum((list(v) for v in FEATURE_SETS.values()), []))
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"missing source columns: {missing}")
    features = features.copy()
    features["split"] = features.split.astype(int)
    features["partition"] = features.partition.astype(str)
    features["model"] = features.model.astype(str)
    for feature_set, columns in FEATURE_SETS.items():
        for centered in (False, True):
            all_test_parts: list[pd.DataFrame] = []
            for fold in range(5):
                fold_frame = features[features.split == fold]
                train = fold_frame[fold_frame.partition == "train"].copy()
                validation = fold_frame[fold_frame.partition == "validation"].copy()
                test = fold_frame[fold_frame.partition == "test"].copy()
                if train.empty or validation.empty or test.empty:
                    raise RuntimeError(f"empty fold partition {fold}")
                val_metrics, _ = fit_predict(train, validation, columns, centered, fold)
                test_metrics, test_pred = fit_predict(train, test, columns, centered, fold)
                row = {"feature_set": feature_set, "feature_count": len(columns), "test_evaluation_role": "frozen_outer_test_once", "test_used_for_selection": False, **test_metrics}
                row.update({f"validation_{key}": value for key, value in val_metrics.items() if key in {"accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"}})
                metrics_rows.append(row)
                test_pred["feature_set"] = feature_set
                test_pred["patient_centered"] = centered
                test_pred["fold"] = fold
                all_test_parts.append(test_pred)
            combined = pd.concat(all_test_parts, ignore_index=True)
            prediction_parts.append(combined)
            mean, lo, hi, count = patient_bootstrap(combined, np.random.default_rng(RNG_SEED + hash(feature_set) % 10000 + int(centered)))
            bootstrap_rows.append({"feature_set": feature_set, "patient_centered": centered, "metric": "macro_auroc", "mean": mean, "q025": lo, "q975": hi, "patient_count": count, "bootstrap_replicates": N_BOOTSTRAP, "bootstrap_unit": "patient"})
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics.to_csv(REPORT_DIR / "h3_chb_feature_ablation_metrics.csv", index=False)
    predictions.to_parquet(REPORT_DIR / "h3_chb_feature_ablation_test_predictions.parquet", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "h3_chb_feature_ablation_patient_bootstrap.csv", index=False)
    summary = metrics.groupby(["feature_set", "patient_centered"], as_index=False).agg(
        folds=("fold", "count"), mean_test_accuracy=("accuracy", "mean"), mean_test_balanced_accuracy=("balanced_accuracy", "mean"), mean_test_macro_f1=("macro_f1", "mean"), mean_test_macro_auroc=("macro_auroc", "mean"),
    )
    summary.to_csv(REPORT_DIR / "h3_chb_feature_ablation_summary.csv", index=False)
    manifest = {
        "analysis_id": RUN_ID,
        "status": "complete",
        "source_run": SOURCE_RUN,
        "source_feature_table": str(FEATURE_PATH.relative_to(ROOT)),
        "source_feature_table_sha256": sha256(FEATURE_PATH),
        "models": list(MODELS),
        "patients": int(features.patient.nunique()),
        "source_rows": int(len(features)),
        "outer_grouping": "frozen CHB canonical patient GroupKFold assignments in v3-groupkfold-confirmatory-v1",
        "feature_sets": {name: list(cols) for name, cols in FEATURE_SETS.items()},
        "centering": "raw and label-free within-partition patient-centered sensitivity",
        "classifier": "multinomial logistic regression C=1 class_weight=balanced max_iter=500",
        "threshold_selection": "none; detector-source labels are fixed",
        "test_used_for_selection": False,
        "chance_accuracy": 1.0 / len(MODELS),
        "chance_macro_auroc": 0.5,
        "bootstrap": {"unit": "patient", "replicates": N_BOOTSTRAP},
        "outputs": ["h3_chb_feature_ablation_metrics.csv", "h3_chb_feature_ablation_summary.csv", "h3_chb_feature_ablation_patient_bootstrap.csv", "h3_chb_feature_ablation_test_predictions.parquet"],
        "claim_boundary": "feature ablation of frozen H3; not a new seizure-detector training experiment",
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
