"""Independent external replication of the detector-source H3 analysis on TUSZ.

The script reuses the frozen TUSZ false-positive events and the feature table
created by ``146_extract_tusz_h3_features.py``.  It fits the same fixed
multinomial logistic classifier inside a *new patient-level GroupKFold* on
TUSZ, with no threshold selection and no feature selection on TUSZ.  A second
patient-centering sensitivity analysis and two pre-specified feature ablations
separate detector output information from EEG/QC information.

This is an external construct replication, not a claim that a classifier
trained on CHB transfers without adaptation.  The frozen CHB H3 classifier
specification (C=1, balanced multinomial logistic regression) is applied to
TUSZ with a TUSZ patient-group cross-validation fit, while the event labels
remain detector source labels by design.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUN_ID = os.environ.get("BFA_RUN_ID", "h3-external-tusz-v1")
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
if os.environ.get("BFA_OUTPUT_DIR"):
    REPORT_DIR = Path(os.environ["BFA_OUTPUT_DIR"])
FEATURE_PATH = Path(os.environ.get("BFA_FEATURE_PATH", str(REPORT_DIR / "tusz_h3_fp_features.parquet")))
FEATURE_MANIFEST_PATH = Path(os.environ.get("BFA_FEATURE_MANIFEST_PATH", str(REPORT_DIR / "tusz_h3_fp_features_manifest.json")))
SOURCE_EVENT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
SOURCE_SCORING_MANIFEST = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_scoring_manifest.json"
OUT_MANIFEST = REPORT_DIR / "h3_external_tusz_manifest.json"
FOLD_ASSIGNMENTS_PATH = REPORT_DIR / "tusz_patient_groupkfold_assignments.csv"
FOLD_METRICS_PATH = REPORT_DIR / "h3_external_tusz_cv_fold_metrics.csv"
SUMMARY_PATH = REPORT_DIR / "h3_external_tusz_summary_metrics.csv"
BOOTSTRAP_PATH = REPORT_DIR / "h3_external_tusz_patient_bootstrap.csv"
PERMUTATION_PATH = REPORT_DIR / "h3_external_tusz_patient_permutation.csv"
MODEL_SUMMARY_PATH = REPORT_DIR / "h3_external_tusz_model_summary.csv"
OOF_PATH = REPORT_DIR / "h3_external_tusz_oof_predictions_full15_patient_centered.parquet"

MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: index for index, name in enumerate(MODELS)}
FEATURES = (
    "event_duration_s",
    "peak_probability",
    "rms_median_uv",
    "rms_iqr_uv",
    "spatial_rms_dispersion",
    "spatial_rms_max_min_log",
    "line_noise_ratio",
    "clipping_fraction",
    "flat_fraction",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "high_low_ratio",
)
FEATURE_SETS = {
    "full15": FEATURES,
    "eeg_only13": tuple(x for x in FEATURES if x not in {"event_duration_s", "peak_probability"}),
    "duration_peak2": ("event_duration_s", "peak_probability"),
    "strict_eeg_only10": (
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
    ),
}
_requested_feature_sets = os.environ.get("BFA_FEATURE_SETS", "").strip()
if _requested_feature_sets:
    _requested = tuple(x.strip() for x in _requested_feature_sets.split(",") if x.strip())
    missing_feature_sets = sorted(set(_requested) - set(FEATURE_SETS))
    if missing_feature_sets:
        raise ValueError(f"unknown BFA_FEATURE_SETS: {missing_feature_sets}")
    FEATURE_SETS = {name: FEATURE_SETS[name] for name in _requested}
CENTERING = ("raw", "patient_centered")
N_SPLITS = 5
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 1000
BASE_SEED = 20260828


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    columns = list(columns)
    result.loc[:, columns] = result.loc[:, columns] - result.groupby("patient_id")[columns].transform("mean")
    return result


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: tuple[str, ...],
    centered: bool,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_frame = _center(train, feature_names) if centered else train.copy()
    test_frame = _center(test, feature_names) if centered else test.copy()
    train_values = train_frame.loc[:, feature_names].replace([np.inf, -np.inf], np.nan)
    test_values = test_frame.loc[:, feature_names].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    test_values = test_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    test_values = test_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    y_train = train_frame["model"].map(LABELS).to_numpy(dtype=int)
    y_test = test_frame["model"].map(LABELS).to_numpy(dtype=int)
    if len(np.unique(y_train)) != len(MODELS):
        raise ValueError("training partition does not contain all detector classes")
    clf = LogisticRegression(
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(scaler.transform(train_values), y_train)
    probabilities = clf.predict_proba(scaler.transform(test_values))
    predictions = clf.predict(scaler.transform(test_values))
    out = test_frame[["patient_id", "recording", "event_index", "model", "fold", "seed", "start_s", "end_s"]].copy()
    out["y"] = y_test
    for index, model in enumerate(MODELS):
        out[f"prob_{model}"] = probabilities[:, index]
    out["prediction"] = predictions
    return out, {
        "train_events": int(len(train_frame)),
        "test_events": int(len(test_frame)),
        "classifier": "multinomial_logistic_C1_class_weight_balanced",
        "fit_partition": "tusz_groupkfold_train",
        "threshold_selection": "none_classifier_source_labels",
    }


def _metrics(pred: pd.DataFrame) -> dict[str, float]:
    y = pred["y"].to_numpy(dtype=int)
    proba = pred[[f"prob_{model}" for model in MODELS]].to_numpy(dtype=float)
    result: dict[str, float] = {
        "events": float(len(pred)),
        "patients": float(pred["patient_id"].nunique()),
        "accuracy": float(accuracy_score(y, pred["prediction"].to_numpy(dtype=int))),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred["prediction"].to_numpy(dtype=int))),
    }
    try:
        result["macro_auroc"] = float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(len(MODELS))))
        )
        for index, model in enumerate(MODELS):
            result[f"{model}_auroc"] = float(roc_auc_score((y == index).astype(int), proba[:, index]))
    except ValueError:
        result["macro_auroc"] = float("nan")
        for model in MODELS:
            result[f"{model}_auroc"] = float("nan")
    return result


def _bootstrap_patient_auc(pred: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    patients = np.asarray(sorted(pred["patient_id"].unique()), dtype=object)
    by_patient = {patient: np.flatnonzero(pred["patient_id"].to_numpy() == patient) for patient in patients}
    y = pred["y"].to_numpy(dtype=int)
    proba = pred[[f"prob_{model}" for model in MODELS]].to_numpy(dtype=float)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([by_patient[patient] for patient in selected])
        try:
            values.append(float(roc_auc_score(y[indices], proba[indices], multi_class="ovr", average="macro", labels=list(range(len(MODELS))))))
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), int(len(array))


def _permute_patient_labels(pred: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> tuple[float, float, int]:
    """Conditional held-out test-label permutation within patient blocks.

    The classifier is fitted only once per held-out fold.  The permutation test
    therefore asks whether its fixed held-out scores align with detector-source
    labels beyond the patient block, without pretending to be a fresh training
    null distribution.
    """
    patients = np.asarray(sorted(pred["patient_id"].unique()), dtype=object)
    patient_positions = [np.flatnonzero(pred["patient_id"].to_numpy() == patient) for patient in patients]
    y = pred["y"].to_numpy(dtype=int)
    proba = pred[[f"prob_{model}" for model in MODELS]].to_numpy(dtype=float)
    observed = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(len(MODELS)))))
    null: list[float] = []
    for _ in range(n_permutations):
        permuted = y.copy()
        for positions in patient_positions:
            permuted[positions] = rng.permutation(permuted[positions])
        try:
            null.append(float(roc_auc_score(permuted, proba, multi_class="ovr", average="macro", labels=list(range(len(MODELS))))))
        except ValueError:
            continue
    if not null:
        return observed, float("nan"), 0
    array = np.asarray(null, dtype=float)
    p_value = float((1 + np.sum(array >= observed)) / (len(array) + 1))
    return observed, p_value, int(len(array))


def _make_patient_folds(patients: list[str]) -> pd.DataFrame:
    roster = pd.DataFrame({"patient_id": sorted(patients)})
    group_kfold = GroupKFold(n_splits=N_SPLITS)
    assignment = np.full(len(roster), -1, dtype=int)
    for fold, (_train, test) in enumerate(group_kfold.split(roster, groups=roster["patient_id"])):
        assignment[test] = fold
    if np.any(assignment < 0):
        raise RuntimeError("patient GroupKFold assignment incomplete")
    roster["cv_fold"] = assignment
    return roster


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    formal = os.environ.get("BFA_OUTPUT_DIR") is None
    if formal and any(path.exists() for path in (FOLD_METRICS_PATH, SUMMARY_PATH, BOOTSTRAP_PATH, PERMUTATION_PATH, MODEL_SUMMARY_PATH, OUT_MANIFEST)):
        raise SystemExit(f"refusing to overwrite existing H3 external analysis in {REPORT_DIR}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    features = pd.read_parquet(FEATURE_PATH)
    missing = sorted(set(FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"external feature table missing columns: {missing}")
    if set(features["model"].unique()) != set(MODELS):
        raise ValueError(f"unexpected detector classes: {sorted(features.model.unique())}")

    patient_assignment = _make_patient_folds(sorted(features["patient_id"].unique()))
    patient_assignment.to_csv(FOLD_ASSIGNMENTS_PATH, index=False)
    fold_lookup = patient_assignment.set_index("patient_id")["cv_fold"].to_dict()
    features = features.copy()
    features["cv_fold"] = features["patient_id"].map(fold_lookup).astype(int)

    checkpoint_keys = sorted(features[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    limit_checkpoints = int(os.environ.get("BFA_LIMIT_CHECKPOINTS", "0"))
    if limit_checkpoints > 0:
        checkpoint_keys = checkpoint_keys[:limit_checkpoints]
    n_bootstrap = int(os.environ.get("BFA_N_BOOTSTRAP", str(DEFAULT_BOOTSTRAP)))
    n_permutations = int(os.environ.get("BFA_N_PERMUTATIONS", str(DEFAULT_PERMUTATIONS)))
    print(json.dumps({"run_id": RUN_ID, "patients": int(features.patient_id.nunique()), "events": int(len(features)), "checkpoints": len(checkpoint_keys), "n_bootstrap": n_bootstrap, "n_permutations": n_permutations}, sort_keys=True), flush=True)

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    saved_oof: list[pd.DataFrame] = []

    for checkpoint_index, (external_fold, external_seed) in enumerate(checkpoint_keys):
        unit = features[(features["fold"] == external_fold) & (features["seed"] == external_seed)].copy()
        unit_id = f"fold{int(external_fold)}_seed{int(external_seed)}"
        if unit["model"].nunique() != len(MODELS):
            print(f"H3_EXTERNAL_SKIP checkpoint={unit_id} reason=missing_detector_class", flush=True)
            continue
        for centering in CENTERING:
            centered = centering == "patient_centered"
            for feature_set, feature_names in FEATURE_SETS.items():
                oof_parts: list[pd.DataFrame] = []
                for cv_fold in range(N_SPLITS):
                    train = unit[unit["cv_fold"] != cv_fold].copy()
                    test = unit[unit["cv_fold"] == cv_fold].copy()
                    if train.empty or test.empty:
                        continue
                    try:
                        pred, fit_info = _fit_predict(
                            train,
                            test,
                            feature_names,
                            centered,
                            BASE_SEED + int(external_fold) * 10000 + int(external_seed) + cv_fold,
                        )
                    except ValueError as exc:
                        fold_rows.append({"external_fold": int(external_fold), "external_seed": int(external_seed), "checkpoint_id": unit_id, "centering": centering, "feature_set": feature_set, "cv_fold": int(cv_fold), "status": "skipped", "error": str(exc)})
                        continue
                    metric = _metrics(pred)
                    fold_rows.append({
                        "external_fold": int(external_fold),
                        "external_seed": int(external_seed),
                        "checkpoint_id": unit_id,
                        "centering": centering,
                        "feature_set": feature_set,
                        "cv_fold": int(cv_fold),
                        "status": "ok",
                        "test_used_for_selection": False,
                        "chance_macro_auroc": 0.5,
                        **fit_info,
                        **metric,
                    })
                    pred["external_fold"] = int(external_fold)
                    pred["external_seed"] = int(external_seed)
                    pred["checkpoint_id"] = unit_id
                    pred["centering"] = centering
                    pred["feature_set"] = feature_set
                    pred["cv_fold"] = int(cv_fold)
                    oof_parts.append(pred)
                if not oof_parts:
                    continue
                oof = pd.concat(oof_parts, ignore_index=True)
                metric = _metrics(oof)
                rng_seed = BASE_SEED + int(external_fold) * 100000 + int(external_seed) * 10 + (1 if centered else 0) + list(FEATURE_SETS).index(feature_set) * 1000000
                rng = np.random.default_rng(rng_seed)
                boot_mean, boot_lo, boot_hi, boot_valid = _bootstrap_patient_auc(oof, n_bootstrap, rng)
                observed, permutation_p, permutation_valid = _permute_patient_labels(oof, n_permutations, np.random.default_rng(rng_seed + 991))
                summary_rows.append({
                    "external_fold": int(external_fold),
                    "external_seed": int(external_seed),
                    "checkpoint_id": unit_id,
                    "centering": centering,
                    "feature_set": feature_set,
                    "status": "ok",
                    "test_evaluation_role": "held_out_inner_tusz_groupkfold_oof",
                    "test_used_for_selection": False,
                    "chance_macro_auroc": 0.5,
                    **metric,
                })
                bootstrap_rows.append({
                    "external_fold": int(external_fold),
                    "external_seed": int(external_seed),
                    "checkpoint_id": unit_id,
                    "centering": centering,
                    "feature_set": feature_set,
                    "point_macro_auroc": float(metric["macro_auroc"]),
                    "bootstrap_mean_macro_auroc": boot_mean,
                    "bootstrap_ci_low": boot_lo,
                    "bootstrap_ci_high": boot_hi,
                    "bootstrap_replicates": n_bootstrap,
                    "bootstrap_valid": boot_valid,
                    "bootstrap_unit": "patient",
                })
                permutation_rows.append({
                    "external_fold": int(external_fold),
                    "external_seed": int(external_seed),
                    "checkpoint_id": unit_id,
                    "centering": centering,
                    "feature_set": feature_set,
                    "observed_macro_auroc": observed,
                    "permutation_p_value": permutation_p,
                    "permutations": n_permutations,
                    "permutations_valid": permutation_valid,
                    "permutation_unit": "within_patient_test_labels_conditional_on_fixed_scores",
                })
                if feature_set == "full15" and centered:
                    saved_oof.append(oof)
        print(f"H3_EXTERNAL_DONE checkpoint={unit_id} index={checkpoint_index + 1}/{len(checkpoint_keys)}", flush=True)

    fold_frame = pd.DataFrame(fold_rows)
    summary_frame = pd.DataFrame(summary_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    permutation_frame = pd.DataFrame(permutation_rows)
    if formal:
        fold_frame.to_csv(FOLD_METRICS_PATH, index=False)
        summary_frame.to_csv(SUMMARY_PATH, index=False)
        bootstrap_frame.to_csv(BOOTSTRAP_PATH, index=False)
        permutation_frame.to_csv(PERMUTATION_PATH, index=False)
        if saved_oof:
            pd.concat(saved_oof, ignore_index=True).to_parquet(OOF_PATH, index=False)
    else:
        fold_frame.to_csv(REPORT_DIR / "h3_external_tusz_cv_fold_metrics_smoke.csv", index=False)
        summary_frame.to_csv(REPORT_DIR / "h3_external_tusz_summary_metrics_smoke.csv", index=False)

    if summary_frame.empty:
        raise RuntimeError("no valid external H3 analysis units were produced")
    model_summary = (
        summary_frame.groupby(["centering", "feature_set"], as_index=False)
        .agg(
            units=("macro_auroc", "count"),
            mean_macro_auroc=("macro_auroc", "mean"),
            median_macro_auroc=("macro_auroc", "median"),
            sd_macro_auroc=("macro_auroc", "std"),
            min_macro_auroc=("macro_auroc", "min"),
            max_macro_auroc=("macro_auroc", "max"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    model_summary["chance_macro_auroc"] = 0.5
    model_summary["fraction_units_above_chance"] = [
        float(np.mean(summary_frame[(summary_frame.centering == row.centering) & (summary_frame.feature_set == row.feature_set)]["macro_auroc"] > 0.5))
        for row in model_summary.itertuples(index=False)
    ]
    if formal:
        model_summary.to_csv(MODEL_SUMMARY_PATH, index=False)
    else:
        model_summary.to_csv(REPORT_DIR / "h3_external_tusz_model_summary_smoke.csv", index=False)

    manifest = {
        "analysis_id": RUN_ID,
        "source_run": SOURCE_RUN,
        "source_event_table": str(SOURCE_EVENT_PATH.relative_to(ROOT)),
        "source_event_table_sha256": sha256(SOURCE_EVENT_PATH),
        "source_scoring_manifest": str(SOURCE_SCORING_MANIFEST.relative_to(ROOT)) if SOURCE_SCORING_MANIFEST.exists() else None,
        "source_scoring_manifest_sha256": sha256(SOURCE_SCORING_MANIFEST) if SOURCE_SCORING_MANIFEST.exists() else None,
        "feature_table": str(FEATURE_PATH.relative_to(ROOT)) if FEATURE_PATH.is_relative_to(ROOT) else str(FEATURE_PATH),
        "feature_table_sha256": sha256(FEATURE_PATH),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(ROOT)) if FEATURE_MANIFEST_PATH.exists() and FEATURE_MANIFEST_PATH.is_relative_to(ROOT) else str(FEATURE_MANIFEST_PATH),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST_PATH) if FEATURE_MANIFEST_PATH.exists() else None,
        "patients": int(features["patient_id"].nunique()),
        "events": int(len(features)),
        "checkpoint_units_requested": int(len(checkpoint_keys)),
        "checkpoint_units_completed": int(summary_frame[["external_fold", "external_seed"]].drop_duplicates().shape[0]),
        "inner_patient_groupkfold": "GroupKFold(n_splits=5) on canonical TUSZ patient_id; fixed roster and assignment file",
        "patient_group_assignment_sha256": sha256(FOLD_ASSIGNMENTS_PATH),
        "models": list(MODELS),
        "feature_sets": {name: list(cols) for name, cols in FEATURE_SETS.items()},
        "centering_modes": list(CENTERING),
        "classifier": "multinomial_logistic_C1_class_weight_balanced",
        "scaler_fit_partition": "each TUSZ inner train fold only",
        "feature_cleaning": "train medians, train 1%-99% clipping, train-fitted StandardScaler",
        "test_used_for_selection": False,
        "threshold_selection": "none; detector-source labels are the fixed H3 target",
        "chance_macro_auroc": 0.5,
        "chance_accuracy": 1.0 / len(MODELS),
        "patient_centering": "label-free within-partition patient means, reported as sensitivity analysis",
        "bootstrap": {"unit": "patient", "replicates": n_bootstrap},
        "permutation": {"replicates": n_permutations, "unit": "within-patient held-out labels conditional on fixed scores", "interpretation": "secondary conditional null; not a refit-training null"},
        "external_labels_used_for_feature_or_model_selection": False,
        "formal": formal,
        "outputs": [str(path.relative_to(ROOT)) for path in (FOLD_METRICS_PATH, SUMMARY_PATH, BOOTSTRAP_PATH, PERMUTATION_PATH, MODEL_SUMMARY_PATH, OOF_PATH) if path.exists()] if formal else [],
    }
    if formal:
        OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(model_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
