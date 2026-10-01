"""Duration confounding analysis for the frozen H3 complete-event tables.

This analysis answers three narrowly defined questions without retraining any
seizure detector:

1. Can one complete-event duration predict detector source with three
   independent one-vs-rest (sigmoid-style) heads?
2. Can the same duration predict the three detector sources with a standard
   three-class classifier?
3. Can the already-completed complete-event EEG10 representation predict
   detector source, and can it predict event duration itself?

The complete-event rows are detector-source observations.  Therefore the
duration multi-label target is a one-hot, three-head encoding of the source
label, not the common-window multi-label task used by the separate 10-s
experiment.  No peak probability, detector probability, QC, HMM, or test
result is used as an input.  Duration is log1p-transformed because its
distribution is strongly right-skewed.

The existing complete-event EEG10 three-class results are deliberately
reused (rather than duplicated); this script adds the missing duration-only
classifiers, a direct EEG10-to-duration regression diagnostic, and robust
patient-level duration comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUN_ID = "h3-duration-eeg10-confounding-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
MODELS = ("singlem", "cbramod", "tcn_gat")
LABELS = {name: i for i, name in enumerate(MODELS)}
EEG10 = (
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
N_SPLITS = 5
N_BOOTSTRAP = 2000
RNG_SEED = 20260828
CHB_FEATURES = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h2h3_fp_features.parquet"
TUSZ_FEATURES = ROOT / "outputs/reports/h3-external-tusz-v1/tusz_h3_fp_features.parquet"
TUSZ_ASSIGNMENTS = ROOT / "outputs/reports/h3-external-tusz-v1/tusz_patient_groupkfold_assignments.csv"
CHB_EEG10_REFERENCE = ROOT / "outputs/reports/h3-chb-feature-ablations-v1/h3_chb_feature_ablation_summary.csv"
TUSZ_EEG10_REFERENCE = ROOT / "outputs/reports/h3-external-tusz-strict-eeg-10-v1/h3_external_tusz_model_summary.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite_scaled(train: pd.DataFrame, target: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    cols = list(columns)
    x_train = train.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_target = target.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = x_train.median().fillna(0.0)
    x_train = x_train.fillna(median)
    x_target = x_target.fillna(median)
    lower = x_train.quantile(0.01)
    upper = x_train.quantile(0.99)
    x_train = x_train.clip(lower=lower, upper=upper, axis="columns")
    x_target = x_target.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_target)


def multiclass_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(MODELS))), average="macro", zero_division=0)),
        "events": float(len(y)),
    }
    try:
        out["macro_auroc"] = float(
            roc_auc_score(y, proba, labels=list(range(len(MODELS))), multi_class="ovr", average="macro")
        )
        for i, model in enumerate(MODELS):
            out[f"{model}_auroc"] = float(roc_auc_score((y == i).astype(int), proba[:, i]))
    except ValueError:
        out["macro_auroc"] = math.nan
        for model in MODELS:
            out[f"{model}_auroc"] = math.nan
    return out


def multilabel_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    per_label_balanced: list[float] = []
    per_label_auc: list[float] = []
    for j in range(len(MODELS)):
        per_label_balanced.append(float(balanced_accuracy_score(y[:, j], pred[:, j])))
        try:
            per_label_auc.append(float(roc_auc_score(y[:, j], proba[:, j])))
        except ValueError:
            per_label_auc.append(math.nan)
    out: dict[str, float] = {
        "accuracy": float(np.mean(np.all(pred == y, axis=1))),
        "balanced_accuracy": float(np.nanmean(per_label_balanced)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_auroc": float(np.nanmean(per_label_auc)),
        "events": float(len(y)),
        "true_label_cardinality": float(np.mean(y.sum(axis=1))),
        "predicted_label_cardinality": float(np.mean(pred.sum(axis=1))),
        "predicted_onehot_fraction": float(np.mean(pred.sum(axis=1) == 1)),
    }
    for j, model in enumerate(MODELS):
        out[f"{model}_auroc"] = per_label_auc[j]
    return out


def _metadata(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "chbmit":
        cols = ["patient", "recording", "model", "start_s", "end_s"]
        out = frame.loc[:, cols].copy()
        out = out.rename(columns={"patient": "patient_id"})
    else:
        cols = ["patient_id", "recording", "model", "start_s", "end_s", "fold", "seed"]
        out = frame.loc[:, cols].copy()
    out["true_model"] = out["model"].astype(str)
    return out.drop(columns=["model"])


def fit_multiclass(train: pd.DataFrame, target: pd.DataFrame, columns: tuple[str, ...], dataset: str, random_state: int) -> tuple[dict[str, float], pd.DataFrame]:
    x_train, x_target = _finite_scaled(train, target, columns)
    y_train = train["model"].map(LABELS).astype(int).to_numpy()
    y_target = target["model"].map(LABELS).astype(int).to_numpy()
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", random_state=random_state)
    if len(np.unique(y_train)) != len(MODELS):
        raise RuntimeError("training partition does not contain all detector classes")
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_target)
    out = _metadata(target, dataset)
    out["prediction"] = np.argmax(proba, axis=1)
    for i, model in enumerate(MODELS):
        out[f"prob_{model}"] = proba[:, i]
    return multiclass_metrics(y_target, proba), out


def fit_multilabel(train: pd.DataFrame, target: pd.DataFrame, columns: tuple[str, ...], dataset: str, random_state: int) -> tuple[dict[str, float], pd.DataFrame]:
    x_train, x_target = _finite_scaled(train, target, columns)
    y_train = np.column_stack([(train["model"].astype(str).to_numpy() == model).astype(int) for model in MODELS])
    y_target = np.column_stack([(target["model"].astype(str).to_numpy() == model).astype(int) for model in MODELS])
    proba = np.zeros((len(target), len(MODELS)), dtype=float)
    for j in range(len(MODELS)):
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="liblinear", random_state=random_state + j)
        clf.fit(x_train, y_train[:, j])
        proba[:, j] = clf.predict_proba(x_target)[:, 1]
    out = _metadata(target, dataset)
    out["prediction_cardinality"] = (proba >= 0.5).sum(axis=1)
    for j, model in enumerate(MODELS):
        out[f"prob_{model}"] = proba[:, j]
        out[f"true_{model}"] = y_target[:, j]
    return multilabel_metrics(y_target, proba), out


def fit_duration_regression(train: pd.DataFrame, target: pd.DataFrame, dataset: str, random_state: int) -> tuple[dict[str, float], pd.DataFrame]:
    x_train, x_target = _finite_scaled(train, target, EEG10)
    y_train = train["log_duration"].to_numpy(float)
    y_target = target["log_duration"].to_numpy(float)
    reg = Ridge(alpha=1.0, random_state=random_state)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_target)
    out = _metadata(target, dataset)
    out["true_log_duration"] = y_target
    out["pred_log_duration"] = pred
    rho, rho_p = spearmanr(y_target, pred)
    return {
        "events": float(len(target)),
        "mae_log_duration": float(mean_absolute_error(y_target, pred)),
        "rmse_log_duration": float(np.sqrt(mean_squared_error(y_target, pred))),
        "r2_log_duration": float(r2_score(y_target, pred)),
        "spearman_log_duration": float(rho) if np.isfinite(rho) else math.nan,
        "spearman_p": float(rho_p) if np.isfinite(rho_p) else math.nan,
        "mae_duration_s": float(mean_absolute_error(np.expm1(y_target), np.maximum(0.0, np.expm1(pred)))),
    }, out


def bootstrap_patient_auc(pred: pd.DataFrame, n_bootstrap: int, patient_col: str, seed: int) -> dict[str, float]:
    patients = np.asarray(sorted(pred[patient_col].astype(str).unique()), dtype=object)
    y = pred["true_model"].map(LABELS).to_numpy(int)
    proba = pred[[f"prob_{m}" for m in MODELS]].to_numpy(float)
    positions = {p: np.flatnonzero(pred[patient_col].astype(str).to_numpy() == p) for p in patients}
    observed = multiclass_metrics(y, proba).get("macro_auroc", math.nan)
    values: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([positions[p] for p in sampled])
        try:
            values.append(float(roc_auc_score(y[idx], proba[idx], labels=list(range(len(MODELS))), multi_class="ovr", average="macro")))
        except ValueError:
            continue
    return {
        "bootstrap_mean_macro_auroc": float(np.mean(values)) if values else math.nan,
        "bootstrap_q025_macro_auroc": float(np.quantile(values, 0.025)) if values else math.nan,
        "bootstrap_q975_macro_auroc": float(np.quantile(values, 0.975)) if values else math.nan,
        "observed_macro_auroc": float(observed),
        "bootstrap_valid_replicates": int(len(values)),
        "bootstrap_patients": int(len(patients)),
        "bootstrap_replicates_requested": int(n_bootstrap),
    }


def _make_tusz_assignment(features: pd.DataFrame) -> pd.DataFrame:
    patients = sorted(features["patient_id"].astype(str).unique())
    if TUSZ_ASSIGNMENTS.exists():
        roster = pd.read_csv(TUSZ_ASSIGNMENTS)
        if set(roster.columns) >= {"patient_id", "cv_fold"} and set(roster.patient_id.astype(str)) == set(patients):
            roster = roster.loc[:, ["patient_id", "cv_fold"]].copy()
            roster["patient_id"] = roster.patient_id.astype(str)
            roster["cv_fold"] = roster.cv_fold.astype(int)
            if roster.patient_id.nunique() == len(patients) and set(roster.cv_fold) == set(range(N_SPLITS)):
                return roster.sort_values("patient_id").reset_index(drop=True)
    roster = pd.DataFrame({"patient_id": patients})
    assignment = np.full(len(roster), -1, dtype=int)
    for fold, (_train, test) in enumerate(GroupKFold(n_splits=N_SPLITS).split(roster, groups=roster.patient_id)):
        assignment[test] = fold
    roster["cv_fold"] = assignment
    return roster


def duration_distribution(features: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if dataset == "chbmit":
        patient_col = "patient"
        key = ["patient", "recording", "model", "start_s", "end_s"]
    else:
        patient_col = "patient_id"
        key = ["patient_id", "recording", "model", "start_s", "end_s"]
    dedup = features.drop_duplicates(key).copy()
    rows: list[dict[str, object]] = []
    patient_model = dedup.groupby([patient_col, "model"], as_index=False)["event_duration_s"].median()
    for model, group in dedup.groupby("model", sort=True):
        values = group.event_duration_s.to_numpy(float)
        rows.append({
            "dataset": dataset,
            "model": str(model),
            "unique_events": int(len(values)),
            "patients": int(group[patient_col].nunique()),
            "mean_duration_s": float(np.mean(values)),
            "median_duration_s": float(np.median(values)),
            "p10_duration_s": float(np.quantile(values, 0.10)),
            "p25_duration_s": float(np.quantile(values, 0.25)),
            "p75_duration_s": float(np.quantile(values, 0.75)),
            "p90_duration_s": float(np.quantile(values, 0.90)),
            "iqr_duration_s": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        })
    pivot = patient_model.pivot(index=patient_col, columns="model", values="event_duration_s")
    pair_rows: list[dict[str, object]] = []
    for i, left in enumerate(MODELS):
        for right in MODELS[i + 1 :]:
            common = pivot[[left, right]].dropna()
            diff = common[left].to_numpy(float) - common[right].to_numpy(float)
            p_value = math.nan
            if len(diff) >= 5 and np.any(np.abs(diff) > 1e-12):
                try:
                    p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = math.nan
            pair_rows.append({
                "dataset": dataset,
                "left_model": left,
                "right_model": right,
                "patients_paired": int(len(diff)),
                "median_paired_difference_s": float(np.median(diff)) if len(diff) else math.nan,
                "q025_paired_difference_s": float(np.quantile(diff, 0.025)) if len(diff) else math.nan,
                "q975_paired_difference_s": float(np.quantile(diff, 0.975)) if len(diff) else math.nan,
                "wilcoxon_p": p_value,
            })
    p_values = np.asarray([row["wilcoxon_p"] for row in pair_rows], dtype=float)
    order = np.argsort(np.where(np.isfinite(p_values), p_values, np.inf))
    q_values = np.full(len(pair_rows), np.nan, dtype=float)
    finite = np.isfinite(p_values)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        sorted_indices = finite_indices[np.argsort(p_values[finite])]
        running = 1.0
        for rank, index in reversed(list(enumerate(sorted_indices, start=1))):
            running = min(running, p_values[index] * len(sorted_indices) / rank)
            q_values[index] = running
    for row, q_value in zip(pair_rows, q_values):
        row["wilcoxon_q_bh"] = float(q_value) if np.isfinite(q_value) else math.nan
    kw_p = math.nan
    groups = [patient_model.loc[patient_model.model == model, "event_duration_s"].to_numpy(float) for model in MODELS]
    if all(len(group) >= 5 for group in groups):
        try:
            kw_p = float(kruskal(*groups).pvalue)
        except ValueError:
            kw_p = math.nan
    audit = {
        "dataset": dataset,
        "duration_source_rows": int(len(features)),
        "unique_event_rows_after_deduplication": int(len(dedup)),
        "deduplication_key": key,
        "patient_model_median_rows": int(len(patient_model)),
        "kruskal_p_patient_medians": kw_p,
    }
    return pd.DataFrame(rows), pd.DataFrame(pair_rows), audit


def _reference_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if CHB_EEG10_REFERENCE.exists():
        chb = pd.read_csv(CHB_EEG10_REFERENCE)
        chb = chb[(chb.feature_set == "strict_eeg_only10") & (chb.patient_centered == False)]  # noqa: E712
        for row in chb.to_dict("records"):
            rows.append({"dataset": "chbmit", "experiment": "full_event_eeg10_multiclass_reused", "source": str(CHB_EEG10_REFERENCE.relative_to(ROOT)), **row})
    if TUSZ_EEG10_REFERENCE.exists():
        tusz = pd.read_csv(TUSZ_EEG10_REFERENCE)
        tusz = tusz[(tusz.feature_set == "strict_eeg_only10") & (tusz.centering == "raw")]
        for row in tusz.to_dict("records"):
            rows.append({"dataset": "tusz", "experiment": "full_event_eeg10_multiclass_reused", "source": str(TUSZ_EEG10_REFERENCE.relative_to(ROOT)), **row})
    return pd.DataFrame(rows)


def _add_metric_row(rows: list[dict[str, object]], dataset: str, experiment: str, unit: str, fold: int | str, role: str, metrics: dict[str, float], patients: int) -> None:
    rows.append({"dataset": dataset, "experiment": experiment, "unit": str(unit), "fold": fold, "split_role": role, "test_used_for_selection": False, "patients": int(patients), **metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        "classification_metrics.csv",
        "regression_metrics.csv",
        "classification_patient_bootstrap.csv",
        "duration_distribution.csv",
        "duration_pairwise.csv",
        "duration_audit.json",
        "completed_reference.csv",
        "test_predictions.parquet",
        "regression_predictions.parquet",
        "manifest.json",
    ]
    existing = [name for name in expected_outputs if (REPORT_DIR / name).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing duration analysis outputs: {existing}")

    chb = pd.read_parquet(CHB_FEATURES)
    tusz = pd.read_parquet(TUSZ_FEATURES)
    if "event_type" in chb.columns:
        chb = chb[chb.event_type.astype(str) == "fp"].copy()
    chb["model"] = chb.model.astype(str)
    tusz["model"] = tusz.model.astype(str)
    chb["log_duration"] = np.log1p(np.maximum(chb.event_duration_s.astype(float), 0.0))
    tusz["log_duration"] = np.log1p(np.maximum(tusz.event_duration_s.astype(float), 0.0))

    duration_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    duration_audit: list[dict[str, object]] = []
    for dataset, frame in (("chbmit", chb), ("tusz", tusz)):
        summary, pairs, audit = duration_distribution(frame, dataset)
        duration_tables.append(summary)
        pair_tables.append(pairs)
        duration_audit.append(audit)

    classification_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    regression_prediction_parts: list[pd.DataFrame] = []

    # CHB: use the frozen canonical patient GroupKFold partition assignments.
    for fold in range(N_SPLITS):
        fold_frame = chb[chb.split.astype(int) == fold].copy()
        train = fold_frame[fold_frame.partition.astype(str) == "train"].copy()
        validation = fold_frame[fold_frame.partition.astype(str) == "validation"].copy()
        test = fold_frame[fold_frame.partition.astype(str) == "test"].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise RuntimeError(f"CHB split {fold} has an empty partition")
        for experiment, fit_func, columns in (
            ("duration_only_multilabel", fit_multilabel, ("log_duration",)),
            ("duration_only_multiclass", fit_multiclass, ("log_duration",)),
        ):
            val_metrics, _ = fit_func(train, validation, columns, "chbmit", RNG_SEED + fold)
            test_metrics, pred = fit_func(train, test, columns, "chbmit", RNG_SEED + fold)
            _add_metric_row(classification_rows, "chbmit", experiment, f"split{fold}", fold, "validation", val_metrics, validation.patient.nunique())
            _add_metric_row(classification_rows, "chbmit", experiment, f"split{fold}", fold, "outer_test", test_metrics, test.patient.nunique())
            pred["dataset"] = "chbmit"
            pred["experiment"] = experiment
            pred["unit"] = f"split{fold}"
            pred["fold"] = fold
            pred["split_role"] = "outer_test"
            prediction_parts.append(pred)
        val_reg_metrics, _ = fit_duration_regression(train, validation, "chbmit", RNG_SEED + fold)
        test_reg_metrics, reg_pred = fit_duration_regression(train, test, "chbmit", RNG_SEED + fold)
        _add_metric_row(regression_rows, "chbmit", "eeg10_to_duration_ridge", f"split{fold}", fold, "validation", val_reg_metrics, validation.patient.nunique())
        _add_metric_row(regression_rows, "chbmit", "eeg10_to_duration_ridge", f"split{fold}", fold, "outer_test", test_reg_metrics, test.patient.nunique())
        reg_pred["dataset"] = "chbmit"
        reg_pred["experiment"] = "eeg10_to_duration_ridge"
        reg_pred["unit"] = f"split{fold}"
        reg_pred["fold"] = fold
        reg_pred["split_role"] = "outer_test"
        regression_prediction_parts.append(reg_pred)

    # TUSZ: reuse the frozen patient-level GroupKFold assignment and evaluate
    # each of the 15 frozen detector-output units with held-out patient folds.
    roster = _make_tusz_assignment(tusz)
    fold_lookup = roster.set_index("patient_id").cv_fold.to_dict()
    tusz["cv_fold"] = tusz.patient_id.astype(str).map(fold_lookup).astype(int)
    units = sorted(tusz[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    for external_fold, external_seed in units:
        unit = tusz[(tusz.fold.astype(int) == int(external_fold)) & (tusz.seed.astype(int) == int(external_seed))].copy()
        unit_id = f"fold{int(external_fold)}_seed{int(external_seed)}"
        for cv_fold in range(N_SPLITS):
            train = unit[unit.cv_fold != cv_fold].copy()
            test = unit[unit.cv_fold == cv_fold].copy()
            if min(len(train), len(test)) == 0:
                continue
            for experiment, fit_func, columns in (
                ("duration_only_multilabel", fit_multilabel, ("log_duration",)),
                ("duration_only_multiclass", fit_multiclass, ("log_duration",)),
            ):
                test_metrics, pred = fit_func(train, test, columns, "tusz", RNG_SEED + int(external_fold) * 10000 + int(external_seed) + cv_fold)
                _add_metric_row(classification_rows, "tusz", experiment, unit_id, cv_fold, "held_out_patient_fold", test_metrics, test.patient_id.nunique())
                pred["dataset"] = "tusz"
                pred["experiment"] = experiment
                pred["unit"] = unit_id
                pred["cv_fold"] = cv_fold
                pred["split_role"] = "held_out_patient_fold"
                prediction_parts.append(pred)
            test_reg_metrics, reg_pred = fit_duration_regression(train, test, "tusz", RNG_SEED + int(external_fold) * 10000 + int(external_seed) + cv_fold)
            _add_metric_row(regression_rows, "tusz", "eeg10_to_duration_ridge", unit_id, cv_fold, "held_out_patient_fold", test_reg_metrics, test.patient_id.nunique())
            reg_pred["dataset"] = "tusz"
            reg_pred["experiment"] = "eeg10_to_duration_ridge"
            reg_pred["unit"] = unit_id
            reg_pred["cv_fold"] = cv_fold
            reg_pred["split_role"] = "held_out_patient_fold"
            regression_prediction_parts.append(reg_pred)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    # CHB bootstrap is over the combined outer-test predictions; TUSZ is per
    # external checkpoint unit, matching the existing TUSZ H3 reporting unit.
    for dataset, experiment in [("chbmit", "duration_only_multilabel"), ("chbmit", "duration_only_multiclass")]:
        pred = predictions[(predictions.dataset == dataset) & (predictions.experiment == experiment)].copy()
        row = {"dataset": dataset, "experiment": experiment, "unit": "all_outer_tests", **bootstrap_patient_auc(pred, args.bootstrap, "patient_id", RNG_SEED + len(bootstrap_rows))}
        bootstrap_rows.append(row)
    for experiment in ("duration_only_multilabel", "duration_only_multiclass"):
        for unit_id in sorted(predictions.loc[(predictions.dataset == "tusz") & (predictions.experiment == experiment), "unit"].unique()):
            pred = predictions[(predictions.dataset == "tusz") & (predictions.experiment == experiment) & (predictions.unit == unit_id)].copy()
            row = {"dataset": "tusz", "experiment": experiment, "unit": unit_id, **bootstrap_patient_auc(pred, args.bootstrap, "patient_id", RNG_SEED + len(bootstrap_rows))}
            bootstrap_rows.append(row)

    duration_distribution_frame = pd.concat(duration_tables, ignore_index=True)
    duration_pairwise_frame = pd.concat(pair_tables, ignore_index=True)
    duration_distribution_frame.to_csv(REPORT_DIR / "duration_distribution.csv", index=False)
    duration_pairwise_frame.to_csv(REPORT_DIR / "duration_pairwise.csv", index=False)
    (REPORT_DIR / "duration_audit.json").write_text(json.dumps(duration_audit, indent=2, sort_keys=True) + "\n")
    classification_frame = pd.DataFrame(classification_rows)
    regression_frame = pd.DataFrame(regression_rows)
    classification_frame.to_csv(REPORT_DIR / "classification_metrics.csv", index=False)
    regression_frame.to_csv(REPORT_DIR / "regression_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "classification_patient_bootstrap.csv", index=False)
    predictions.to_parquet(REPORT_DIR / "test_predictions.parquet", index=False)
    pd.concat(regression_prediction_parts, ignore_index=True).to_parquet(REPORT_DIR / "regression_predictions.parquet", index=False)
    reference = _reference_rows()
    reference.to_csv(REPORT_DIR / "completed_reference.csv", index=False)

    manifest = {
        "analysis_id": RUN_ID,
        "status": "complete",
        "purpose": "complete-event duration confounding and EEG10 duration derivability for H3 detector-source analysis",
        "datasets": {
            "chbmit": {"feature_table": str(CHB_FEATURES.relative_to(ROOT)), "sha256": sha256(CHB_FEATURES), "patients": int(chb.patient.nunique()), "rows": int(len(chb)), "grouping": "frozen canonical patient GroupKFold split/partition"},
            "tusz": {"feature_table": str(TUSZ_FEATURES.relative_to(ROOT)), "sha256": sha256(TUSZ_FEATURES), "patients": int(tusz.patient_id.nunique()), "rows": int(len(tusz)), "grouping": "frozen/reused patient-level GroupKFold assignment; 15 external fold-seed units"},
        },
        "models": list(MODELS),
        "new_experiments": {
            "duration_only_multilabel": {"input": ["log1p(event_duration_s)"], "target": "three independent one-vs-rest detector-source heads; one-hot event-row encoding", "threshold": "fixed 0.5; no test selection"},
            "duration_only_multiclass": {"input": ["log1p(event_duration_s)"], "target": "three-class detector-source label", "classifier": "balanced multinomial logistic regression C=1", "threshold": "argmax; no test selection"},
            "eeg10_to_duration_ridge": {"input": list(EEG10), "target": "log1p(event_duration_s)", "classifier": "Ridge(alpha=1) after train-only clipping/scaling", "purpose": "direct duration-derivability diagnostic"},
        },
        "reused_experiment": "full_event_eeg10_multiclass_reused from the frozen h3-chb-feature-ablations-v1 and h3-external-tusz-strict-eeg-10-v1 outputs; it was not rerun",
        "duration_deduplication": "unique patient-recording-model-start-end event key for descriptive duration comparisons",
        "preprocessing": "train-only median imputation, 1%-99% clipping, StandardScaler; log1p duration",
        "patient_bootstrap": {"replicates": int(args.bootstrap), "unit": "patient", "metric": "macro AUROC for duration classifiers"},
        "test_used_for_selection": False,
        "outputs": expected_outputs,
        "claim_boundary": "duration-only source classification measures confounding/identifiability, not detector performance; complete-event EEG10 summarizes the event interval and may carry duration-correlated information",
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": manifest, "classification_rows": len(classification_frame), "regression_rows": len(regression_frame)}, indent=2, sort_keys=True))
    print("\\nDuration distribution:")
    print(duration_distribution_frame.to_string(index=False))
    print("\\nNew classification test summary:")
    print(classification_frame[classification_frame.split_role != "validation"].groupby(["dataset", "experiment"], as_index=False)[["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"]].mean().to_string(index=False))
    print("\\nEEG10 -> duration regression test summary:")
    print(regression_frame[regression_frame.split_role != "validation"].groupby("dataset", as_index=False)[["mae_log_duration", "rmse_log_duration", "r2_log_duration", "spearman_log_duration", "mae_duration_s"]].mean().to_string(index=False))


if __name__ == "__main__":
    main()
