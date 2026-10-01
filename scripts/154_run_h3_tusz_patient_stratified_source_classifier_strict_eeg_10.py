"""TUSZ strict-EEG-10 patient and patient-stratum extension of H3.

This is the TUSZ counterpart to the already frozen CHB-MIT patient analysis.
It keeps the H3 target fixed: the detector source (SingLEM, CBraMod, or
TCN-GAT) of an already observed false-positive event.  It is not a seizure
detector, router, or model-selection experiment.

For every TUSZ patient, rows are ordered by the recording path and event
time.  The first 70 percent are used only to fit a source classifier and to
rank patients by detector-centroid separation; the last 30 percent are a
chronological descriptive evaluation.  The two extreme five-patient strata
are then used for pooled early-fit classifiers and own/cross-stratum late
evaluation.  The parent external detector outputs are frozen and no
threshold, detector, or external artifact label is fitted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUN_ID = os.environ.get(
    "BFA_RUN_ID", "h3-patient-stratified-source-classifier-tusz-strict-eeg-10-v1"
)
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
PARENT_RUN = "v3-groupkfold-confirmatory-v1"
FEATURE_RUN = "h3-external-tusz-v1"
FEATURE_PATH = ROOT / "outputs" / "reports" / FEATURE_RUN / "tusz_h3_fp_features.parquet"
FEATURE_MANIFEST_PATH = ROOT / "outputs" / "reports" / FEATURE_RUN / "tusz_h3_fp_features_manifest.json"
SOURCE_EVENT_PATH = ROOT / "outputs" / "reports" / PARENT_RUN / "tusz_groupkfold_external_v1_events.parquet"
SOURCE_SCORING_MANIFEST = ROOT / "outputs" / "reports" / PARENT_RUN / "tusz_groupkfold_external_v1_scoring_manifest.json"

MODELS = ("singlem", "cbramod", "tcn_gat")
MODEL_TO_ID = {model: index for index, model in enumerate(MODELS)}
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
SEEDS = (17, 42, 3407)
N_EXTREME = 5
FIT_FRACTION = 0.70
N_BOOTSTRAP = 2000
N_ASSOCIATION_PERMUTATIONS = 1000
RNG_SEED = 20260828


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_progress(phase: str, **payload: object) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"analysis_id": RUN_ID, "phase": phase, "updated_unix": time.time(), **payload}
    (REPORT_DIR / "progress.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def center_fit_and_eval(
    fit: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fit a patient mean on the early portion and apply it to both portions."""

    fit_out = fit.copy()
    eval_out = evaluation.copy()
    values = fit_out.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    global_mean = values.mean().fillna(0.0).to_numpy(float)
    patient_mean = values.mean().fillna(pd.Series(global_mean, index=FEATURES)).to_numpy(float)
    fit_matrix = fit_out.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    eval_matrix = eval_out.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    fit_matrix = np.nan_to_num(fit_matrix, nan=global_mean) - patient_mean
    eval_matrix = np.nan_to_num(eval_matrix, nan=global_mean) - patient_mean
    fit_out.loc[:, FEATURES] = fit_matrix
    eval_out.loc[:, FEATURES] = eval_matrix
    return fit_out, eval_out, patient_mean


def metrics_row(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    scope: str,
    seed: int,
    patient: str | None,
    eval_group: str | None,
    fit_event_rows: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = np.argmax(probabilities, axis=1)
    row: dict[str, object] = {
        "scope": scope,
        "seed": int(seed),
        "patient": patient,
        "eval_group": eval_group,
        "status": "complete",
        "fit_event_rows": int(fit_event_rows),
        "eval_event_rows": int(len(y_true)),
        "fit_source_classes": len(MODELS),
        "event_rows": int(len(y_true)),
        "patients": 0,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=list(range(len(MODELS))), average="macro", zero_division=0)
        ),
        "chance_accuracy": 1.0 / len(MODELS),
        "chance_macro_auroc": 0.5,
    }
    try:
        row["macro_auroc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=list(range(len(MODELS))),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        row["macro_auroc"] = math.nan
    confusion = confusion_matrix(y_true, predictions, labels=list(range(len(MODELS))))
    confusion_rows = []
    for true_id, true_model in enumerate(MODELS):
        for pred_id, pred_model in enumerate(MODELS):
            confusion_rows.append(
                {
                    "scope": scope,
                    "seed": int(seed),
                    "patient": patient,
                    "eval_group": eval_group,
                    "true_model": true_model,
                    "predicted_model": pred_model,
                    "count": int(confusion[true_id, pred_id]),
                }
            )
    return row, confusion_rows


def fit_source_classifier(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    scope: str,
    seed: int,
    patient: str | None,
    eval_group: str | None,
    artifact_path: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame, dict[str, object] | None]:
    fit_values = fit.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    eval_values = evaluation.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    median = fit_values.median().fillna(0.0)
    fit_values = fit_values.fillna(median)
    eval_values = eval_values.fillna(median)
    lower = fit_values.quantile(0.01)
    upper = fit_values.quantile(0.99)
    fit_values = fit_values.clip(lower=lower, upper=upper, axis="columns")
    eval_values = eval_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(fit_values)
    y_fit = fit.model.map(MODEL_TO_ID).astype(int).to_numpy()
    y_eval = evaluation.model.map(MODEL_TO_ID).astype(int).to_numpy()
    if len(np.unique(y_fit)) != len(MODELS):
        raise RuntimeError(f"{scope} seed={seed} fit portion does not contain all detector classes")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=RNG_SEED + int(seed),
    )
    classifier.fit(scaler.transform(fit_values), y_fit)
    probabilities = classifier.predict_proba(scaler.transform(eval_values))
    row, confusion_rows = metrics_row(
        y_eval, probabilities, scope, seed, patient, eval_group, fit_event_rows=len(fit)
    )
    pred = evaluation[["patient_id", "recording", "event_index", "start_s", "end_s", "model", "fold", "seed"]].copy()
    pred["scope"] = scope
    pred["eval_group"] = eval_group
    pred["true_model"] = pred.model.astype(str)
    pred["predicted_model"] = [MODELS[index] for index in np.argmax(probabilities, axis=1)]
    pred["predicted_probability"] = np.max(probabilities, axis=1)
    for index, model in enumerate(MODELS):
        pred[f"probability_{model}"] = probabilities[:, index]
    pred = pred.drop(columns=["model"])
    bundle = {
        "classifier": classifier,
        "scaler": scaler,
        "features": list(FEATURES),
        "model_labels": list(MODELS),
        "fit_medians": median.to_dict(),
        "fit_clip_lower": lower.to_dict(),
        "fit_clip_upper": upper.to_dict(),
        "scope": scope,
        "seed": int(seed),
    }
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, artifact_path)
    row["patients"] = int(pred.patient_id.nunique())
    return row, confusion_rows, pred, bundle


def association_test(fit_rows: pd.DataFrame) -> dict[str, object]:
    table = pd.crosstab(fit_rows.patient_id.astype(str), fit_rows.model.astype(str)).reindex(columns=MODELS, fill_value=0)
    observed = table.to_numpy(dtype=int)
    if observed.shape[0] > 1 and np.all(observed.sum(axis=0) > 0):
        chi2, p_value, dof, expected = chi2_contingency(observed)
    else:
        chi2, p_value, dof, expected = math.nan, math.nan, math.nan, np.zeros_like(observed, dtype=float)
    labels = fit_rows.model.astype(str).map(MODEL_TO_ID).to_numpy(copy=True)
    patient_index = fit_rows.patient_id.astype(str).to_numpy(copy=False)
    rng = np.random.default_rng(RNG_SEED + 911)
    null_chi2: list[float] = []
    for _ in range(N_ASSOCIATION_PERMUTATIONS):
        # Shuffle source labels across patients while preserving each
        # patient's row count and the global detector totals.  A within-patient
        # shuffle would leave the patient x detector contingency table
        # unchanged and is therefore not a valid association null.
        shuffled = rng.permutation(labels)
        null_table = np.zeros_like(observed, dtype=int)
        patient_codes, _ = pd.factorize(patient_index, sort=True)
        np.add.at(null_table, (patient_codes, shuffled), 1)
        if null_table.shape[0] > 1 and np.all(null_table.sum(axis=0) > 0):
            expected_null = null_table.sum(axis=1, keepdims=True) * null_table.sum(axis=0, keepdims=True) / max(1, null_table.sum())
            null_chi2.append(float(np.sum((null_table - expected_null) ** 2 / np.where(expected_null > 0, expected_null, 1.0))))
    null = np.asarray(null_chi2, dtype=float)
    empirical = float((1 + np.sum(null >= float(chi2))) / (1 + len(null))) if np.isfinite(chi2) and len(null) else math.nan
    total_events = max(1, int(observed.sum()))
    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = float(math.sqrt(float(chi2) / total_events / min_dim)) if np.isfinite(chi2) and min_dim > 0 else math.nan
    return {
        "method": "patient x detector-source contingency chi-square",
        "null": "global detector-label shuffle preserving patient row counts and global detector totals",
        "fit_rows": int(len(fit_rows)),
        "patients": int(fit_rows.patient_id.nunique()),
        "models": list(MODELS),
        "observed_table": {str(patient): {model: int(table.loc[patient, model]) for model in MODELS} for patient in table.index},
        "chi2": float(chi2),
        "cramers_v": cramers_v,
        "degrees_of_freedom": int(dof) if np.isfinite(dof) else None,
        "p_value": float(p_value),
        "permutation_replicates": N_ASSOCIATION_PERMUTATIONS,
        "permutation_p_value": empirical,
        "null_chi2_q975": float(np.quantile(null, 0.975)) if len(null) else math.nan,
    }


def patient_bootstrap(patient_metrics: pd.DataFrame, metric: str, scope: str, eval_group: str | None) -> dict[str, object]:
    subset = patient_metrics[patient_metrics.scope.astype(str) == scope].copy()
    if eval_group is not None:
        subset = subset[subset.eval_group.astype(str) == eval_group]
    if subset.empty or metric not in subset:
        return {"scope": scope, "eval_group": eval_group, "metric": metric, "status": "empty"}
    # Average seeds within each patient first so no patient with many rows
    # dominates the patient-level interval.
    per_patient = subset.groupby("patient_id", as_index=False)[metric].mean()
    values = per_patient[metric].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"scope": scope, "eval_group": eval_group, "metric": metric, "status": "empty"}
    rng = np.random.default_rng(RNG_SEED + sum(ord(ch) for ch in f"{scope}:{eval_group}:{metric}"))
    sampled = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(N_BOOTSTRAP)])
    return {
        "scope": scope,
        "eval_group": eval_group,
        "metric": metric,
        "status": "complete",
        "patients": int(len(values)),
        "valid_patients": int(len(values)),
        "mean": float(np.mean(values)),
        "q025": float(np.quantile(sampled, 0.025)),
        "q975": float(np.quantile(sampled, 0.975)),
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_unit": "patient",
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for required_path in (FEATURE_PATH, FEATURE_MANIFEST_PATH, SOURCE_EVENT_PATH, SOURCE_SCORING_MANIFEST):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    write_progress("loading", feature_set="strict-EEG-10", feature_count=len(FEATURES))
    features = pd.read_parquet(FEATURE_PATH).copy()
    required = {
        "patient_id", "recording", "event_index", "start_s", "end_s", "model", "fold", "seed", *FEATURES
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"missing TUSZ feature columns: {missing}")
    features["patient_id"] = features.patient_id.astype(str)
    features["recording"] = features.recording.astype(str)
    features["model"] = features.model.astype(str)
    features["fold"] = features.fold.astype(int)
    features["seed"] = features.seed.astype(int)
    features = features.dropna(subset=list(FEATURES)).copy()
    patients = sorted(features.patient_id.unique())
    if len(patients) < 2 * N_EXTREME + 1:
        raise RuntimeError(f"not enough TUSZ patients for extreme strata: {len(patients)}")
    write_progress("loaded", patients=len(patients), feature_rows=len(features))

    patient_fit: dict[tuple[str, int], pd.DataFrame] = {}
    patient_eval: dict[tuple[str, int], pd.DataFrame] = {}
    patient_means: dict[tuple[str, int], np.ndarray] = {}
    separation_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for patient_index, patient in enumerate(patients):
        for seed in SEEDS:
            rows = features[(features.patient_id == patient) & (features.seed == seed)].copy()
            if len(rows) < 2:
                raise RuntimeError(f"patient={patient} seed={seed} has fewer than two events")
            rows = rows.sort_values(
                ["recording", "event_index", "start_s", "end_s", "model", "fold"], kind="stable"
            ).reset_index(drop=True)
            cut = max(1, min(len(rows) - 1, int(math.floor(FIT_FRACTION * len(rows)))))
            fit = rows.iloc[:cut].copy()
            evaluation = rows.iloc[cut:].copy()
            fit_centered, eval_centered, mean = center_fit_and_eval(fit, evaluation)
            patient_fit[(patient, seed)] = fit_centered
            patient_eval[(patient, seed)] = eval_centered
            patient_means[(patient, seed)] = mean
            # Detector centroid distances are calculated on the early portion
            # only, before any late evaluation is observed.
            centroids: dict[str, np.ndarray] = {}
            counts: dict[str, int] = {}
            for model in MODELS:
                subset = fit_centered[fit_centered.model == model]
                counts[model] = int(len(subset))
                if not subset.empty:
                    centroids[model] = subset.loc[:, FEATURES].to_numpy(float).mean(axis=0)
            distances: list[float] = []
            for left_index, left in enumerate(MODELS):
                for right in MODELS[left_index + 1 :]:
                    if left not in centroids or right not in centroids:
                        continue
                    distance = float(np.linalg.norm(centroids[left] - centroids[right]))
                    distances.append(distance)
                    distance_rows.append({"patient_id": patient, "seed": seed, "left_model": left, "right_model": right, "centroid_distance": distance})
            separation_rows.append({
                "patient_id": patient,
                "seed": seed,
                "fit_event_rows": int(len(fit_centered)),
                "eval_event_rows": int(len(eval_centered)),
                "mean_pairwise_centroid_distance": float(np.mean(distances)) if distances else math.nan,
                "min_pairwise_centroid_distance": float(np.min(distances)) if distances else math.nan,
                "max_pairwise_centroid_distance": float(np.max(distances)) if distances else math.nan,
                **{f"{model}_fit_events": counts[model] for model in MODELS},
            })
            counts_fit = fit.model.value_counts().to_dict()
            counts_eval = evaluation.model.value_counts().to_dict()
            for partition, count_map, n_rows in (("fit", counts_fit, len(fit)), ("chronological_eval", counts_eval, len(evaluation))):
                entropy = 0.0
                for model in MODELS:
                    proportion = float(count_map.get(model, 0) / n_rows) if n_rows else 0.0
                    if proportion > 0:
                        entropy -= proportion * math.log(proportion)
                entropy /= math.log(len(MODELS))
                distribution_rows.append({"patient_id": patient, "seed": seed, "partition": partition, "event_rows": int(n_rows), "source_entropy_normalized": entropy, **{f"{model}_count": int(count_map.get(model, 0)) for model in MODELS}})
        if patient_index == 0 or (patient_index + 1) % 5 == 0:
            write_progress("patient_separation", patients_done=patient_index + 1, patients_total=len(patients))

    separation_df = pd.DataFrame(separation_rows)
    ranking = separation_df.groupby("patient_id", as_index=False).agg(
        mean_pairwise_centroid_distance=("mean_pairwise_centroid_distance", "mean"),
        sd_pairwise_centroid_distance=("mean_pairwise_centroid_distance", "std"),
        min_pairwise_centroid_distance=("min_pairwise_centroid_distance", "mean"),
        max_pairwise_centroid_distance=("max_pairwise_centroid_distance", "mean"),
        fit_event_rows=("fit_event_rows", "mean"),
    ).sort_values(["mean_pairwise_centroid_distance", "patient_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranking["rank_high_to_low"] = np.arange(1, len(ranking) + 1)
    ranking["stratum"] = "middle"
    ranking.loc[: N_EXTREME - 1, "stratum"] = "high_separation"
    ranking.loc[len(ranking) - N_EXTREME :, "stratum"] = "low_separation"
    high_patients = ranking.loc[ranking.stratum == "high_separation", "patient_id"].tolist()
    low_patients = ranking.loc[ranking.stratum == "low_separation", "patient_id"].tolist()
    stratum_by_patient = ranking.set_index("patient_id").stratum.to_dict()
    ranking.to_csv(REPORT_DIR / "patient_detector_separation_ranking.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(REPORT_DIR / "patient_detector_centroid_distances.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(REPORT_DIR / "patient_detector_source_distribution.csv", index=False)
    fit_all = pd.concat(list(patient_fit.values()), ignore_index=True)
    association = association_test(fit_all)
    (REPORT_DIR / "patient_detector_association_stats.json").write_text(json.dumps(association, indent=2, sort_keys=True) + "\n")
    write_progress("strata_frozen", high_patients=high_patients, low_patients=low_patients)

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    artifact_rows: list[dict[str, object]] = []
    skipped_patient_seeds: list[dict[str, object]] = []
    for patient_index, patient in enumerate(patients):
        for seed in SEEDS:
            artifact = REPORT_DIR / f"source_classifier_patient_{patient}_seed{seed}.joblib"
            try:
                row, cm, pred, _ = fit_source_classifier(
                    patient_fit[(patient, seed)], patient_eval[(patient, seed)], "single_patient", seed,
                    patient, stratum_by_patient[patient], artifact
                )
            except RuntimeError as exc:
                skipped = {
                    "scope": "single_patient",
                    "seed": int(seed),
                    "patient": patient,
                    "eval_group": stratum_by_patient[patient],
                    "status": "skipped_missing_fit_source_class",
                    "reason": str(exc),
                    "fit_event_rows": int(len(patient_fit[(patient, seed)])),
                    "eval_event_rows": int(len(patient_eval[(patient, seed)])),
                }
                metric_rows.append(skipped)
                skipped_patient_seeds.append(skipped)
                continue
            metric_rows.append(row)
            confusion_rows.extend(cm)
            prediction_parts.append(pred)
            artifact_rows.append({"scope": "single_patient", "patient_id": patient, "seed": seed, "path": artifact.name, "sha256": sha256_file(artifact)})
        if patient_index == 0 or (patient_index + 1) % 5 == 0:
            write_progress("single_patient_classifiers", patients_done=patient_index + 1, patients_total=len(patients))

    groups = {"high_separation": high_patients, "low_separation": low_patients}
    for trained_group, train_patients in groups.items():
        for seed in SEEDS:
            fit_group = pd.concat([patient_fit[(patient, seed)] for patient in train_patients], ignore_index=True)
            eval_own = pd.concat([patient_eval[(patient, seed)] for patient in train_patients], ignore_index=True)
            artifact = REPORT_DIR / f"source_classifier_group_{trained_group}_seed{seed}.joblib"
            row, cm, pred, bundle = fit_source_classifier(fit_group, eval_own, f"group_{trained_group}_own", seed, None, trained_group, artifact)
            row["trained_group"] = trained_group
            metric_rows.append(row)
            confusion_rows.extend(cm)
            prediction_parts.append(pred)
            artifact_rows.append({"scope": f"group_{trained_group}", "patient_id": None, "seed": seed, "path": artifact.name, "sha256": sha256_file(artifact)})
            opposite_group = "low_separation" if trained_group == "high_separation" else "high_separation"
            eval_patients = groups[opposite_group]
            eval_cross = pd.concat([patient_eval[(patient, seed)] for patient in eval_patients], ignore_index=True)
            # Re-center each opposite patient with that patient's early mean,
            # while retaining the trained group's scaler and classifier.
            # fit_source_classifier already stores a trained scaler/classifier;
            # apply the bundle to the pre-centered rows here.
            values = eval_cross.loc[:, FEATURES].to_numpy(float)
            scaler = bundle["scaler"]
            classifier = bundle["classifier"]
            probabilities = classifier.predict_proba(scaler.transform(values))
            y_eval = eval_cross.model.map(MODEL_TO_ID).astype(int).to_numpy()
            scope = f"group_{trained_group}_cross"
            row, cm = metrics_row(y_eval, probabilities, scope, seed, None, opposite_group, len(fit_group))
            row["trained_group"] = trained_group
            row["patients"] = int(eval_cross.patient_id.nunique())
            metric_rows.append(row)
            confusion_rows.extend(cm)
            pred = eval_cross[["patient_id", "recording", "event_index", "start_s", "end_s", "model", "fold", "seed"]].copy()
            pred["scope"] = scope
            pred["eval_group"] = opposite_group
            pred["true_model"] = pred.model.astype(str)
            pred["predicted_model"] = [MODELS[index] for index in np.argmax(probabilities, axis=1)]
            pred["predicted_probability"] = np.max(probabilities, axis=1)
            for index, model in enumerate(MODELS):
                pred[f"probability_{model}"] = probabilities[:, index]
            prediction_parts.append(pred.drop(columns=["model"]))

    metrics_df = pd.DataFrame(metric_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics_df.to_csv(REPORT_DIR / "source_classifier_metrics.csv", index=False)
    confusion_df.to_csv(REPORT_DIR / "source_classifier_confusion.csv", index=False)
    predictions.to_parquet(REPORT_DIR / "source_classifier_predictions.parquet", index=False)
    pd.DataFrame(artifact_rows).to_json(REPORT_DIR / "source_classifier_artifacts.json", orient="records", indent=2)

    patient_metric_rows: list[dict[str, object]] = []
    for (scope, seed, eval_group), group in predictions.groupby(["scope", "seed", "eval_group"], dropna=False):
        for patient, patient_rows in group.groupby("patient_id", sort=True):
            y_true = patient_rows.true_model.map(MODEL_TO_ID).astype(int).to_numpy()
            probs = patient_rows[[f"probability_{model}" for model in MODELS]].to_numpy(float)
            row, _ = metrics_row(y_true, probs, str(scope), int(seed), str(patient), str(eval_group), 0)
            row["patient_id"] = str(patient)
            patient_metric_rows.append(row)
    patient_metrics = pd.DataFrame(patient_metric_rows)
    patient_metrics.to_csv(REPORT_DIR / "source_classifier_patient_metrics.csv", index=False)
    bootstrap_rows: list[dict[str, object]] = []
    for scope in sorted(metrics_df.scope.astype(str).unique()):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"):
            bootstrap_rows.append(patient_bootstrap(patient_metrics, metric, scope, None))
            for group in ("high_separation", "low_separation"):
                bootstrap_rows.append(patient_bootstrap(patient_metrics, metric, scope, group))
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "source_classifier_patient_bootstrap.csv", index=False)
    method_summary = metrics_df.groupby(["scope", "eval_group"], dropna=False, as_index=False).agg(
        rows=("event_rows", "sum"), seeds=("seed", "nunique"), accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean"), macro_auroc=("macro_auroc", "mean")
    )
    method_summary.to_csv(REPORT_DIR / "source_classifier_method_summary.csv", index=False)

    manifest = {
        "analysis_id": RUN_ID,
        "status": "complete_posthoc_patient_stratified_h3_extension",
        "claim_boundary": "post-hoc exploratory TUSZ extension of H3; parent external detector outputs were frozen; this is not a detector-generalization claim",
        "question": "whether detector-source feature separability varies by TUSZ patient and by patient-separation stratum",
        "parent_h3_release": PARENT_RUN,
        "target": "detector source label: SingLEM, CBraMod, or TCN-GAT",
        "feature_set": {"name": "strict-EEG-10", "columns": list(FEATURES), "excluded": ["event_duration_s", "peak_probability"], "pca": False},
        "patients": len(patients),
        "skipped_patient_seed_runs": skipped_patient_seeds,
        "source_rows": int(len(features)),
        "checkpoint_units": sorted({f"fold{int(fold)}_seed{int(seed)}" for fold, seed in features[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None)}),
        "seeds": list(SEEDS),
        "within_patient_split": {"fit_fraction": FIT_FRACTION, "evaluation_fraction": 1.0 - FIT_FRACTION, "ordering": "recording path, event index, start/end time, model, fold; all frozen TUSZ event rows", "patient_centering": "early-fit patient mean applied to chronological evaluation"},
        "patient_separation": {"primary_score": "mean pairwise Euclidean distance between detector centroids after early-fit patient centering", "n_extreme_each": N_EXTREME, "high_patients": high_patients, "low_patients": low_patients, "selection_used_late_evaluation": False},
        "group_classifiers": "one pooled early-fit classifier per stratum and seed; own-stratum late evaluation plus cross-stratum descriptive application; no cross-stratum refitting",
        "classifier": "multinomial logistic regression C=1, class_weight=balanced, max_iter=500",
        "model_training": "source_classifier_only; no seizure-detector backbone retraining; no threshold selection",
        "chance_accuracy": 1.0 / len(MODELS),
        "chance_macro_auroc": 0.5,
        "bootstrap": {"unit": "patient", "replicates": N_BOOTSTRAP},
        "association_test": {"method": "patient x detector-source contingency chi-square and within-patient label-shuffle null", "permutation_replicates": N_ASSOCIATION_PERMUTATIONS, "fit_only": True},
        "source_hashes": {"feature_table": sha256_file(FEATURE_PATH), "feature_manifest": sha256_file(FEATURE_MANIFEST_PATH), "source_event_table": sha256_file(SOURCE_EVENT_PATH), "source_scoring_manifest": sha256_file(SOURCE_SCORING_MANIFEST)},
        "outputs": ["patient_detector_source_distribution.csv", "patient_detector_centroid_distances.csv", "patient_detector_separation_ranking.csv", "patient_detector_association_stats.json", "source_classifier_metrics.csv", "source_classifier_confusion.csv", "source_classifier_predictions.parquet", "source_classifier_patient_metrics.csv", "source_classifier_patient_bootstrap.csv", "source_classifier_method_summary.csv", "source_classifier_artifacts.json"],
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_progress("complete", status="complete", classifier_rows=len(metrics_df), high_patients=high_patients, low_patients=low_patients)


if __name__ == "__main__":
    main()
