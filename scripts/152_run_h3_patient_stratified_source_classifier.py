"""Patient-stratified extension of the H3 detector-source classifier.

The word ``router`` is intentionally not used in this script.  The target is
the source detector of an already observed false-positive event:

    SingLEM | CBraMod | TCN-GAT

The classifier is therefore a three-class detector-source classifier, not a
model-selection or sensitivity/FA router.  The analysis uses a frozen EEG/QC
feature set (by default EEG/QC-only 13; a strict-EEG-10 sensitivity variant
can be selected with ``BFA_FEATURE_SET=strict-EEG-10``).  No peak probability
or event duration is used.

For every patient and model seed, frozen outer-test FP events are ordered by
recording chronology.  The first 70% are a within-patient fit portion and the
last 30% are a chronological evaluation portion.  Patient centering is fit on
the early portion and applied to the later portion; a global StandardScaler
and multinomial logistic classifier are then fit on the early data only.
Patients are ranked using only early-portion pairwise detector centroid
separation.  The top and bottom five patients form frozen high- and
low-separation groups.  Group classifiers are fit on early events from one
group and evaluated on late events from that group and, descriptively, on the
opposite group.

This is a post-hoc extension of the parent H3 release because the parent
outer-test results were already inspected.  It is not a new confirmatory
claim and does not change the frozen H3 result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
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
RUN_ID = os.environ.get("BFA_RUN_ID", "h3-patient-stratified-source-classifier-v1")
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
PARENT_RUN = "v3-groupkfold-confirmatory-v1"
FEATURE_PATH = ROOT / "outputs" / "reports" / PARENT_RUN / "h2h3_fp_features.parquet"
FEATURE_MANIFEST_PATH = ROOT / "outputs" / "reports" / PARENT_RUN / "h2h3_fp_features_manifest.json"
H6_ORDER_PATH = ROOT / "outputs" / "reports" / "h6-supervised-personalized-router-v1" / "chronological_split_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
MODEL_TO_ID = {model: index for index, model in enumerate(MODELS)}
SEEDS = (17, 42, 3407)
FOLDS = tuple(range(5))
FEATURE_SET_NAME = os.environ.get("BFA_FEATURE_SET", "EEG/QC-only-13")
FEATURE_SET_COLUMNS = {
    "EEG/QC-only-13": (
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
    ),
    # Strict EEG-only sensitivity set: 2 RMS + 2 spatial + 5 band-power +
    # 1 high/low ratio, as specified for this extension.
    "strict-EEG-10": (
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
if FEATURE_SET_NAME not in FEATURE_SET_COLUMNS:
    raise ValueError(f"Unknown BFA_FEATURE_SET={FEATURE_SET_NAME!r}; choose one of {sorted(FEATURE_SET_COLUMNS)}")
FEATURES = FEATURE_SET_COLUMNS[FEATURE_SET_NAME]
N_EXTREME = 5
N_BOOTSTRAP = 2000
N_ASSOCIATION_PERMUTATIONS = 1000
RNG_SEED = 20260828
FIT_FRACTION = 0.70


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


def chronological_order_map(order_manifest: dict[str, object]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in order_manifest["patients"]:
        patient = str(row["patient"])
        ordered = [str(x) for x in row["calibration_recordings"]] + [str(x) for x in row["holdout_recordings"]]
        result[patient] = {recording: index for index, recording in enumerate(ordered)}
    return result


def center_by_patient_fit(
    fit: pd.DataFrame,
    other: pd.DataFrame,
    means: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Fit patient centering on ``fit`` and apply the same means to ``other``."""

    fit_out = fit.copy()
    other_out = other.copy()
    fit_values = fit_out.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    global_mean = fit_values.mean().fillna(0.0).to_numpy(float)
    if means is None:
        means = {}
        for patient, group in fit_out.groupby("patient", sort=False):
            values = group.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).mean().fillna(pd.Series(global_mean, index=FEATURES)).to_numpy(float)
            means[str(patient)] = values
    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = np.array(result.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float), dtype=float, copy=True)
        for index, patient in enumerate(result.patient.astype(str)):
            matrix[index] = np.nan_to_num(matrix[index], nan=global_mean) - means.get(str(patient), global_mean)
        result.loc[:, FEATURES] = matrix
        return result
    return transform(fit_out), transform(other_out), means


def source_distribution(rows: pd.DataFrame, patient: str, seed: int, partition: str) -> list[dict[str, object]]:
    counts = rows.model.astype(str).value_counts().to_dict()
    total = int(len(rows))
    entropy = 0.0
    proportions: dict[str, float] = {}
    for model in MODELS:
        proportion = float(counts.get(model, 0) / total) if total else 0.0
        proportions[model] = proportion
        if proportion > 0:
            entropy -= proportion * math.log(proportion)
    entropy /= math.log(len(MODELS)) if len(MODELS) > 1 else 1.0
    return [
        {
            "patient": patient,
            "seed": seed,
            "partition": partition,
            "event_rows": total,
            "source_entropy_normalized": entropy,
            "source_count": int(counts.get(model, 0)),
            "source_proportion": proportions[model],
            "model": model,
        }
        for model in MODELS
    ]


def centroid_separation(centered_fit: pd.DataFrame, patient: str, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    subset = centered_fit[centered_fit.patient.astype(str) == str(patient)]
    centroids: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for model in MODELS:
        model_rows = subset[subset.model.astype(str) == model]
        counts[model] = int(len(model_rows))
        if model_rows.empty:
            continue
        centroids[model] = model_rows.loc[:, FEATURES].to_numpy(float).mean(axis=0)
    distances: list[dict[str, object]] = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            if left not in centroids or right not in centroids:
                distance = math.nan
            else:
                distance = float(np.linalg.norm(centroids[left] - centroids[right]))
            distances.append({"patient": patient, "seed": seed, "left_model": left, "right_model": right, "centroid_distance": distance})
    finite = [float(row["centroid_distance"]) for row in distances if math.isfinite(float(row["centroid_distance"]))]
    summary = {
        "patient": patient,
        "seed": seed,
        "fit_event_rows": int(len(subset)),
        "mean_pairwise_centroid_distance": float(np.mean(finite)) if finite else math.nan,
        "min_pairwise_centroid_distance": float(np.min(finite)) if finite else math.nan,
        "max_pairwise_centroid_distance": float(np.max(finite)) if finite else math.nan,
        "singlem_fit_events": counts["singlem"],
        "cbramod_fit_events": counts["cbramod"],
        "tcn_gat_fit_events": counts["tcn_gat"],
    }
    return summary, distances


def classifier_metrics(y_true: np.ndarray, probabilities: np.ndarray, scope: str, seed: int, patient: str | None, eval_group: str | None) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = np.argmax(probabilities, axis=1)
    labels = list(range(len(MODELS)))
    row: dict[str, object] = {
        "scope": scope,
        "seed": seed,
        "patient": patient,
        "eval_group": eval_group,
        "event_rows": int(len(y_true)),
        "source_class_count": int(len(np.unique(y_true))),
        "accuracy": float(accuracy_score(y_true, predictions)) if len(y_true) else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)) if len(y_true) else math.nan,
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)) if len(y_true) else math.nan,
        "chance_accuracy": 1.0 / len(MODELS),
        "chance_macro_auroc": 1.0 / len(MODELS),
    }
    try:
        row["macro_auroc"] = float(roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        row["macro_auroc"] = math.nan
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    confusion_rows = [
        {
            "scope": scope,
            "seed": seed,
            "patient": patient,
            "eval_group": eval_group,
            "true_model": MODELS[true_id],
            "predicted_model": MODELS[pred_id],
            "count": int(matrix[true_id, pred_id]),
        }
        for true_id in labels
        for pred_id in labels
    ]
    return row, confusion_rows


def fit_source_classifier(
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    scope: str,
    seed: int,
    patient: str | None,
    eval_group: str | None,
    center_means: dict[str, np.ndarray],
    artifact_path: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], object | None]:
    if fit_frame.empty or eval_frame.empty:
        return ({"scope": scope, "seed": seed, "patient": patient, "eval_group": eval_group, "status": "empty_fit_or_eval"}, [], [], None)
    fit_centered, eval_centered, _ = center_by_patient_fit(fit_frame, eval_frame, means=center_means)
    scaler = StandardScaler().fit(fit_centered.loc[:, FEATURES].to_numpy(float))
    classifier = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=RNG_SEED + seed)
    y_fit = fit_centered.model.map(MODEL_TO_ID).astype(int).to_numpy()
    y_eval = eval_centered.model.map(MODEL_TO_ID).astype(int).to_numpy()
    if len(np.unique(y_fit)) < len(MODELS):
        return ({"scope": scope, "seed": seed, "patient": patient, "eval_group": eval_group, "status": "missing_fit_source_class", "fit_event_rows": int(len(fit_frame)), "eval_event_rows": int(len(eval_frame))}, [], [], None)
    classifier.fit(scaler.transform(fit_centered.loc[:, FEATURES].to_numpy(float)), y_fit)
    probabilities = classifier.predict_proba(scaler.transform(eval_centered.loc[:, FEATURES].to_numpy(float)))
    metrics, confusion_rows = classifier_metrics(y_eval, probabilities, scope, seed, patient, eval_group)
    metrics.update({"status": "complete", "fit_event_rows": int(len(fit_frame)), "fit_source_classes": int(len(np.unique(y_fit)))})
    prediction_rows = eval_centered[["patient", "recording", "start_s", "end_s", "model"]].copy()
    prediction_rows["scope"] = scope
    prediction_rows["seed"] = seed
    prediction_rows["eval_group"] = eval_group
    prediction_rows["true_model"] = prediction_rows.model.astype(str)
    prediction_rows["predicted_model"] = [MODELS[index] for index in np.argmax(probabilities, axis=1)]
    prediction_rows["predicted_probability"] = np.max(probabilities, axis=1)
    for index, model in enumerate(MODELS):
        prediction_rows[f"probability_{model}"] = probabilities[:, index]
    prediction_rows = prediction_rows.drop(columns=["model"])
    if artifact_path is not None:
        import joblib
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "classifier": classifier, "features": list(FEATURES), "center_means": center_means, "scope": scope, "seed": seed}, artifact_path)
    return metrics, confusion_rows, prediction_rows.to_dict(orient="records"), classifier


def bootstrap_metric(rows: pd.DataFrame, metric: str, scope: str, group: str | None = None) -> dict[str, object]:
    subset = rows[rows.scope == scope].copy()
    if group is not None:
        subset = subset[subset.eval_group == group]
    if subset.empty:
        return {"scope": scope, "eval_group": group, "metric": metric, "status": "empty"}
    if "patient" in subset.columns and subset.patient.notna().any():
        per_patient = subset.groupby("patient", as_index=False)[metric].mean()
    else:
        per_patient = subset.groupby("seed", as_index=False)[metric].mean()
    values = per_patient[metric].astype(float).to_numpy()
    rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in scope) + (0 if group is None else sum(ord(c) for c in group)))
    samples = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        samples[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    return {
        "scope": scope,
        "eval_group": group,
        "metric": metric,
        "unit_count": int(len(values)),
        "bootstrap_replicates": N_BOOTSTRAP,
        "mean": float(np.mean(values)),
        "q025": float(np.quantile(samples, 0.025)),
        "q975": float(np.quantile(samples, 0.975)),
        "status": "complete",
    }


def association_test(distribution: pd.DataFrame) -> dict[str, object]:
    """Patient/source association using counts, with a small label-shuffle null."""

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED + 101)
    for seed in SEEDS:
        subset = distribution[(distribution.partition == "fit") & (distribution.seed == seed)]
        table = subset.pivot_table(index="patient", columns="model", values="source_count", aggfunc="sum", fill_value=0).reindex(columns=MODELS, fill_value=0)
        observed = table.to_numpy(float)
        chi2, p_value, _, expected = chi2_contingency(observed, correction=False)
        n = int(observed.sum())
        min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
        cramers_v = float(math.sqrt(chi2 / n / min_dim)) if n and min_dim > 0 else math.nan
        # Shuffle source labels across events while preserving each patient's
        # event-row count and the global source totals for this seed.
        event_labels = np.concatenate([np.full(int(observed[i, j]), j, dtype=np.int8) for i in range(observed.shape[0]) for j in range(observed.shape[1])])
        patient_ids = np.concatenate([np.full(int(observed[i].sum()), i, dtype=np.int32) for i in range(observed.shape[0])])
        null_chi = np.empty(N_ASSOCIATION_PERMUTATIONS)
        for permutation in range(N_ASSOCIATION_PERMUTATIONS):
            shuffled = rng.permutation(event_labels)
            counts = np.zeros_like(observed)
            np.add.at(counts, (patient_ids, shuffled), 1)
            expected_null = counts.sum(axis=1, keepdims=True) * counts.sum(axis=0, keepdims=True) / max(1, counts.sum())
            null_chi[permutation] = float(np.sum((counts - expected_null) ** 2 / np.where(expected_null > 0, expected_null, 1.0)))
        permutation_p = float((1 + np.sum(null_chi >= chi2)) / (1 + N_ASSOCIATION_PERMUTATIONS))
        rows.append({"seed": seed, "patients": int(table.shape[0]), "events": n, "chi2": float(chi2), "chi2_p_value": float(p_value), "cramers_v": cramers_v, "permutation_replicates": N_ASSOCIATION_PERMUTATIONS, "permutation_p_value": permutation_p, "null_chi2_q975": float(np.quantile(null_chi, 0.975))})
    # Patient-balanced source-share dispersion is less dominated by very long
    # recordings than the count-based chi-square statistic.
    proportions = distribution[distribution.partition == "fit"].pivot_table(index=["patient", "seed"], columns="model", values="source_proportion", fill_value=0.0).reindex(columns=MODELS, fill_value=0.0)
    variance = proportions.var(axis=0, ddof=1).to_dict()
    rows.append({"seed": "all", "patients": int(proportions.index.get_level_values(0).nunique()), "events": int(distribution[distribution.partition == "fit"].event_rows.sum()), "chi2": math.nan, "chi2_p_value": math.nan, "cramers_v": math.nan, "permutation_replicates": N_ASSOCIATION_PERMUTATIONS, "permutation_p_value": math.nan, "null_chi2_q975": math.nan, "patient_balanced_source_proportion_variance": json.dumps({k: float(v) for k, v in variance.items()})})
    return {"tests": rows, "interpretation": "A patient/source association shows that detector-origin event prevalence varies by patient; it does not imply that a detector is clinically better for that patient."}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_progress("loading", status="started")
    for path in (FEATURE_PATH, FEATURE_MANIFEST_PATH, H6_ORDER_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    features = pd.read_parquet(FEATURE_PATH)
    features = features[features.partition.astype(str) == "test"].copy()
    features["patient"] = features.patient.astype(str)
    features["recording"] = features.recording.astype(str)
    features["model"] = features.model.astype(str)
    features["seed"] = features.seed.astype(int)
    features["fold"] = features.split.astype(int)
    features = features.dropna(subset=list(FEATURES))
    order_map = chronological_order_map(json.loads(H6_ORDER_PATH.read_text()))
    patients = sorted(features.patient.unique())
    if set(patients) != set(order_map):
        raise RuntimeError("H3 test patients and frozen chronological order manifest differ")
    write_progress("loaded", patients=len(patients), feature_rows=len(features), feature_set=FEATURE_SET_NAME, feature_count=len(FEATURES))

    distribution_rows: list[dict[str, object]] = []
    separation_rows: list[dict[str, object]] = []
    pair_distance_rows: list[dict[str, object]] = []
    patient_fit_frames: dict[tuple[str, int], pd.DataFrame] = {}
    patient_eval_frames: dict[tuple[str, int], pd.DataFrame] = {}
    patient_center_means: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for index, patient in enumerate(patients):
        for seed in SEEDS:
            rows = features[(features.patient == patient) & (features.seed == seed)].copy()
            rows["recording_order"] = rows.recording.map(order_map[patient]).fillna(999999)
            rows = rows.sort_values(["recording_order", "start_s", "end_s", "model"], kind="stable")
            cut = max(1, min(len(rows) - 1, int(math.floor(FIT_FRACTION * len(rows)))))
            fit = rows.iloc[:cut].copy()
            evaluation = rows.iloc[cut:].copy()
            patient_fit_frames[(patient, seed)] = fit
            patient_eval_frames[(patient, seed)] = evaluation
            fit_centered, _, means = center_by_patient_fit(fit, fit, means=None)
            patient_center_means[(patient, seed)] = means
            distribution_rows.extend(source_distribution(fit, patient, seed, "fit"))
            distribution_rows.extend(source_distribution(evaluation, patient, seed, "chronological_eval"))
            summary, distances = centroid_separation(fit_centered, patient, seed)
            separation_rows.append(summary)
            pair_distance_rows.extend(distances)
        if index == 0 or (index + 1) % 5 == 0:
            write_progress("patient_separation", patients_done=index + 1, patients_total=len(patients))
    distribution_df = pd.DataFrame(distribution_rows)
    separation_df = pd.DataFrame(separation_rows)
    distances_df = pd.DataFrame(pair_distance_rows)
    distribution_df.to_csv(REPORT_DIR / "patient_detector_source_distribution.csv", index=False)
    distances_df.to_csv(REPORT_DIR / "patient_detector_centroid_distances.csv", index=False)
    ranking = separation_df.groupby("patient", as_index=False).agg(
        mean_pairwise_centroid_distance=("mean_pairwise_centroid_distance", "mean"),
        sd_pairwise_centroid_distance=("mean_pairwise_centroid_distance", "std"),
        min_pairwise_centroid_distance=("min_pairwise_centroid_distance", "mean"),
        max_pairwise_centroid_distance=("max_pairwise_centroid_distance", "mean"),
        fit_event_rows=("fit_event_rows", "mean"),
    ).sort_values(["mean_pairwise_centroid_distance", "patient"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranking["rank_high_to_low"] = np.arange(1, len(ranking) + 1)
    ranking["stratum"] = "middle"
    ranking.loc[: N_EXTREME - 1, "stratum"] = "high_separation"
    ranking.loc[len(ranking) - N_EXTREME :, "stratum"] = "low_separation"
    ranking.to_csv(REPORT_DIR / "patient_detector_separation_ranking.csv", index=False)
    high_patients = ranking.loc[ranking.stratum == "high_separation", "patient"].tolist()
    low_patients = ranking.loc[ranking.stratum == "low_separation", "patient"].tolist()
    stratum_by_patient = ranking.set_index("patient").stratum.to_dict()
    write_progress("strata_frozen", high_patients=high_patients, low_patients=low_patients)

    association = association_test(distribution_df)
    (REPORT_DIR / "patient_detector_association_stats.json").write_text(json.dumps(association, indent=2, sort_keys=True, default=float) + "\n")

    classifier_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    # Single-patient classifiers: every patient and seed is retained.  They
    # are fit on the early 70% and evaluated on the later 30%.
    for index, patient in enumerate(patients):
        for seed in SEEDS:
            fit = patient_fit_frames[(patient, seed)]
            evaluation = patient_eval_frames[(patient, seed)]
            artifact = REPORT_DIR / f"source_classifier_patient_{patient}_seed{seed}.joblib"
            metrics, cm, preds, _ = fit_source_classifier(
                fit,
                evaluation,
                scope="single_patient",
                seed=seed,
                patient=patient,
                eval_group=stratum_by_patient[patient],
                center_means=patient_center_means[(patient, seed)],
                artifact_path=artifact,
            )
            classifier_rows.append(metrics)
            confusion_rows.extend(cm)
            prediction_rows.extend(preds)
            if artifact.exists():
                artifact_rows.append({"scope": "single_patient", "patient": patient, "seed": seed, "path": artifact.name, "sha256": sha256_file(artifact)})
        if index == 0 or (index + 1) % 5 == 0:
            write_progress("single_patient_classifiers", patients_done=index + 1, patients_total=len(patients))

    # Pooled group classifiers, one fit per group and seed.  Each classifier
    # is evaluated on both its own late portion and the opposite group's late
    # portion; the cross application is descriptive, with no refitting.
    for trained_group, train_patients in (("high_separation", high_patients), ("low_separation", low_patients)):
        for seed in SEEDS:
            fit_group = pd.concat([patient_fit_frames[(patient, seed)] for patient in train_patients], ignore_index=True)
            eval_patients = {"high_separation": high_patients, "low_separation": low_patients}
            artifact = REPORT_DIR / f"source_classifier_group_{trained_group}_seed{seed}.joblib"
            first_eval = pd.concat([patient_eval_frames[(patient, seed)] for patient in eval_patients[trained_group]], ignore_index=True)
            metrics, cm, preds, _ = fit_source_classifier(
                fit_group,
                first_eval,
                scope=f"group_{trained_group}_own",
                seed=seed,
                patient=None,
                eval_group=trained_group,
                center_means={patient: patient_center_means[(patient, seed)][patient] for patient in train_patients},
                artifact_path=artifact,
            )
            classifier_rows.append(metrics)
            confusion_rows.extend(cm)
            prediction_rows.extend(preds)
            if artifact.exists():
                artifact_rows.append({"scope": f"group_{trained_group}", "patient": None, "seed": seed, "path": artifact.name, "sha256": sha256_file(artifact)})
            # Reload the saved classifier bundle to cross-apply while using
            # each opposite patient's own early centering mean.
            import joblib
            bundle = joblib.load(artifact) if artifact.exists() else None
            for eval_group, eval_group_patients in eval_patients.items():
                if eval_group == trained_group or bundle is None:
                    continue
                eval_frame = pd.concat([patient_eval_frames[(patient, seed)] for patient in eval_group_patients], ignore_index=True)
                fit_eval, centered_eval, _ = center_by_patient_fit(fit_group, eval_frame, means={patient: patient_center_means[(patient, seed)][patient] for patient in train_patients + eval_group_patients})
                scaler = bundle["scaler"]
                classifier = bundle["classifier"]
                probabilities = classifier.predict_proba(scaler.transform(centered_eval.loc[:, FEATURES].to_numpy(float)))
                y_eval = centered_eval.model.map(MODEL_TO_ID).astype(int).to_numpy()
                cross_scope = f"group_{trained_group}_cross"
                metrics, cm = classifier_metrics(y_eval, probabilities, cross_scope, seed, None, eval_group)
                metrics.update({"status": "complete", "fit_event_rows": int(len(fit_group)), "fit_source_classes": len(MODELS), "trained_group": trained_group})
                classifier_rows.append(metrics)
                confusion_rows.extend(cm)
                pred = centered_eval[["patient", "recording", "start_s", "end_s", "model"]].copy()
                pred["scope"] = cross_scope
                pred["seed"] = seed
                pred["eval_group"] = eval_group
                pred["true_model"] = pred.model.astype(str)
                pred["predicted_model"] = [MODELS[index] for index in np.argmax(probabilities, axis=1)]
                pred["predicted_probability"] = np.max(probabilities, axis=1)
                for model_index, model in enumerate(MODELS):
                    pred[f"probability_{model}"] = probabilities[:, model_index]
                prediction_rows.extend(pred.drop(columns=["model"]).to_dict(orient="records"))

    classifier_df = pd.DataFrame(classifier_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    classifier_df["stratum"] = classifier_df.patient.map(stratum_by_patient) if "patient" in classifier_df else None
    classifier_df.to_csv(REPORT_DIR / "source_classifier_metrics.csv", index=False)
    confusion_df.to_csv(REPORT_DIR / "source_classifier_confusion.csv", index=False)
    prediction_df.to_parquet(REPORT_DIR / "source_classifier_predictions.parquet", index=False)
    pd.DataFrame(artifact_rows).to_json(REPORT_DIR / "source_classifier_artifacts.json", orient="records", indent=2)

    # Patient-level group metrics for confidence intervals.  The raw event
    # predictions are used only to calculate descriptive per-patient metrics;
    # no later holdout choice is made from these values.
    patient_metric_rows: list[dict[str, object]] = []
    for (scope, seed, eval_group), group in prediction_df.groupby(["scope", "seed", "eval_group"], dropna=False):
        for patient, patient_rows in group.groupby("patient", sort=True):
            y_true = patient_rows.true_model.map(MODEL_TO_ID).astype(int).to_numpy()
            probs = patient_rows[[f"probability_{model}" for model in MODELS]].to_numpy(float)
            metrics, _ = classifier_metrics(y_true, probs, str(scope), int(seed), str(patient), str(eval_group) if pd.notna(eval_group) else None)
            patient_metric_rows.append(metrics)
    patient_metrics_df = pd.DataFrame(patient_metric_rows)
    patient_metrics_df.to_csv(REPORT_DIR / "source_classifier_patient_metrics.csv", index=False)
    bootstrap_rows: list[dict[str, object]] = []
    for scope in sorted(classifier_df.scope.dropna().unique()):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"):
            bootstrap_rows.append(bootstrap_metric(patient_metrics_df, metric, scope))
            for group in ("high_separation", "low_separation"):
                bootstrap_rows.append(bootstrap_metric(patient_metrics_df, metric, scope, group))
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "source_classifier_patient_bootstrap.csv", index=False)

    method_summary = classifier_df.groupby(["scope", "eval_group"], dropna=False, as_index=False).agg(
        rows=("event_rows", "sum"), seeds=("seed", "nunique"), accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean"), macro_auroc=("macro_auroc", "mean")
    )
    method_summary.to_csv(REPORT_DIR / "source_classifier_method_summary.csv", index=False)

    manifest = {
        "analysis_id": RUN_ID,
        "status": "complete_posthoc_patient_stratified_h3_extension",
        "claim_boundary": "post-hoc exploratory extension of H3; parent outer-test results were previously inspected; not a new confirmatory result and not patient-independent generalization",
        "parent_h3_release": PARENT_RUN,
        "question": "whether detector-source feature separability varies by patient and by patient-separation stratum",
        "target": "detector source label: SingLEM, CBraMod, or TCN-GAT",
        "features": {"name": FEATURE_SET_NAME, "columns": list(FEATURES), "excluded": ["peak_probability", "event_duration_s"], "pca": False},
        "patients": len(patients),
        "seeds": list(SEEDS),
        "outer_test_source": "frozen v3-groupkfold-confirmatory-v1 test events; each patient is used once as an outer-test patient",
        "within_patient_split": {"fit_fraction": FIT_FRACTION, "evaluation_fraction": 1.0 - FIT_FRACTION, "ordering": "frozen H6 recording chronology then event start/end", "patient_centering": "means fit on early portion and applied to later portion"},
        "classifier": "multinomial logistic regression C=1, class_weight=balanced, max_iter=500; chance accuracy and macro-AUROC = 1/3",
        "patient_separation": {"primary_score": "mean pairwise Euclidean distance between detector centroids after fit-only patient centering", "n_extreme_each": N_EXTREME, "high_patients": high_patients, "low_patients": low_patients, "selection_used_late_evaluation": False},
        "group_classifiers": "one pooled early-fit classifier per stratum and seed; own-stratum late evaluation plus cross-stratum descriptive application; no cross-stratum refitting",
        "source_hashes": {"feature_table": sha256_file(FEATURE_PATH), "feature_manifest": sha256_file(FEATURE_MANIFEST_PATH), "chronology_manifest": sha256_file(H6_ORDER_PATH)},
        "association_test": {"method": "patient x detector-source contingency chi-square and label-shuffle null", "permutation_replicates": N_ASSOCIATION_PERMUTATIONS},
        "outputs": {"patient_detector_source_distribution": "patient_detector_source_distribution.csv", "patient_detector_association": "patient_detector_association_stats.json", "patient_detector_centroid_distances": "patient_detector_centroid_distances.csv", "patient_detector_separation_ranking": "patient_detector_separation_ranking.csv", "source_classifier_metrics": "source_classifier_metrics.csv", "source_classifier_patient_metrics": "source_classifier_patient_metrics.csv", "source_classifier_confusion": "source_classifier_confusion.csv", "source_classifier_predictions": "source_classifier_predictions.parquet", "source_classifier_patient_bootstrap": "source_classifier_patient_bootstrap.csv", "source_classifier_method_summary": "source_classifier_method_summary.csv", "source_classifier_artifacts": "source_classifier_artifacts.json"},
        "model_training": "source_classifier_only; no seizure-detector backbone retraining; no threshold selection",
    }
    (REPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n")
    write_progress("complete", status="complete", classifier_rows=len(classifier_df), high_patients=high_patients, low_patients=low_patients)


if __name__ == "__main__":
    main()
