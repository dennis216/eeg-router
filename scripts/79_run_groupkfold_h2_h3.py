"""Run train-fit/validation-select/test-once H2 and H3 analyses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
FEATURE_PATH = ROOT / "outputs" / "reports" / RUN_ID / os.environ.get("BFA_H2H3_FEATURE_INPUT", "h2h3_fp_features.parquet")
OUTPUT_PREFIX = os.environ.get("BFA_H2H3_OUTPUT_PREFIX", "")
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
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
K_CANDIDATES = (3, 4, 5, 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_fit_transform(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA, pd.Series, pd.Series]:
    train_values = train.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    other_values = other.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    other_values = other_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    other_values = other_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    train_scaled = scaler.transform(train_values)
    other_scaled = scaler.transform(other_values)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=20260821).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(other_scaled), scaler, pca, lower, upper


def _clean_matrix(frame: pd.DataFrame, median: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    values = frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).fillna(median)
    return values.clip(lower=lower, upper=upper, axis="columns")


def _select_k(train_points: np.ndarray, validation_points: np.ndarray, fold: int) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in K_CANDIDATES:
        if len(train_points) <= k or len(validation_points) <= k:
            continue
        clusterer = KMeans(n_clusters=k, n_init=20, random_state=20260821 + fold).fit(train_points)
        labels = clusterer.predict(validation_points)
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(validation_points, labels))
    if not scores:
        raise RuntimeError(f"cannot choose H2 cluster count on fold {fold}")
    return int(max(scores, key=scores.get)), scores


def _js_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) + 1e-12
    b = np.asarray(b, dtype=float) + 1e-12
    a /= a.sum()
    b /= b.sum()
    midpoint = 0.5 * (a + b)
    divergence = 0.5 * np.sum(a * np.log(a / midpoint)) + 0.5 * np.sum(b * np.log(b / midpoint))
    return float(1.0 - np.sqrt(max(divergence, 0.0)))


def _profiles(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    counts = frame.groupby(["patient", "model", "seed", "cluster"]).size().rename("events").reset_index()
    totals = counts.groupby(["patient", "model", "seed"], as_index=False)["events"].sum().rename(columns={"events": "total"})
    result = counts.merge(totals, on=["patient", "model", "seed"], how="left")
    result["proportion"] = result.events / result.total
    wide = result.pivot_table(index=["patient", "model", "seed"], columns="cluster", values="proportion", fill_value=0.0)
    wide = wide.reindex(columns=range(k), fill_value=0.0).reset_index()
    wide.columns = [str(column) if not isinstance(column, int) else f"cluster_{column}" for column in wide.columns]
    return wide


def _h2_profiles(frame: pd.DataFrame, fold: int, partition: str, k: int) -> list[dict[str, object]]:
    profiles = _profiles(frame, k)
    if profiles.empty:
        return []
    rows: list[dict[str, object]] = []
    for model in MODELS:
        subset = profiles[profiles.model == model]
        if subset.empty:
            continue
        vector = subset[[f"cluster_{i}" for i in range(k)]].mean().to_numpy(float)
        rows.append(
            {
                "fold": fold,
                "partition": partition,
                "model": model,
                "events": int(len(frame[frame.model == model])),
                "patients": int(subset.patient.nunique()),
                "seed_count": int(subset.seed.nunique()),
                "profile": json.dumps(vector.tolist()),
                **{f"cluster_{i}_proportion": float(vector[i]) for i in range(k)},
            }
        )
    for left_index, left_model in enumerate(MODELS):
        for right_model in MODELS[left_index + 1 :]:
            left = profiles[profiles.model == left_model]
            right = profiles[profiles.model == right_model]
            merged = left.merge(right, on=["patient", "seed"], suffixes=("_left", "_right"))
            similarities: list[float] = []
            for row in merged.itertuples(index=False):
                a = np.asarray([getattr(row, f"cluster_{i}_left") for i in range(k)])
                b = np.asarray([getattr(row, f"cluster_{i}_right") for i in range(k)])
                similarities.append(_js_similarity(a, b))
            if not merged.empty:
                left_mean = left[[f"cluster_{i}" for i in range(k)]].mean().to_numpy(float)
                right_mean = right[[f"cluster_{i}" for i in range(k)]].mean().to_numpy(float)
                rows.append(
                    {
                        "fold": fold,
                        "partition": partition,
                        "model": f"{left_model}__{right_model}",
                        "events": int(len(left) + len(right)),
                        "patients": int(merged.patient.nunique()),
                        "seed_count": int(merged.seed.nunique()),
                        "profile_similarity_js": _js_similarity(left_mean, right_mean),
                        "patient_median_similarity_js": float(np.median(similarities)),
                    }
                )
    return rows


def _fingerprint(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, *, fold: int, centered: bool
) -> dict[str, object]:
    train_frame = train.copy()
    validation_frame = validation.copy()
    test_frame = test.copy()
    if centered:
        for target in (train_frame, validation_frame, test_frame):
            target.loc[:, FEATURES] = target.loc[:, FEATURES] - target.groupby("patient")[list(FEATURES)].transform("mean")
    train_values = train_frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    validation_values = validation_frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    test_values = test_frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    validation_values = validation_values.fillna(median)
    test_values = test_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    validation_values = validation_values.clip(lower=lower, upper=upper, axis="columns")
    test_values = test_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    labels = {model: index for index, model in enumerate(MODELS)}
    y_train = train_frame.model.map(labels).to_numpy()
    y_validation = validation_frame.model.map(labels).to_numpy()
    y_test = test_frame.model.map(labels).to_numpy()
    if len(np.unique(y_train)) < len(MODELS):
        return {"fold": fold, "patient_centered": centered, "status": "insufficient_train_model_classes"}
    classifier = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=20260821 + fold)
    classifier.fit(scaler.transform(train_values), y_train)
    rows: dict[str, object] = {
        "fold": fold,
        "patient_centered": centered,
        "train_events": int(len(train_frame)),
        "validation_events": int(len(validation_frame)),
        "test_events": int(len(test_frame)),
        "classifier": "multinomial_logistic_C1_class_weight_balanced",
        "fit_partition": "outer_train_discovery",
        "test_used_for_selection": False,
        "chance_macro_auroc": 1.0 / len(MODELS),
    }
    for partition, values, target in (
        ("validation", validation_values, y_validation),
        ("test", test_values, y_test),
    ):
        if len(target) == 0:
            continue
        probabilities = classifier.predict_proba(scaler.transform(values))
        predictions = classifier.predict(scaler.transform(values))
        rows[f"{partition}_accuracy"] = float(accuracy_score(target, predictions))
        rows[f"{partition}_balanced_accuracy"] = float(balanced_accuracy_score(target, predictions))
        try:
            rows[f"{partition}_macro_auroc"] = float(
                roc_auc_score(target, probabilities, multi_class="ovr", average="macro", labels=list(range(len(MODELS))))
            )
        except ValueError:
            rows[f"{partition}_macro_auroc"] = float("nan")
    return rows


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    assignments: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    fingerprints: list[dict[str, object]] = []
    for fold in FOLDS:
        fold_frame = features[features.split == fold].copy()
        train = fold_frame[fold_frame.partition == "train"].copy()
        validation = fold_frame[fold_frame.partition == "validation"].copy()
        test = fold_frame[fold_frame.partition == "test"].copy()
        if train.empty or validation.empty or test.empty:
            raise RuntimeError(f"empty H2/H3 partition on fold {fold}")
        train_points, validation_points, scaler, pca, lower, upper = _clean_fit_transform(train, validation)
        test_matrix = _clean_matrix(test, train.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).median().fillna(0.0), lower, upper)
        test_points = pca.transform(scaler.transform(test_matrix))
        k, scores = _select_k(train_points, validation_points, fold)
        clusterer = KMeans(n_clusters=k, n_init=50, random_state=20260821 + fold).fit(train_points)
        for target, points, partition in ((train, train_points, "train"), (validation, validation_points, "validation"), (test, test_points, "test")):
            assigned = target.copy()
            assigned["cluster"] = clusterer.predict(points)
            assigned["pca_components"] = int(pca.n_components_)
            assigned["cluster_fit_partition"] = "outer_train_discovery"
            assigned["cluster_selection_partition"] = "validation"
            assignments.append(assigned)
            profiles.extend(_h2_profiles(assigned, fold, partition, k))
        selections.append(
            {
                "fold": fold,
                "selected_k": k,
                "pca_components": int(pca.n_components_),
                "validation_silhouette_selected": float(scores[k]),
                "silhouette_scores": json.dumps(scores, sort_keys=True),
                "scaler_fit_partition": "outer_train_discovery",
                "cluster_fit_partition": "outer_train_discovery",
                "cluster_selection_partition": "validation",
                "test_assignment_only": True,
            }
        )
        for centered in (False, True):
            fingerprints.append(_fingerprint(train, validation, test, fold=fold, centered=centered))
        print(f"H2H3_ANALYSIS_DONE fold={fold} k={k} train={len(train)} validation={len(validation)} test={len(test)}", flush=True)

    assignments_frame = pd.concat(assignments, ignore_index=True)
    assignments_frame.to_parquet(REPORT_DIR / f"{OUTPUT_PREFIX}h2_groupkfold_cluster_assignments.parquet", index=False)
    pd.DataFrame(selections).to_csv(REPORT_DIR / f"{OUTPUT_PREFIX}h2_groupkfold_cluster_selection.csv", index=False)
    pd.DataFrame(profiles).to_csv(REPORT_DIR / f"{OUTPUT_PREFIX}h2_groupkfold_profiles.csv", index=False)
    pd.DataFrame(fingerprints).to_csv(REPORT_DIR / f"{OUTPUT_PREFIX}h3_groupkfold_model_fingerprint.csv", index=False)
    summary = {
        "release_id": RUN_ID,
        "h2": {
            "status": "preliminary_confirmatory_layer",
            "fit_partition": "outer_train_discovery",
            "cluster_selection_partition": "validation",
            "test_assignment_only": True,
            "cluster_candidates": list(K_CANDIDATES),
            "features": list(FEATURES),
        },
        "h3": {
            "status": "preliminary_confirmatory_layer",
            "fit_partition": "outer_train_discovery",
            "test_used_for_selection": False,
            "patient_centering_sensitivity": True,
            "classifier": "multinomial_logistic_C1_class_weight_balanced",
        },
        "source_feature_sha256": _sha256(FEATURE_PATH),
        "outputs": [
            f"{OUTPUT_PREFIX}h2_groupkfold_cluster_assignments.parquet",
            f"{OUTPUT_PREFIX}h2_groupkfold_cluster_selection.csv",
            f"{OUTPUT_PREFIX}h2_groupkfold_profiles.csv",
            f"{OUTPUT_PREFIX}h3_groupkfold_model_fingerprint.csv",
        ],
    }
    (REPORT_DIR / f"{OUTPUT_PREFIX}h2_h3_groupkfold_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(pd.DataFrame(selections).to_string(index=False))
    print(pd.DataFrame(fingerprints).to_string(index=False))


if __name__ == "__main__":
    main()
