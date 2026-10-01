"""Run frozen validation-fit H2 phenotype and H3 fingerprint analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "outputs" / "events" / "v3-compact-extension-v1_fp_features.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
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
SEED = 20260811


def _fit_transform(train: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    lower = train.loc[:, FEATURES].quantile(0.01)
    upper = train.loc[:, FEATURES].quantile(0.99)
    train_clipped = train.loc[:, FEATURES].clip(lower=lower, upper=upper, axis="columns")
    other_clipped = other.loc[:, FEATURES].clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_clipped)
    train_scaled = scaler.transform(train_clipped)
    other_scaled = scaler.transform(other_clipped)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=SEED).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(other_scaled), scaler, pca


def _choose_k(points: np.ndarray) -> tuple[int, dict[int, float]]:
    rng = np.random.default_rng(SEED)
    sample = points
    if len(points) > 12000:
        sample = points[rng.choice(len(points), size=12000, replace=False)]
    scores: dict[int, float] = {}
    for k in K_CANDIDATES:
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels))
    chosen = max(scores, key=scores.get)
    return int(chosen), scores


def _patient_cluster_profiles(frame: pd.DataFrame, cluster_col: str, k: int) -> pd.DataFrame:
    counts = frame.groupby(["patient", "model", "seed", cluster_col]).size().rename("events").reset_index()
    totals = counts.groupby(["patient", "model", "seed"], as_index=False)["events"].sum().rename(columns={"events": "total"})
    result = counts.merge(totals, on=["patient", "model", "seed"], how="left")
    result["proportion"] = result.events / result.total
    wide = result.pivot_table(index=["patient", "model", "seed"], columns=cluster_col, values="proportion", fill_value=0.0)
    wide = wide.reindex(columns=range(k), fill_value=0.0).reset_index()
    wide.columns = [str(column) if not isinstance(column, int) else f"cluster_{column}" for column in wide.columns]
    return wide


def _js_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) + 1e-12
    b = np.asarray(b, dtype=float) + 1e-12
    a /= a.sum()
    b /= b.sum()
    midpoint = 0.5 * (a + b)
    divergence = 0.5 * np.sum(a * np.log(a / midpoint)) + 0.5 * np.sum(b * np.log(b / midpoint))
    return float(1.0 - np.sqrt(max(divergence, 0.0)))


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    cluster_rows: list[pd.DataFrame] = []
    profile_rows: list[dict[str, object]] = []
    fingerprint_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for split in sorted(features.split.unique()):
        validation = features[(features.split == split) & (features.partition == "validation")].copy()
        test = features[(features.split == split) & (features.partition == "test")].copy()
        train_points, test_points, scaler, pca = _fit_transform(validation, test)
        k, silhouette_scores = _choose_k(train_points)
        clusterer = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(train_points)
        validation["cluster"] = clusterer.labels_
        test["cluster"] = clusterer.predict(test_points)
        validation["pca_components"] = pca.n_components_
        test["pca_components"] = pca.n_components
        validation["cluster_fit_partition"] = "validation"
        test["cluster_fit_partition"] = "validation"
        cluster_rows.extend([validation, test])
        selection_rows.append(
            {
                "split": int(split),
                "selected_k": k,
                "pca_components": int(pca.n_components_),
                "validation_silhouette_selected": silhouette_scores[k],
                "silhouette_scores": json.dumps(silhouette_scores, sort_keys=True),
                "scaler_fit_partition": "validation",
                "cluster_fit_partition": "validation",
                "test_assignment_only": True,
            }
        )
        for partition, frame in (("validation", validation), ("test", test)):
            profiles = _patient_cluster_profiles(frame, "cluster", k)
            for model in MODELS:
                subset = profiles[profiles.model == model]
                model_mean = subset[[f"cluster_{index}" for index in range(k)]].mean().to_numpy(float)
                profile_rows.append(
                    {
                        "split": int(split),
                        "partition": partition,
                        "model": model,
                        "seed_count": int(subset.seed.nunique()),
                        "events": int(len(frame[frame.model == model])),
                        "profile": json.dumps(model_mean.tolist()),
                        **{f"cluster_{index}_proportion": float(model_mean[index]) for index in range(k)},
                    }
                )
            for left_index, left_model in enumerate(MODELS):
                for right_model in MODELS[left_index + 1 :]:
                    left = profiles[profiles.model == left_model]
                    right = profiles[profiles.model == right_model]
                    merged = left.merge(right, on=["patient", "seed"], suffixes=("_left", "_right"))
                    patient_similarities = []
                    for row in merged.itertuples(index=False):
                        a = np.asarray([getattr(row, f"cluster_{index}_left") for index in range(k)])
                        b = np.asarray([getattr(row, f"cluster_{index}_right") for index in range(k)])
                        patient_similarities.append(_js_similarity(a, b))
                    left_mean = left[[f"cluster_{index}" for index in range(k)]].mean().to_numpy(float)
                    right_mean = right[[f"cluster_{index}" for index in range(k)]].mean().to_numpy(float)
                    profile_rows.append(
                        {
                            "split": int(split),
                            "partition": partition,
                            "model": f"{left_model}__{right_model}",
                            "seed_count": int(merged.seed.nunique()),
                            "events": int(len(left) + len(right)),
                            "profile_similarity_js": _js_similarity(left_mean, right_mean),
                            "patient_median_similarity_js": float(np.median(patient_similarities)) if patient_similarities else 0.0,
                        }
                    )
        labels = {model: index for index, model in enumerate(MODELS)}
        validation_y = validation.model.map(labels).to_numpy()
        test_y = test.model.map(labels).to_numpy()
        for centred in (False, True):
            train_x = validation.loc[:, FEATURES].copy()
            eval_x = test.loc[:, FEATURES].copy()
            if centred:
                train_x = train_x - train_x.groupby(validation.patient).transform("mean")
                eval_x = eval_x - eval_x.groupby(test.patient).transform("mean")
            lower = train_x.quantile(0.01)
            upper = train_x.quantile(0.99)
            train_x = train_x.clip(lower=lower, upper=upper, axis="columns")
            eval_x = eval_x.clip(lower=lower, upper=upper, axis="columns")
            # Fit the frozen scaler on validation only and apply it to test.
            fit_scaler = StandardScaler().fit(train_x)
            model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=SEED)
            model.fit(fit_scaler.transform(train_x), validation_y)
            probabilities = model.predict_proba(fit_scaler.transform(eval_x))
            predictions = model.predict(fit_scaler.transform(eval_x))
            fingerprint_rows.append(
                {
                    "split": int(split),
                    "patient_centered": centred,
                    "validation_events": int(len(validation)),
                    "test_events": int(len(test)),
                    "test_accuracy": float(accuracy_score(test_y, predictions)),
                    "test_balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
                    "test_macro_auroc": float(roc_auc_score(test_y, probabilities, multi_class="ovr", average="macro")),
                    "chance_accuracy": 1.0 / len(MODELS),
                    "classifier": "multinomial_logistic_C1_class_weight_balanced",
                    "fit_partition": "validation",
                    "test_used_for_selection": False,
                }
            )
    assignments = pd.concat(cluster_rows, ignore_index=True)
    assignments.to_parquet(REPORT_DIR / "h2_cluster_assignments.parquet", index=False)
    pd.DataFrame(selection_rows).to_csv(REPORT_DIR / "h2_cluster_selection.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(REPORT_DIR / "h2_cluster_profiles.csv", index=False)
    pd.DataFrame(fingerprint_rows).to_csv(REPORT_DIR / "h3_model_fingerprint.csv", index=False)
    summary = {
        "release_id": "v3-compact-extension-v1",
        "h2": {
            "status": "complete_exploratory",
            "fit_partition": "validation",
            "test_assignment_only": True,
            "cluster_selection": "max validation silhouette among k=3..6",
            "splits": sorted(features.split.unique().tolist()),
        },
        "h3": {
            "status": "complete_preliminary",
            "fit_partition": "validation",
            "test_used_for_selection": False,
            "patient_centering_sensitivity": True,
        },
        "outputs": [
            "h2_cluster_assignments.parquet",
            "h2_cluster_selection.csv",
            "h2_cluster_profiles.csv",
            "h3_model_fingerprint.csv",
        ],
    }
    (REPORT_DIR / "h2_h3_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(pd.DataFrame(selection_rows).to_string(index=False))
    print(pd.DataFrame(fingerprint_rows).to_string(index=False))


if __name__ == "__main__":
    main()
