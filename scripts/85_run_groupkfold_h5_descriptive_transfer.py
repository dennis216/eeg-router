"""Run the standard GroupKFold H5 descriptive transfer analysis.

This script is deliberately descriptive: it does not train a suppressor,
router, detector, or target-architecture classifier.  For each outer fold and
leave-one-architecture-out direction, source architectures are used to fit a
feature transform and phenotype prototypes on the frozen outer-train
partition.  Source validation selects the cluster count and the unknown-radius
work point.  The held-out target architecture's outer-test rows are assigned
once, without reading any target event labels.  The output reports coverage,
unknown rate, nearest-prototype distance, and cluster-profile shift only.

This is a new standard GroupKFold namespace.  It must not be replaced by the
legacy compact or greedy-split H5 implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
FEATURE_PATH = REPORT_DIR / "h2h3_fp_features.parquet"
FEATURE_MANIFEST_PATH = REPORT_DIR / "h2h3_fp_features_manifest.json"
OUTPUT_SUMMARY = REPORT_DIR / "h5_groupkfold_descriptive_transfer.csv"
OUTPUT_ASSIGNMENTS = REPORT_DIR / "h5_groupkfold_descriptive_transfer_assignments.parquet"
OUTPUT_MANIFEST = REPORT_DIR / "h5_groupkfold_descriptive_transfer_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
DIRECTIONS = {
    "SC_to_T": (("singlem", "cbramod"), "tcn_gat"),
    "ST_to_C": (("singlem", "tcn_gat"), "cbramod"),
    "CT_to_S": (("cbramod", "tcn_gat"), "singlem"),
}
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
UNKNOWN_QUANTILE = 0.95
RANDOM_SEED = 20260822


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float) + 1e-12
    right = np.asarray(right, dtype=float) + 1e-12
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * np.log(left / midpoint))
    divergence += 0.5 * np.sum(right * np.log(right / midpoint))
    return float(divergence)


def _prepare_fit_transform(
    train: pd.DataFrame,
    other_frames: tuple[pd.DataFrame, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], StandardScaler, PCA, pd.Series, pd.Series, pd.Series]:
    """Fit cleaning, scaling and PCA on source train only."""
    train_values = train.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    median = train_values.median().fillna(0.0)
    train_values = train_values.fillna(median)
    lower = train_values.quantile(0.01)
    upper = train_values.quantile(0.99)
    train_values = train_values.clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(train_values)
    train_scaled = scaler.transform(train_values)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_SEED).fit(train_scaled)

    transformed: list[np.ndarray] = []
    for frame in other_frames:
        values = frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(median).clip(lower=lower, upper=upper, axis="columns")
        transformed.append(pca.transform(scaler.transform(values)))
    return pca.transform(train_scaled), tuple(transformed), scaler, pca, median, lower, upper


def _select_k(
    train_points: np.ndarray,
    validation_points: np.ndarray,
    fold: int,
    direction: str,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in K_CANDIDATES:
        if len(train_points) <= k or len(validation_points) <= k:
            continue
        clusterer = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_SEED + fold,
        ).fit(train_points)
        labels = clusterer.predict(validation_points)
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(validation_points, labels))
    if not scores:
        raise RuntimeError(f"no valid source-validation K on fold={fold} direction={direction}")
    # max() is deterministic because K_CANDIDATES is ordered and ties retain
    # the first candidate.
    selected = int(max(scores, key=scores.get))
    return selected, scores


def _nearest_distances(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sqrt(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1), distances.argmin(axis=1)


def _profile(labels: np.ndarray, distances: np.ndarray, radius: float, k: int) -> tuple[np.ndarray, float]:
    known = distances <= radius
    counts = np.bincount(labels[known], minlength=k).astype(float)
    profile = counts / max(counts.sum(), 1.0)
    return profile, float(known.mean()) if len(known) else 0.0


def _direction(
    features: pd.DataFrame,
    fold: int,
    direction: str,
    source_models: tuple[str, ...],
    target_model: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fold_frame = features[features.split == fold].copy()
    source_train = fold_frame[
        (fold_frame.partition == "train") & fold_frame.model.isin(source_models)
    ].copy()
    source_validation = fold_frame[
        (fold_frame.partition == "validation") & fold_frame.model.isin(source_models)
    ].copy()
    # The target architecture is selected only to identify the held-out test
    # rows.  Its event_type column is dropped and never read for evaluation.
    target_test = fold_frame[
        (fold_frame.partition == "test") & (fold_frame.model == target_model)
    ].copy()
    if len(source_train) < 20 or len(source_validation) < 20 or target_test.empty:
        raise RuntimeError(
            f"insufficient H5 data fold={fold} direction={direction}: "
            f"source_train={len(source_train)} source_validation={len(source_validation)} "
            f"target_test={len(target_test)}"
        )
    target_test = target_test.drop(columns=["event_type"], errors="ignore")

    train_points, transformed, _, pca, _, _, _ = _prepare_fit_transform(
        source_train, (source_validation, target_test)
    )
    validation_points, target_points = transformed
    selected_k, silhouette_scores = _select_k(train_points, validation_points, fold, direction)
    clusterer = KMeans(
        n_clusters=selected_k,
        n_init=50,
        random_state=RANDOM_SEED + fold,
    ).fit(train_points)
    validation_distances, validation_labels = _nearest_distances(
        validation_points, clusterer.cluster_centers_
    )
    target_distances, target_labels = _nearest_distances(
        target_points, clusterer.cluster_centers_
    )
    unknown_radius = float(np.quantile(validation_distances, UNKNOWN_QUANTILE))
    source_profile, source_coverage = _profile(
        validation_labels, validation_distances, unknown_radius, selected_k
    )

    rows: list[dict[str, object]] = []
    target_records = target_test.reset_index(drop=True)
    target_seed_values = sorted(target_records.seed.astype(int).unique().tolist())
    strata = [(str(seed), target_records[target_records.seed.astype(int) == seed]) for seed in target_seed_values]
    strata.append(("pooled", target_records))
    for target_seed, stratum in strata:
        indices = stratum.index.to_numpy(dtype=int)
        distances = target_distances[indices]
        labels = target_labels[indices]
        profile, coverage = _profile(labels, distances, unknown_radius, selected_k)
        rows.append(
            {
                "release_id": RUN_ID,
                "status": "descriptive_transfer_only",
                "fold": int(fold),
                "direction": direction,
                "source_models": "+".join(source_models),
                "target_model": target_model,
                "target_seed": target_seed,
                "source_train_fp_events": int(len(source_train)),
                "source_validation_fp_events": int(len(source_validation)),
                "target_test_rows_labels_excluded": int(len(stratum)),
                "pca_components": int(pca.n_components_),
                "selected_k": int(selected_k),
                "validation_silhouette_selected": float(silhouette_scores[selected_k]),
                "validation_silhouette_scores": json.dumps(silhouette_scores, sort_keys=True),
                "unknown_radius_source_validation_q95": unknown_radius,
                "source_validation_coverage": source_coverage,
                "target_coverage": coverage,
                "target_unknown_rate": float(1.0 - coverage),
                "target_mean_nearest_distance": float(np.mean(distances)) if len(distances) else np.nan,
                "target_p95_nearest_distance": float(np.quantile(distances, 0.95)) if len(distances) else np.nan,
                "target_source_profile_js_divergence": _js_divergence(source_profile, profile),
                "source_validation_cluster_profile": json.dumps(source_profile.tolist()),
                "target_assigned_cluster_profile": json.dumps(profile.tolist()),
                "source_fit_partition": "outer_train_discovery",
                "selection_partition": "source_validation",
                "assignment_partition": "outer_test",
                "target_architecture_labels_used": False,
                "functional_transfer_claim": False,
            }
        )

    assignments: list[dict[str, object]] = []
    for index, row in target_records.iterrows():
        known = bool(target_distances[index] <= unknown_radius)
        assignments.append(
            {
                "release_id": RUN_ID,
                "status": "descriptive_transfer_only",
                "fold": int(fold),
                "direction": direction,
                "target_model": target_model,
                "target_seed": int(row.seed),
                "patient": str(row.patient),
                "recording": str(row.recording),
                "start_s": float(row.start_s),
                "end_s": float(row.end_s),
                "nearest_distance": float(target_distances[index]),
                "unknown": bool(not known),
                "assigned_cluster": int(target_labels[index]) if known else -1,
                "target_architecture_labels_used": False,
            }
        )
    return rows, assignments


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    features = pd.read_parquet(FEATURE_PATH)
    expected = {"partition", "split", "seed", "model", "patient", "recording", *FEATURES}
    missing = expected.difference(features.columns)
    if missing:
        raise RuntimeError(f"H5 feature table missing columns: {sorted(missing)}")
    if set(features.model.unique()) != set(MODELS):
        raise RuntimeError(f"unexpected models in H5 feature table: {sorted(features.model.unique())}")
    if set(features.split.unique()) != set(FOLDS):
        raise RuntimeError(f"unexpected folds in H5 feature table: {sorted(features.split.unique())}")
    if set(features.seed.unique()) != set(SEEDS):
        raise RuntimeError(f"unexpected seeds in H5 feature table: {sorted(features.seed.unique())}")

    summary: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for fold in FOLDS:
        for direction, (source_models, target_model) in DIRECTIONS.items():
            rows, target_assignments = _direction(
                features, fold, direction, source_models, target_model
            )
            summary.extend(rows)
            assignments.extend(target_assignments)
            pooled = rows[-1]
            print(
                "H5_GROUPKFOLD_DESCRIPTIVE_DONE "
                f"fold={fold} direction={direction} target={target_model} "
                f"pooled_unknown_rate={pooled['target_unknown_rate']:.3f}",
                flush=True,
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary)
    assignment_frame = pd.DataFrame(assignments)
    summary_frame.to_csv(OUTPUT_SUMMARY, index=False)
    assignment_frame.to_parquet(OUTPUT_ASSIGNMENTS, index=False)
    source_manifest_hash = _sha256(FEATURE_MANIFEST_PATH) if FEATURE_MANIFEST_PATH.exists() else None
    manifest = {
        "release_id": RUN_ID,
        "h5_namespace": "h5-standard-groupkfold-descriptive-v1",
        "status": "descriptive_transfer_only",
        "directions": {
            name: {"source_models": list(source), "target_model": target}
            for name, (source, target) in DIRECTIONS.items()
        },
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "bundle_count": len(FOLDS) * len(DIRECTIONS),
        "summary_rows": int(len(summary_frame)),
        "assignment_rows": int(len(assignment_frame)),
        "source_feature_table": str(FEATURE_PATH.relative_to(ROOT)),
        "source_feature_table_sha256": _sha256(FEATURE_PATH),
        "source_feature_manifest_sha256": source_manifest_hash,
        "source_fit_partition": "outer_train_discovery",
        "selection_partition": "source_validation",
        "assignment_partition": "outer_test",
        "target_architecture_labels_used": False,
        "target_architecture_excluded_from_fit": True,
        "unknown_radius": "95th percentile of source-validation nearest-prototype distance",
        "feature_names": list(FEATURES),
        "cluster_candidates": list(K_CANDIDATES),
        "pca_variance_target": 0.90,
        "random_seed": RANDOM_SEED,
        "functional_transfer_claim": False,
        "summary_sha256": _sha256(OUTPUT_SUMMARY),
        "assignments_sha256": _sha256(OUTPUT_ASSIGNMENTS),
        "legacy_compact_or_greedy_code_used": False,
        "interpretation": (
            "Leave-one-architecture-out descriptive coverage, unknown-rate, "
            "nearest-prototype distance, and cluster-profile shift only; no "
            "target-label fitting, threshold adjustment, suppressor, router, "
            "or functional transfer claim."
        ),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
