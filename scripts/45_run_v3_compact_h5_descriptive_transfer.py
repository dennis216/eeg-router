"""Run descriptive H5 leave-one-architecture-out phenotype transfer.

This is deliberately not a functional suppressor/router experiment.  For each
direction and split, source-architecture train FP events fit the representation
and prototypes; source validation FP events select K and an unknown radius; the
target architecture's test labels are never read.  Only coverage, unknown rate,
distance, and cluster-distribution shift are reported.
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


ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "outputs" / "events" / "v3-compact-extension-v1_h4_event_features.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
DIRECTIONS = {
    "SC_to_T": (("singlem", "cbramod"), "tcn_gat"),
    "ST_to_C": (("singlem", "tcn_gat"), "cbramod"),
    "CT_to_S": (("cbramod", "tcn_gat"), "singlem"),
}
SIGNAL_FEATURES = (
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
SPLITS = (17, 42)
K_CANDIDATES = (3, 4, 5, 6)
UNKNOWN_QUANTILE = 0.95
SEED = 20260811


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float) + 1e-12
    right = np.asarray(right, dtype=float) + 1e-12
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def _points_fit(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA, pd.Series, pd.Series]:
    lower = frame.loc[:, SIGNAL_FEATURES].quantile(0.01)
    upper = frame.loc[:, SIGNAL_FEATURES].quantile(0.99)
    clipped = frame.loc[:, SIGNAL_FEATURES].clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(clipped)
    points = scaler.transform(clipped)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=SEED).fit(points)
    return pca.transform(points), scaler.transform(clipped), scaler, pca, lower, upper


def _transform(frame: pd.DataFrame, lower: pd.Series, upper: pd.Series, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    clipped = frame.loc[:, SIGNAL_FEATURES].clip(lower=lower, upper=upper, axis="columns")
    return pca.transform(scaler.transform(clipped))


def _fit_direction(features: pd.DataFrame, split: int, direction: str, source_models: tuple[str, ...], target_model: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_train = features[(features.split == split) & (features.partition == "train_fit_subset") & features.model.isin(source_models) & (features.event_type == "fp")].copy()
    source_validation = features[(features.split == split) & (features.partition == "validation") & features.model.isin(source_models) & (features.event_type == "fp")].copy()
    target = features[(features.split == split) & (features.partition == "test") & (features.model == target_model)].copy()
    # Target TP/FP fields are intentionally dropped before any transformation.
    target = target.drop(columns=["event_type"], errors="ignore")
    if len(source_train) < 20 or len(source_validation) < 20 or target.empty:
        raise RuntimeError(f"insufficient H5 data split={split} direction={direction}: train={len(source_train)} validation={len(source_validation)} target={len(target)}")
    train_points, _, scaler, pca, lower, upper = _points_fit(source_train)
    validation_points = _transform(source_validation, lower, upper, scaler, pca)
    target_points = _transform(target, lower, upper, scaler, pca)
    rng = np.random.default_rng(SEED)
    validation_sample = validation_points
    if len(validation_sample) > 12000:
        validation_sample = validation_sample[rng.choice(len(validation_sample), size=12000, replace=False)]
    silhouette_scores: dict[int, float] = {}
    for k in K_CANDIDATES:
        if len(train_points) < k or len(validation_sample) <= k:
            continue
        fit_for_selection = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(train_points)
        val_labels = fit_for_selection.predict(validation_sample)
        if len(np.unique(val_labels)) > 1:
            silhouette_scores[k] = float(silhouette_score(validation_sample, val_labels))
    if not silhouette_scores:
        raise RuntimeError(f"no valid H5 K selection split={split} direction={direction}")
    selected_k = int(max(silhouette_scores, key=silhouette_scores.get))
    clusterer = KMeans(n_clusters=selected_k, n_init=50, random_state=SEED).fit(train_points)
    validation_labels = clusterer.predict(validation_points)
    target_labels = clusterer.predict(target_points)
    validation_distances = np.sqrt(((validation_points[:, None, :] - clusterer.cluster_centers_[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    target_distances = np.sqrt(((target_points[:, None, :] - clusterer.cluster_centers_[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    unknown_radius = float(np.quantile(validation_distances, UNKNOWN_QUANTILE))
    target_known = target_distances <= unknown_radius
    source_profile = np.bincount(validation_labels[validation_distances <= unknown_radius], minlength=selected_k).astype(float)
    target_profile = np.bincount(target_labels[target_known], minlength=selected_k).astype(float)
    source_profile /= max(source_profile.sum(), 1.0)
    target_profile /= max(target_profile.sum(), 1.0)
    summary_rows: list[dict[str, object]] = []
    for target_seed in sorted(target.seed.unique().tolist()):
        seed_mask = target.seed.to_numpy() == target_seed
        distances = target_distances[seed_mask]
        known = distances <= unknown_radius
        labels = target_labels[seed_mask]
        profile = np.bincount(labels[known], minlength=selected_k).astype(float)
        profile /= max(profile.sum(), 1.0)
        summary_rows.append(
            {
                "direction": direction,
                "split": split,
                "target_seed": int(target_seed),
                "source_models": "+".join(source_models),
                "target_model": target_model,
                "status": "descriptive_transfer_only",
                "source_train_fp_events": int(len(source_train)),
                "source_validation_fp_events": int(len(source_validation)),
                "target_test_events_labels_excluded": int(seed_mask.sum()),
                "pca_components": int(pca.n_components_),
                "selected_k": selected_k,
                "validation_silhouette_selected": float(silhouette_scores[selected_k]),
                "unknown_radius_validation_q95": unknown_radius,
                "target_coverage": float(known.mean()) if len(known) else 0.0,
                "target_unknown_rate": float((~known).mean()) if len(known) else 1.0,
                "target_mean_nearest_distance": float(np.mean(distances)) if len(distances) else np.nan,
                "target_p95_nearest_distance": float(np.quantile(distances, 0.95)) if len(distances) else np.nan,
                "target_source_profile_js_divergence": _js_divergence(source_profile, profile),
                "target_assigned_cluster_profile": json.dumps(profile.tolist()),
                "test_target_labels_used": False,
            }
        )
    pooled_profile = np.bincount(target_labels[target_known], minlength=selected_k).astype(float)
    pooled_profile /= max(pooled_profile.sum(), 1.0)
    summary_rows.append(
        {
            "direction": direction,
            "split": split,
            "target_seed": "pooled",
            "source_models": "+".join(source_models),
            "target_model": target_model,
            "status": "descriptive_transfer_only",
            "source_train_fp_events": int(len(source_train)),
            "source_validation_fp_events": int(len(source_validation)),
            "target_test_events_labels_excluded": int(len(target)),
            "pca_components": int(pca.n_components_),
            "selected_k": selected_k,
            "validation_silhouette_selected": float(silhouette_scores[selected_k]),
            "unknown_radius_validation_q95": unknown_radius,
            "target_coverage": float(target_known.mean()),
            "target_unknown_rate": float((~target_known).mean()),
            "target_mean_nearest_distance": float(np.mean(target_distances)),
            "target_p95_nearest_distance": float(np.quantile(target_distances, 0.95)),
            "target_source_profile_js_divergence": _js_divergence(source_profile, pooled_profile),
            "target_assigned_cluster_profile": json.dumps(pooled_profile.tolist()),
            "test_target_labels_used": False,
        }
    )
    assignment_rows: list[dict[str, object]] = []
    target_records = target.reset_index(drop=True)
    for index, row in target_records.iterrows():
        assignment_rows.append(
            {
                "direction": direction,
                "split": split,
                "target_seed": int(row.seed),
                "target_model": target_model,
                "patient": str(row.patient),
                "recording": str(row.recording),
                "start_s": float(row.start_s),
                "end_s": float(row.end_s),
                "nearest_distance": float(target_distances[index]),
                "unknown": bool(not target_known[index]),
                "assigned_cluster": int(target_labels[index]) if target_known[index] else -1,
            }
        )
    return summary_rows, assignment_rows


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    summary: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for split in SPLITS:
        for direction, (source_models, target_model) in DIRECTIONS.items():
            rows, target_assignments = _fit_direction(features, split, direction, source_models, target_model)
            summary.extend(rows)
            assignments.extend(target_assignments)
            print(f"H5_DESCRIPTIVE_DONE split={split} direction={direction} target={target_model} pooled_unknown_rate={rows[-1]['target_unknown_rate']:.3f}", flush=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "h5_descriptive_transfer.csv"
    assignment_path = REPORT_DIR / "h5_descriptive_transfer_assignments.parquet"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.DataFrame(assignments).to_parquet(assignment_path, index=False)
    manifest = {
        "release_id": "v3-compact-extension-v1",
        "status": "descriptive_transfer_only",
        "h4_status_at_start": "preliminary_negative",
        "directions": {direction: {"source_models": list(source), "target_model": target} for direction, (source, target) in DIRECTIONS.items()},
        "splits": list(SPLITS),
        "source_fit_partition": "train_fit_subset",
        "source_validation_partition": "validation",
        "target_assignment_partition": "test",
        "target_labels_used": False,
        "target_architecture_excluded_from_fit": True,
        "unknown_radius": "95th percentile of source-validation nearest-prototype distance",
        "summary_sha256": _sha256(summary_path),
        "assignments_sha256": _sha256(assignment_path),
        "h4_event_features_sha256": _sha256(FEATURE_PATH),
        "functional_transfer_claim": False,
        "interpretation": "descriptive phenotype coverage/unknown/distribution-shift analysis only; no suppressor, router, threshold change, or target-label fit",
    }
    (REPORT_DIR / "h5_descriptive_transfer_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
