"""Run the validation-only H4 gate for the standard GroupKFold release.

This script is deliberately separate from the older compact-v3 H4 scripts.
It reads only validation probabilities when deciding the branch.  The frozen
outer-test probabilities are not opened by this script.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
EVENT_PATH = REPORT_DIR / "h2h3_events.parquet"
FEATURE_MANIFEST_PATH = REPORT_DIR / "h2h3_fp_features_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
MAX_ONSET_DIFF_S = 30.0
WEIGHT_STEP = 0.25
QUANTILE_LEVELS = np.asarray([0.50, 0.65, 0.80, 0.90, 0.97, 0.995], dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str) == str(recording)]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def _load_validation(fold: int, seed: int, model: str) -> tuple[pd.DataFrame, float]:
    run = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
    metrics_path = run / "validation" / "metrics_step_05000.json"
    probability_path = run / "validation" / "probabilities_step_05000.parquet"
    metrics = json.loads(metrics_path.read_text())
    threshold = float(metrics["selected_event_operating_point"]["threshold"])
    table = pd.read_parquet(probability_path)
    required = {"patient", "recording", "end", "probability"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{probability_path} missing columns: {sorted(missing)}")
    return table[["patient", "recording", "end", "probability"]].copy(), threshold


def _truth_metrics(
    table: pd.DataFrame,
    threshold: float,
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> dict[str, float]:
    true_positive = 0
    false_alarm = 0
    truth_count = 0
    nonseizure_hours = 0.0
    delays: list[float] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if ordered.empty:
            continue
        evaluation_start = float(ordered.end.iloc[0])
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=float(threshold),
        )
        truth = _truths(seizures, str(recording), evaluation_start)
        matched = match_events(predictions, truth)
        true_positive += len(matched.pairs)
        false_alarm += len(matched.unmatched_predictions)
        truth_count += len(truth)
        for pair in matched.pairs:
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - truth[pair.truth_index][0]))
        duration = float(durations[str(recording)])
        seizure_seconds = sum(max(0.0, end - start) for start, end in truth)
        nonseizure_hours += max(0.0, duration - evaluation_start - seizure_seconds) / 3600.0
    sensitivity = float(true_positive / truth_count) if truth_count else 0.0
    fa_per_24h = float(false_alarm * 24.0 / nonseizure_hours) if nonseizure_hours else math.inf
    return {
        "true_positive_events": int(true_positive),
        "false_alarm_events": int(false_alarm),
        "truth_events": int(truth_count),
        "sensitivity": sensitivity,
        "fa_per_24h": fa_per_24h,
        "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
        "nonseizure_hours": nonseizure_hours,
    }


def _combine(tables: dict[str, pd.DataFrame], weights: tuple[float, float, float]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in MODELS:
        frame = tables[model].rename(columns={"probability": f"probability_{model}"})
        keys = ["patient", "recording", "end"]
        merged = frame if merged is None else merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["patient", "recording", "end", "probability"])
    merged["probability"] = sum(
        weight * merged[f"probability_{model}"] for weight, model in zip(weights, MODELS, strict=True)
    )
    return merged[["patient", "recording", "end", "probability"]]


def _weight_grid() -> list[tuple[float, float, float]]:
    values = np.arange(0.0, 1.0 + WEIGHT_STEP / 2.0, WEIGHT_STEP)
    return [
        (float(a), float(b), float(1.0 - a - b))
        for a in values
        for b in values
        if a + b <= 1.0 + 1e-9
    ]


def _oracle(
    tables: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> dict[str, object]:
    baselines: list[dict[str, object]] = []
    for model in MODELS:
        metrics = _truth_metrics(tables[model], thresholds[model], seizures, durations)
        baselines.append({"model": model, "threshold": thresholds[model], **metrics})
    baseline_frame = pd.DataFrame(baselines)
    target_sensitivity = float(baseline_frame.sensitivity.max())
    eligible_baselines = baseline_frame[baseline_frame.sensitivity >= target_sensitivity - 0.03 - 1e-12]
    if eligible_baselines.empty:
        eligible_baselines = baseline_frame.loc[[int(baseline_frame.sensitivity.idxmax())]]
    baseline = eligible_baselines.sort_values(
        ["fa_per_24h", "sensitivity"], ascending=[True, False]
    ).iloc[0].to_dict()

    candidates: list[dict[str, object]] = []
    for weights in _weight_grid():
        combined = _combine(tables, weights)
        if combined.empty:
            continue
        values = combined.probability.to_numpy(dtype=float)
        candidate_thresholds = sorted(
            set(float(value) for value in thresholds.values())
            | set(float(value) for value in np.quantile(values, QUANTILE_LEVELS))
        )
        for threshold in candidate_thresholds:
            metrics = _truth_metrics(combined, threshold, seizures, durations)
            if metrics["sensitivity"] + 1e-12 < target_sensitivity - 0.03:
                continue
            candidates.append(
                {
                    "weight_singlem": weights[0],
                    "weight_cbramod": weights[1],
                    "weight_tcn_gat": weights[2],
                    "threshold": float(threshold),
                    **metrics,
                }
            )
    if candidates:
        oracle = sorted(candidates, key=lambda row: (float(row["fa_per_24h"]), -float(row["sensitivity"]))) [0]
    else:
        oracle = {
            "weight_singlem": math.nan,
            "weight_cbramod": math.nan,
            "weight_tcn_gat": math.nan,
            "threshold": math.nan,
            "sensitivity": 0.0,
            "fa_per_24h": math.inf,
            "true_positive_events": 0,
            "false_alarm_events": 0,
            "truth_events": 0,
            "mean_delay_s": math.nan,
            "nonseizure_hours": 0.0,
        }
    baseline_fa = float(baseline["fa_per_24h"])
    oracle_fa = float(oracle["fa_per_24h"])
    reduction = float((baseline_fa - oracle_fa) / baseline_fa) if baseline_fa > 0 and math.isfinite(oracle_fa) else 0.0
    return {
        "baseline_model": str(baseline["model"]),
        "baseline_threshold": float(baseline["threshold"]),
        "baseline_sensitivity": float(baseline["sensitivity"]),
        "baseline_fa_per_24h": baseline_fa,
        "target_sensitivity": target_sensitivity,
        "oracle_weight_singlem": float(oracle["weight_singlem"]),
        "oracle_weight_cbramod": float(oracle["weight_cbramod"]),
        "oracle_weight_tcn_gat": float(oracle["weight_tcn_gat"]),
        "oracle_threshold": float(oracle["threshold"]),
        "oracle_sensitivity": float(oracle["sensitivity"]),
        "oracle_fa_per_24h": oracle_fa,
        "oracle_fa_reduction": reduction,
        "oracle_sensitivity_loss_vs_target": float(target_sensitivity - float(oracle["sensitivity"])),
        "oracle_sensitivity_loss_vs_baseline": float(float(baseline["sensitivity"]) - float(oracle["sensitivity"])),
        "oracle_candidate_count": int(len(candidates)),
    }


def _jaccard(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int | float]:
    left_count = 0
    right_count = 0
    matched = 0
    patients = sorted(set(left.patient.astype(str)) | set(right.patient.astype(str)))
    for patient in patients:
        recordings = sorted(
            set(left.loc[left.patient.astype(str) == patient, "recording"].astype(str))
            | set(right.loc[right.patient.astype(str) == patient, "recording"].astype(str))
        )
        for recording in recordings:
            left_times = sorted(
                left.loc[
                    (left.patient.astype(str) == patient) & (left.recording.astype(str) == recording),
                    "start_s",
                ].astype(float).tolist()
            )
            right_times = sorted(
                right.loc[
                    (right.patient.astype(str) == patient) & (right.recording.astype(str) == recording),
                    "start_s",
                ].astype(float).tolist()
            )
            i = j = 0
            while i < len(left_times) and j < len(right_times):
                delta = left_times[i] - right_times[j]
                if abs(delta) <= MAX_ONSET_DIFF_S:
                    matched += 1
                    i += 1
                    j += 1
                elif delta < -MAX_ONSET_DIFF_S:
                    i += 1
                else:
                    j += 1
            left_count += len(left_times)
            right_count += len(right_times)
    union = left_count + right_count - matched
    return {
        "left_fp_events": left_count,
        "right_fp_events": right_count,
        "matched_fp_events": matched,
        "fp_jaccard": float(matched / union) if union else 0.0,
    }


def _h2_gate() -> dict[str, object]:
    selection = pd.read_csv(REPORT_DIR / "h2_groupkfold_cluster_selection.csv")
    selected_k = sorted(int(value) for value in selection.selected_k.tolist())
    silhouettes = [float(value) for value in selection.validation_silhouette_selected.tolist()]
    # H2 was explicitly exploratory in the protocol.  A shared-phenotype
    # suppressor is therefore not selected unless stable phenotype evidence is
    # frozen in advance; the current run has fold-varying k and low silhouettes.
    return {
        "shared_phenotype_gate": False,
        "reason": "H2 remains exploratory; selected cluster counts vary across folds and no pre-frozen cross-seed stability criterion is available",
        "selected_k_by_fold": selected_k,
        "validation_silhouette_by_fold": silhouettes,
        "test_used_for_gate": False,
    }


def main() -> None:
    events = pd.read_parquet(EVENT_PATH)
    validation_events = events[events.partition == "validation"].copy()
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    oracle_rows: list[dict[str, object]] = []
    jaccard_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            tables: dict[str, pd.DataFrame] = {}
            thresholds: dict[str, float] = {}
            for model in MODELS:
                table, threshold = _load_validation(fold, seed, model)
                tables[model] = table
                thresholds[model] = threshold
            oracle_rows.append({"fold": fold, "seed": seed, **_oracle(tables, thresholds, seizures, durations)})
            group = validation_events[(validation_events.fold == fold) & (validation_events.seed == seed)]
            for left_model, right_model in itertools.combinations(MODELS, 2):
                left = group[(group.model == left_model) & (group.event_type == "fp")]
                right = group[(group.model == right_model) & (group.event_type == "fp")]
                jaccard_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        **_jaccard(left, right),
                    }
                )

    oracle_frame = pd.DataFrame(oracle_rows)
    jaccard_frame = pd.DataFrame(jaccard_rows)
    pair_pass = bool((jaccard_frame.fp_jaccard <= 0.50).any())
    oracle_pass = bool(
        (
            (oracle_frame.oracle_fa_reduction >= 0.25)
            & (oracle_frame.oracle_sensitivity_loss_vs_target <= 0.03 + 1e-12)
        ).any()
    )
    feature_manifest = json.loads(FEATURE_MANIFEST_PATH.read_text())
    required_features = set(feature_manifest["feature_names"])
    allowed_features = {
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
    }
    feature_pass = bool(required_features <= allowed_features and feature_manifest.get("fit_or_selection_performed") is False)
    h2 = _h2_gate()
    if pair_pass and oracle_pass and feature_pass and not bool(h2["shared_phenotype_gate"]):
        branch = "router_candidate"
    elif pair_pass and oracle_pass and feature_pass and bool(h2["shared_phenotype_gate"]):
        branch = "suppressor_candidate"
    else:
        branch = "negative_result"
    gates = {
        "release_id": RUN_ID,
        "gate": "H4_functional_gate",
        "status": "frozen_validation_only",
        "branch": branch,
        "pair_gate": {
            "pass": pair_pass,
            "criterion": "at least one validation FP pair has Jaccard <= 0.50",
            "minimum_jaccard": float(jaccard_frame.fp_jaccard.min()) if not jaccard_frame.empty else math.nan,
            "passing_comparisons": int((jaccard_frame.fp_jaccard <= 0.50).sum()),
            "comparisons": int(len(jaccard_frame)),
        },
        "oracle_gate": {
            "pass": oracle_pass,
            "criterion": "validation-only convex-combination oracle reduces FA/24h >=25% with sensitivity loss <=3pp",
            "passing_groups": int(
                (
                    (oracle_frame.oracle_fa_reduction >= 0.25)
                    & (oracle_frame.oracle_sensitivity_loss_vs_target <= 0.03 + 1e-12)
                ).sum()
            ),
            "groups": int(len(oracle_frame)),
        },
        "feature_timing_gate": {
            "pass": feature_pass,
            "criterion": "all frozen H2/H3 features are available by event end and no fit/selection is performed in feature extraction",
            "feature_names": sorted(required_features),
        },
        "h2_shared_phenotype_gate": h2,
        "test_used_for_branch_selection": False,
        "source_hashes": {
            "event_table_sha256": _sha256(EVENT_PATH),
            "feature_manifest_sha256": _sha256(FEATURE_MANIFEST_PATH),
            "h2_selection_sha256": _sha256(REPORT_DIR / "h2_groupkfold_cluster_selection.csv"),
            "h3_fingerprint_sha256": _sha256(REPORT_DIR / "h3_groupkfold_model_fingerprint.csv"),
        },
        "next_step": (
            "run a standard GroupKFold router only if a matching implementation is available"
            if branch == "router_candidate"
            else "freeze H4 as negative_result; do not train a functional model"
            if branch == "negative_result"
            else "run a standard GroupKFold phenotype suppressor"
        ),
    }
    oracle_frame.to_csv(REPORT_DIR / "h4_groupkfold_validation_oracle.csv", index=False)
    jaccard_frame.to_csv(REPORT_DIR / "h4_groupkfold_validation_fp_jaccard.csv", index=False)
    (REPORT_DIR / "h4_groupkfold_gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    print("\nValidation oracle summary:\n", oracle_frame.to_string(index=False))
    print("\nValidation FP Jaccard summary:\n", jaccard_frame.to_string(index=False))


if __name__ == "__main__":
    main()
