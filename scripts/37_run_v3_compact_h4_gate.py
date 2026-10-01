"""Freeze and evaluate the validation-only H4 router gate.

This script deliberately reads validation probabilities and frozen metadata only.
It does not read test probabilities, test labels, or test AUROC when choosing the
H4 branch.  The oracle is a constrained validation-only convex-combination upper
bound: a simplex weight grid and validation thresholds are searched, then the
best operating point within three percentage points of the best single-model
sensitivity is retained.  This is an upper bound for a trainable router, not a
reported test result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
EVENT_PATH = ROOT / "outputs" / "events" / "v3-compact-extension-v1_validation_events.parquet"
FEATURE_DICT = REPORT_DIR / "feature_dictionary.csv"
MODELS = ("singlem", "cbramod", "tcn_gat")
RUNS = (
    ("BSN-pilot-v1", 17, (17, 42, 3407)),
    ("BSN-pilot-v1-multisplit", 42, (17, 42)),
    ("v3-lite-2split", 42, (3407,)),
)
SPLITS = (17, 42)
MAX_ONSET_DIFF_S = 30.0
WEIGHT_STEP = 0.25
# A deterministic, validation-only threshold set.  It contains each frozen
# single-model validation threshold plus six score quantiles per weight.  This
# keeps the oracle auditable without spending hours re-eventizing every 0.01
# threshold on every recording.
QUANTILE_LEVELS = np.asarray([0.50, 0.65, 0.80, 0.90, 0.97, 0.995], dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording: str, evaluation_start: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording]
    return [
        (max(float(row.start_s), evaluation_start), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start
    ]


def _weight_grid() -> list[tuple[float, float, float]]:
    values = np.arange(0.0, 1.0 + WEIGHT_STEP / 2.0, WEIGHT_STEP)
    return [
        (float(a), float(b), float(1.0 - a - b))
        for a in values
        for b in values
        if a + b <= 1.0 + 1e-8
    ]


def _evaluate(table: pd.DataFrame, score_column: str, threshold: float, truths: pd.DataFrame) -> dict[str, float]:
    tp = fp = truth_count = 0
    recording_rows: list[tuple[str, int, int, float]] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        evaluation_start = float(ordered.end.iloc[0])
        predictions = eventize(
            ordered.end.to_numpy(float),
            ordered[score_column].to_numpy(float),
            threshold=float(threshold),
        )
        truth = _truths(truths, str(recording), evaluation_start)
        matched = match_events(predictions, truth)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        truth_count += len(truth)
        recording_rows.append((str(recording), len(matched.pairs), len(matched.unmatched_predictions), evaluation_start))
    # The denominator is identical for all methods within a split/seed.  It is
    # computed from recording durations and seizure overlap, without labels from
    # any held-out test partition.
    nonseizure_hours = 0.0
    durations = truths.attrs["recording_durations"]
    for recording, _, _, evaluation_start in recording_rows:
        duration = float(durations[recording])
        usable = max(0.0, duration - evaluation_start)
        for start, end in _truths(truths, recording, evaluation_start):
            usable -= max(0.0, min(duration, end) - max(evaluation_start, start))
        nonseizure_hours += max(0.0, usable) / 3600.0
    sensitivity = float(tp / truth_count) if truth_count else 0.0
    fa = float(fp / nonseizure_hours * 24.0) if nonseizure_hours else math.inf
    return {
        "threshold": float(threshold),
        "true_positive_events": int(tp),
        "false_alarm_events": int(fp),
        "truth_events": int(truth_count),
        "sensitivity": sensitivity,
        "fa_per_24h": fa,
        "nonseizure_hours": nonseizure_hours,
    }


def _pair_jaccard(events: pd.DataFrame, split: int, seed: int, left: str, right: str) -> dict[str, float]:
    left_frame = events[(events.split == split) & (events.seed == seed) & (events.model == left) & (events.event_type == "fp")]
    right_frame = events[(events.split == split) & (events.seed == seed) & (events.model == right) & (events.event_type == "fp")]
    left_count = right_count = matched = 0
    patients = sorted(set(left_frame.patient.astype(str)) | set(right_frame.patient.astype(str)))
    for patient in patients:
        recordings = sorted(
            set(left_frame.loc[left_frame.patient.astype(str) == patient, "recording"].astype(str))
            | set(right_frame.loc[right_frame.patient.astype(str) == patient, "recording"].astype(str))
        )
        for recording in recordings:
            left_times = sorted(left_frame.loc[(left_frame.patient.astype(str) == patient) & (left_frame.recording.astype(str) == recording), "start_s"].astype(float).tolist())
            right_times = sorted(right_frame.loc[(right_frame.patient.astype(str) == patient) & (right_frame.recording.astype(str) == recording), "start_s"].astype(float).tolist())
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
        "left_fp_events": int(left_count),
        "right_fp_events": int(right_count),
        "matched_fp_events": int(matched),
        "fp_jaccard": float(matched / union) if union else 0.0,
    }


def _load_validation_run(namespace: str, split: int, seed: int, model: str) -> tuple[pd.DataFrame, dict[str, object]]:
    run = ROOT / "runs" / namespace / model / f"split{split}_seed{seed}_main"
    metrics = json.loads((run / "validation" / "metrics_step_05000.json").read_text())
    path = run / "validation" / "probabilities_step_05000.parquet"
    table = pd.read_parquet(path)
    required = {"patient", "recording", "end", "probability"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table[["patient", "recording", "end", "probability"]].copy(), metrics


def _combine_tables(tables: dict[str, pd.DataFrame], weights: tuple[float, float, float]) -> pd.DataFrame:
    merged = None
    for model in MODELS:
        frame = tables[model].rename(columns={"probability": f"probability_{model}"})
        key = ["patient", "recording", "end"]
        merged = frame if merged is None else merged.merge(frame, on=key, how="inner", validate="one_to_one")
    merged["score"] = sum(weight * merged[f"probability_{model}"] for weight, model in zip(weights, MODELS, strict=True))
    return merged[["patient", "recording", "end", "score"]]


def _oracle_for_group(tables: dict[str, pd.DataFrame], metrics: dict[str, dict[str, object]], truths: pd.DataFrame) -> tuple[dict[str, float], dict[str, object]]:
    baseline_rows: list[dict[str, float | str]] = []
    for model in MODELS:
        threshold = float(metrics[model]["selected_event_operating_point"]["threshold"])
        result = _evaluate(tables[model].rename(columns={"probability": "score"}), "score", threshold, truths)
        baseline_rows.append({"model": model, **result})
    baseline_frame = pd.DataFrame(baseline_rows)
    target_sensitivity = float(baseline_frame.sensitivity.max())
    eligible = baseline_frame[baseline_frame.sensitivity >= target_sensitivity - 0.03 - 1e-12]
    if eligible.empty:
        eligible = baseline_frame.loc[[int(baseline_frame.sensitivity.idxmax())]]
    baseline = eligible.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0].to_dict()
    candidates: list[dict[str, object]] = []
    for weights in _weight_grid():
        combined = _combine_tables(tables, weights)
        # Add quantiles so the grid is dense where the score distribution lives,
        # while retaining each frozen single-model operating threshold.
        score_values = combined.score.to_numpy(float)
        thresholds = sorted(
            set(float(metrics[model]["selected_event_operating_point"]["threshold"]) for model in MODELS)
            | set(float(x) for x in np.quantile(score_values, QUANTILE_LEVELS))
        )
        for threshold in thresholds:
            result = _evaluate(combined, "score", threshold, truths)
            if result["sensitivity"] + 1e-12 < target_sensitivity - 0.03:
                continue
            candidates.append({"weight_singlem": weights[0], "weight_cbramod": weights[1], "weight_tcn_gat": weights[2], **result})
    if not candidates:
        oracle = {"sensitivity": 0.0, "fa_per_24h": math.inf, "threshold": math.nan}
    else:
        oracle = sorted(candidates, key=lambda row: (float(row["fa_per_24h"]), -float(row["sensitivity"]))) [0]
    baseline_fa = float(baseline["fa_per_24h"])
    oracle_fa = float(oracle["fa_per_24h"])
    reduction = float((baseline_fa - oracle_fa) / baseline_fa) if baseline_fa > 0 and math.isfinite(oracle_fa) else 0.0
    result = {
        "baseline_best_single": baseline,
        "baseline_target_sensitivity": target_sensitivity,
        "oracle": oracle,
        "oracle_fa_reduction": reduction,
        "oracle_sensitivity_loss_vs_baseline": float(baseline["sensitivity"] - oracle["sensitivity"]),
        "weight_grid_step": WEIGHT_STEP,
        "threshold_grid_count_per_weight": len(thresholds),
        "oracle_definition": "validation-only convex-combination grid; frozen single-model thresholds plus six score quantiles; candidates constrained to within 3pp of the highest validation single-model sensitivity",
    }
    return result, {"baseline_rows": baseline_rows, "candidate_count": len(candidates)}


def _h2_shared_phenotype_gate() -> tuple[bool, dict[str, object]]:
    profiles = pd.read_csv(REPORT_DIR / "h2_cluster_profiles.csv")
    validation = profiles[(profiles.partition == "validation") & profiles.model.str.contains("__")].copy()
    similarities = validation.patient_median_similarity_js.dropna().astype(float).tolist()
    # The compact protocol requires more than high pairwise similarity: at
    # least two stable interpretable phenotypes and multi-seed stability.  The
    # current H2 artifact is explicitly exploratory and contains neither a
    # stable-phenotype count nor ARI >= .60, so it cannot select suppressor.
    detail = {
        "status": "exploratory",
        "validation_pair_patient_median_js": similarities,
        "minimum_validation_pair_js": min(similarities) if similarities else None,
        "required_stable_phenotypes": 2,
        "required_cross_seed_ari": 0.60,
        "stable_phenotype_count_available": False,
        "shared_phenotype_gate": False,
        "reason": "high validation profile similarity alone is insufficient; the frozen H2 result has no validated stable-phenotype/ARI evidence",
    }
    return False, detail


def main() -> None:
    started = time.time()
    events = pd.read_parquet(EVENT_PATH)
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    seizures.attrs["recording_durations"] = durations
    run_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    jaccard_rows: list[dict[str, object]] = []
    all_tables: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    all_metrics: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    for namespace, split, seeds in RUNS:
        for seed in seeds:
            tables: dict[str, pd.DataFrame] = {}
            metrics: dict[str, dict[str, object]] = {}
            for model in MODELS:
                table, model_metrics = _load_validation_run(namespace, split, seed, model)
                tables[model] = table
                metrics[model] = model_metrics
                op = model_metrics["selected_event_operating_point"]
                run_rows.append({
                    "split": split,
                    "seed": seed,
                    "model": model,
                    "validation_threshold": float(op["threshold"]),
                    "validation_sensitivity": float(op["event_sensitivity"]),
                    "validation_fa_per_24h": float(op["fa_per_24h"]),
                    "validation_probability_sha256": model_metrics["probability_sha256"],
                })
            all_tables[(split, seed)] = tables
            all_metrics[(split, seed)] = metrics
            for left, right in itertools.combinations(MODELS, 2):
                jaccard = _pair_jaccard(events, split, seed, left, right)
                jaccard_rows.append({"split": split, "seed": seed, "left_model": left, "right_model": right, **jaccard})
            oracle, extra = _oracle_for_group(tables, metrics, seizures)
            oracle_rows.append({"split": split, "seed": seed, **oracle, **extra})
            print(f"completed validation gate split={split} seed={seed} oracle_fa_reduction={oracle['oracle_fa_reduction']:.3f}", flush=True)
    jaccard_frame = pd.DataFrame(jaccard_rows)
    oracle_frame = pd.DataFrame(oracle_rows)
    min_jaccard = float(jaccard_frame.fp_jaccard.min()) if not jaccard_frame.empty else math.inf
    pair_pass = bool(min_jaccard <= 0.50)
    oracle_pass_rows = oracle_frame[
        (oracle_frame.oracle_fa_reduction >= 0.25)
        & (oracle_frame.oracle_sensitivity_loss_vs_baseline <= 0.03 + 1e-12)
    ]
    oracle_pass = bool(len(oracle_pass_rows) > 0)
    features = pd.read_csv(FEATURE_DICT)
    feature_rows = features.to_dict(orient="records")
    as_bool = lambda series: series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
    h4_allowed = as_bool(features.h4_allowed)
    available_by_event_end = as_bool(features.available_by_event_end)
    retrospective_only = as_bool(features.retrospective_only)
    feature_pass = bool(
        len(features) > 0
        and h4_allowed.all()
        and available_by_event_end.all()
        and (~retrospective_only).all()
    )
    h2_pass, h2_detail = _h2_shared_phenotype_gate()
    if pair_pass and oracle_pass and feature_pass and not h2_pass:
        branch = "router"
    elif h2_pass:
        branch = "suppressor"
    else:
        branch = "negative_result"
    input_hashes = {
        "validation_event_table_sha256": _sha256(EVENT_PATH),
        "test_event_table_sha256_frozen_but_not_read_for_gate": json.loads((REPORT_DIR / "event_table_manifest.json").read_text())["event_table_sha256"],
        "h2_summary_sha256": _sha256(REPORT_DIR / "h2_h3_summary.json"),
        "h2_cluster_profiles_sha256": _sha256(REPORT_DIR / "h2_cluster_profiles.csv"),
        "h3_fingerprint_sha256": _sha256(REPORT_DIR / "h3_model_fingerprint.csv"),
        "feature_dictionary_sha256": _sha256(FEATURE_DICT),
        "split17_manifest_sha256": _sha256(ROOT / "manifests" / "splits" / "split_17.json"),
        "split42_manifest_sha256": _sha256(ROOT / "manifests" / "splits" / "split_42.json"),
    }
    gates = {
        "release_id": "v3-compact-extension-v1",
        "gate": "H4_functional_router_gate",
        "status": "frozen",
        "branch": branch,
        "decision_partition": "validation_only",
        "test_used_for_branch_selection": False,
        "inputs": input_hashes,
        "feature_dictionary": feature_rows,
        "feature_availability_pass": feature_pass,
        "fp_jaccard_gate": {
            "threshold": 0.50,
            "minimum_validation_jaccard": min_jaccard,
            "pass": pair_pass,
            "pairs": jaccard_rows,
        },
        "oracle_gate": {
            "fa_reduction_threshold": 0.25,
            "sensitivity_loss_threshold": 0.03,
            "pass": oracle_pass,
            "passing_groups": [{"split": int(row.split), "seed": int(row.seed)} for row in oracle_pass_rows.itertuples(index=False)],
            "groups": oracle_rows,
        },
        "h2_shared_phenotype_gate": h2_detail,
        "interpretation": "router candidate only; no router test result is claimed by this gate artifact",
        "created_utc": "2026-08-11T00:00:00Z",
        "runtime_seconds": float(time.time() - started),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(REPORT_DIR / "h4_validation_single_model.csv", index=False)
    jaccard_frame.to_csv(REPORT_DIR / "h4_validation_fp_jaccard.csv", index=False)
    oracle_frame.to_csv(REPORT_DIR / "h4_validation_oracle.csv", index=False)
    (REPORT_DIR / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"branch": branch, "pair_pass": pair_pass, "oracle_pass": oracle_pass, "feature_pass": feature_pass, "h2_shared_phenotype_pass": h2_pass, "minimum_jaccard": min_jaccard, "runtime_seconds": time.time() - started}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
