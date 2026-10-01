"""Development-only H4 protocol and work-point diagnostics.

This script does not train a detector or router.  It reuses the frozen H4
candidate events, validation probabilities, test probabilities, and router
artifacts.  Thresholds are selected from validation only.  The old release is
never overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_ID = "v3-groupkfold-confirmatory-v1"
ANALYSIS_ID = "h4-router-revision-v1"
SOURCE = ROOT / "outputs" / "reports" / SOURCE_ID
OUT = ROOT / "outputs" / "reports" / ANALYSIS_ID
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = range(5)
SEEDS = (17, 42, 3407)
SENSITIVITY_MARGIN = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_module():
    path = ROOT / "scripts" / "84_run_groupkfold_h4_router.py"
    spec = importlib.util.spec_from_file_location("h4_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(row[0], row[1]) for row in merged]


def interval_length(intervals: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in merge_intervals(intervals)))


def intersection_length(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return float(total)


def partition_metadata(
    fold: int,
    partition: str,
    recordings: pd.DataFrame,
    probability_table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, str]]:
    split = json.loads((ROOT / "manifests" / "groupkfold_cv_v1" / f"fold_{fold}.json").read_text())
    patient_ids = {str(value) for value in split[partition]}
    selected = recordings[recordings.patient_id.astype(str).isin(patient_ids)].copy()
    first_prediction_end = (
        probability_table.groupby("recording", sort=False).end.min().astype(float).to_dict()
    )
    missing = sorted(set(selected.recording_id.astype(str)) - set(map(str, first_prediction_end)))
    if missing:
        raise RuntimeError(
            f"fold={fold} partition={partition} has {len(missing)} recordings without prediction windows"
        )
    starts = {str(row.recording_id): float(first_prediction_end[str(row.recording_id)]) for row in selected.itertuples(index=False)}
    patients = {str(row.recording_id): str(row.patient_id) for row in selected.itertuples(index=False)}
    return starts, patients


def truths(
    seizures: pd.DataFrame, recording: str, evaluation_start: float, duration: float
) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str) == str(recording)]
    return merge_intervals(
        [
            (max(evaluation_start, float(row.start_s)), min(duration, float(row.end_s)))
            for row in rows.itertuples(index=False)
            if float(row.end_s) > evaluation_start and float(row.start_s) < duration
        ]
    )


def score_events(
    events: pd.DataFrame,
    starts: dict[str, float],
    patients: dict[str, str],
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> dict[str, float]:
    tp = fp = truth_count = 0
    nonseizure_seconds = valid_seconds = alarm_seconds = false_alarm_seconds = 0.0
    delays: list[float] = []
    for recording in patients:
        start = float(starts[recording])
        duration = float(durations[recording])
        truth = truths(seizures, recording, start, duration)
        group = events[events.recording.astype(str) == recording] if not events.empty else events
        predictions: list[Event] = []
        intervals: list[tuple[float, float]] = []
        if not group.empty:
            for row in group.sort_values("event_start", kind="stable").itertuples(index=False):
                left = max(start, float(row.event_start))
                right = min(duration, float(row.event_end))
                if right <= left:
                    continue
                predictions.append(Event(left, right, float(row.peak_probability)))
                intervals.append((left, right))
        matched = match_events(predictions, truth)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        truth_count += len(truth)
        delays.extend(
            max(0.0, predictions[p.prediction_index].start_s - truth[p.truth_index][0])
            for p in matched.pairs
        )
        valid = max(0.0, duration - start)
        truth_s = interval_length(truth)
        alarm_s = interval_length(intervals)
        overlap_s = intersection_length(intervals, truth)
        valid_seconds += valid
        nonseizure_seconds += max(0.0, valid - truth_s)
        alarm_seconds += alarm_s
        false_alarm_seconds += max(0.0, alarm_s - overlap_s)
    return {
        "true_positive_events": int(tp),
        "false_alarm_events": int(fp),
        "truth_events": int(truth_count),
        "sensitivity": float(tp / truth_count) if truth_count else math.nan,
        "fa_per_24h": float(fp * 86400.0 / nonseizure_seconds) if nonseizure_seconds else math.inf,
        "false_alarm_minutes_per_24h": float(false_alarm_seconds / valid_seconds * 1440.0) if valid_seconds else math.inf,
        "alarm_minutes_per_24h": float(alarm_seconds / valid_seconds * 1440.0) if valid_seconds else math.inf,
        "mean_delay_s": float(np.mean(delays)) if delays else math.nan,
        "valid_monitoring_hours": float(valid_seconds / 3600.0),
        "nonseizure_hours": float(nonseizure_seconds / 3600.0),
    }


def router_choices(candidates: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=["patient", "recording", "event_start", "event_end", "peak_probability", "router_score", "selected_model", "group_id"]
        )
    frame = candidates.copy()
    frame["router_score"] = np.asarray(scores, dtype=float)
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby("group_id", sort=True):
        row = group.sort_values(
            ["router_score", "peak_probability"], ascending=[False, False], kind="stable"
        ).iloc[0]
        rows.append(
            {
                "patient": str(row.patient),
                "recording": str(row.recording),
                "event_start": float(row.start_s),
                "event_end": float(row.end_s),
                "peak_probability": float(row.peak_probability),
                "router_score": float(row.router_score),
                "selected_model": str(row.source_model),
                "group_id": str(row.group_id),
            }
        )
    return pd.DataFrame(rows)


def router_select_own_interval(choices: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if choices.empty:
        return choices.copy()
    return choices[choices.router_score >= float(threshold)].copy()


def exact_thresholds(scores: np.ndarray) -> list[float]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("no finite validation router scores")
    quantile_values = np.quantile(values, np.linspace(0.0, 1.0, 129))
    candidates = set(float(value) for value in quantile_values)
    candidates.add(float(np.nextafter(values.min(), -np.inf)))
    candidates.add(float(np.nextafter(values.max(), np.inf)))
    return sorted(candidates)


def select_workpoint(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    target_sensitivity: float,
    starts: dict[str, float],
    patients: dict[str, str],
    seizures: pd.DataFrame,
    durations: dict[str, float],
) -> tuple[float | None, dict[str, float] | None, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    choices = router_choices(candidates, scores)
    for threshold in exact_thresholds(scores):
        events = router_select_own_interval(choices, threshold)
        rows.append({"threshold": threshold, **score_events(events, starts, patients, seizures, durations)})
    frame = pd.DataFrame(rows)
    eligible = frame[frame.sensitivity >= target_sensitivity - SENSITIVITY_MARGIN].copy()
    if eligible.empty:
        return None, None, frame
    ordered = eligible.sort_values(
        ["false_alarm_minutes_per_24h", "fa_per_24h", "mean_delay_s", "threshold"],
        ascending=[True, True, True, False],
        na_position="last",
        kind="stable",
    )
    row = ordered.iloc[0]
    return float(row.threshold), row.drop(labels=["threshold"]).to_dict(), frame


def checkpoint_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(
        ROOT.parent / "b_false_alarm_atlas" / "outputs" / "reports" / ANALYSIS_ID / "_missing.csv"
    ) if False else pd.read_csv(
        Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/reports/h4-final-linkage-expert-audit-v1/expert_validation_checkpoint_curves.csv")
    )
    chosen: list[dict[str, object]] = []
    for (model, fold, seed), group in curves.groupby(["model", "fold", "seed"], sort=True):
        ordered = group.sort_values(
            ["window_average_precision", "window_auroc", "step"],
            ascending=[False, False, True],
            kind="stable",
        )
        best = ordered.iloc[0]
        final = group.sort_values("step").iloc[-1]
        chosen.append(
            {
                "model": model,
                "fold": int(fold),
                "seed": int(seed),
                "selected_step_by_validation_ap": int(best.step),
                "selected_validation_ap": float(best.window_average_precision),
                "final_step": int(final.step),
                "final_validation_ap": float(final.window_average_precision),
                "ap_gain_vs_final": float(best.window_average_precision - final.window_average_precision),
                "final_is_selected": bool(int(best.step) == int(final.step)),
            }
        )
    selected = pd.DataFrame(chosen)
    summary = selected.groupby("model", as_index=False).agg(
        runs=("fold", "size"),
        final_selected_fraction=("final_is_selected", "mean"),
        mean_ap_gain_vs_final=("ap_gain_vs_final", "mean"),
        median_selected_step=("selected_step_by_validation_ap", "median"),
    )
    return selected, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    features = pd.read_parquet(SOURCE / "h4_groupkfold_event_features.parquet")
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    old_validation = pd.read_csv(SOURCE / "h4_groupkfold_router_validation.csv")
    old_test = pd.read_csv(SOURCE / "h4_groupkfold_router_test.csv")
    fixed_weights = pd.read_csv(SOURCE / "h4_groupkfold_validation_oracle.csv")

    corrected_rows: list[dict[str, object]] = []
    sweep_rows: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        for seed in SEEDS:
            fold_features = features[(features.split == fold) & (features.seed == seed)].copy()
            validation = fold_features[fold_features.partition == "validation"].copy()
            test = fold_features[fold_features.partition == "test"].copy()
            val_candidates = source._candidate_groups(validation)
            test_candidates = source._candidate_groups(test)
            val_tables = {model: source._load_probability_table(fold, seed, model, "validation") for model in MODELS}
            test_tables = {model: source._load_probability_table(fold, seed, model, "test") for model in MODELS}
            val_starts, val_patients = partition_metadata(fold, "validation", recordings, val_tables["cbramod"])
            test_starts, test_patients = partition_metadata(fold, "test", recordings, test_tables["cbramod"])
            metadata_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "validation_recordings": len(val_patients),
                    "test_recordings": len(test_patients),
                    "validation_start_min_s": min(val_starts.values()),
                    "validation_start_max_s": max(val_starts.values()),
                    "test_start_min_s": min(test_starts.values()),
                    "test_start_max_s": max(test_starts.values()),
                }
            )

            best_model_row = old_validation[
                (old_validation.fold == fold)
                & (old_validation.seed == seed)
                & (old_validation.method == "best_single")
            ].iloc[0]
            best_model = str(best_model_row.selected_model)
            baseline_val_events = source._event_rows_from_model(validation, best_model)
            baseline_test_events = source._event_rows_from_model(test, best_model)
            baseline_val = score_events(baseline_val_events, val_starts, val_patients, seizures, durations)
            baseline_test = score_events(baseline_test_events, test_starts, test_patients, seizures, durations)
            target = float(baseline_val["sensitivity"])
            corrected_rows.append(
                {"fold": fold, "seed": seed, "partition": "test", "method": "best_single_existing_selection", "selected_model": best_model, "workpoint_status": "existing_validation_selection", **baseline_test}
            )

            for method, aware in (("router_model_aware", True), ("router_no_model_id", False)):
                artifact = SOURCE / f"h4_{method}_fold{fold}_seed{seed}.joblib"
                model = joblib.load(artifact)
                val_matrix, _ = source._router_matrix(val_candidates, aware)
                test_matrix, _ = source._router_matrix(test_candidates, aware)
                val_scores = model.predict_proba(val_matrix)[:, 1]
                test_scores = model.predict_proba(test_matrix)[:, 1]
                threshold, val_score, sweep = select_workpoint(
                    val_candidates, val_scores, target, val_starts, val_patients, seizures, durations
                )
                sweep.insert(0, "seed", seed)
                sweep.insert(0, "fold", fold)
                sweep.insert(0, "method", method)
                sweep_rows.append(sweep)
                if threshold is None:
                    corrected_rows.append(
                        {"fold": fold, "seed": seed, "partition": "test", "method": method, "selected_model": best_model, "workpoint_status": "fallback_to_best_single_no_feasible_validation_router", **baseline_test}
                    )
                    decision_rows.append(
                        {"fold": fold, "seed": seed, "method": method, "feasible": False, "threshold": math.nan, "target_sensitivity": target, "fallback": "best_single"}
                    )
                    continue
                test_choices = router_choices(test_candidates, test_scores)
                test_events = router_select_own_interval(test_choices, threshold)
                test_score = score_events(test_events, test_starts, test_patients, seizures, durations)
                corrected_rows.append(
                    {"fold": fold, "seed": seed, "partition": "test", "method": method, "selected_model": "", "workpoint_status": "validation_feasible", "threshold": threshold, **test_score}
                )
                decision_rows.append(
                    {"fold": fold, "seed": seed, "method": method, "feasible": True, "threshold": threshold, "target_sensitivity": target, "validation_sensitivity": val_score["sensitivity"], "validation_fa_per_24h": val_score["fa_per_24h"], "validation_false_alarm_minutes_per_24h": val_score["false_alarm_minutes_per_24h"], "fallback": "none"}
                )

    corrected = pd.DataFrame(corrected_rows)
    decisions = pd.DataFrame(decision_rows)
    selected_checkpoints, checkpoint_summary = checkpoint_audit()
    corrected.to_csv(OUT / "corrected_test_metrics.csv", index=False)
    decisions.to_csv(OUT / "validation_workpoint_decisions.csv", index=False)
    pd.concat(sweep_rows, ignore_index=True).to_csv(OUT / "validation_router_threshold_sweeps.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(OUT / "evaluation_metadata_audit.csv", index=False)
    selected_checkpoints.to_csv(OUT / "validation_checkpoint_selection.csv", index=False)
    checkpoint_summary.to_csv(OUT / "validation_checkpoint_summary.csv", index=False)

    summary = corrected.groupby(["method", "workpoint_status"], as_index=False).agg(
        runs=("fold", "size"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_fa_per_24h=("fa_per_24h", "mean"),
        mean_false_alarm_minutes_per_24h=("false_alarm_minutes_per_24h", "mean"),
        mean_alarm_minutes_per_24h=("alarm_minutes_per_24h", "mean"),
        mean_delay_s=("mean_delay_s", "mean"),
    )
    summary.to_csv(OUT / "corrected_test_summary.csv", index=False)
    old_test.to_csv(OUT / "frozen_legacy_test_metrics_copy.csv", index=False)

    manifest = {
        "analysis_id": ANALYSIS_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_development_only_diagnostic",
        "detector_training_performed": False,
        "router_training_performed": False,
        "test_used_for_threshold_selection": False,
        "evaluation_start": "first available non-warmup prediction end for every split-manifest recording",
        "router_selected_interval": "selected member event start and end",
        "threshold_selection": "validation only; full-precision score quantiles; sensitivity gate; alarm-time then event-count objective",
        "infeasible_router_action": "fall back to validation-selected best single detector",
        "alarm_time": "union of predicted intervals; false alarm time is prediction union outside original seizure intervals",
        "source_files": {
            "event_features": sha256(SOURCE / "h4_groupkfold_event_features.parquet"),
            "legacy_validation": sha256(SOURCE / "h4_groupkfold_router_validation.csv"),
            "legacy_test": sha256(SOURCE / "h4_groupkfold_router_test.csv"),
            "recordings": sha256(ROOT / "manifests" / "recordings.parquet"),
            "seizures": sha256(ROOT / "manifests" / "seizures.parquet"),
        },
        "outputs": sorted(path.name for path in OUT.iterdir() if path.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(checkpoint_summary.to_string(index=False))


if __name__ == "__main__":
    main()
