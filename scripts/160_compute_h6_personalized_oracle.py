"""Estimate the headroom for a patient-personalized TUSZ router.

This is an audit-only analysis. It does not train a router, alter expert
thresholds, or use raw EEG. For every patient, decision windows are ordered
deterministically by recording path and within-recording time. The first
adaptation_fraction is treated as patient history, the last 20% as a
chronological holdout, and the middle portion is kept as an unused validation
buffer. The analysis compares fixed experts selected from the history with
two non-deployable ceilings:

* patient_oracle_fixed: one expert chosen with perfect knowledge of the
  holdout outcome for that patient;
* interval_oracle: a different expert may be chosen for every holdout
  60-second action interval using the offline future-competence labels.

The output is intentionally a new namespace and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_NAMESPACE = "h6-tusz-raw-pre-router-v1"
NAMESPACE = "h6-personalized-oracle-ceiling-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ("singlem", "cbramod", "tcn_gat")
HOLDOUT_FRACTION = 0.20
ADAPTATION_FRACTIONS = (0.50, 0.60, 0.70)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scorer():
    path = ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py"
    spec = importlib.util.spec_from_file_location("h6_scorer_for_personalized_oracle", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def truths_for_record(record: dict[str, object]) -> list[tuple[float, float]]:
    # Match the frozen H6 scorer's 70-second warm-up convention.
    return [
        (max(70.0, float(event["start_s"])), float(event["end_s"]))
        for event in record["seizure_events"]
        if float(event["end_s"]) > 70.0
    ]


def stage_assignments(labels: pd.DataFrame, adaptation_fraction: float) -> pd.DataFrame:
    """Assign adaptation/validation/holdout without mixing patient histories."""
    if not 0.0 < adaptation_fraction < 1.0 - HOLDOUT_FRACTION:
        raise ValueError(adaptation_fraction)
    frame = labels.copy()
    frame["stage"] = ""
    frame["patient_order"] = -1
    for patient, group in frame.groupby("patient_id", sort=False):
        order = group.sort_values(["recording", "future_start_s", "cache_row"], kind="mergesort").index
        n = len(order)
        holdout_n = max(1, int(np.ceil(n * HOLDOUT_FRACTION)))
        adapt_n = int(np.floor(n * adaptation_fraction))
        # Keep at least one validation decision window whenever possible.
        adapt_n = min(max(1, adapt_n), max(1, n - holdout_n - 1))
        holdout_start = n - holdout_n
        frame.loc[order, "patient_order"] = np.arange(n, dtype=int)
        frame.loc[order[:adapt_n], "stage"] = "adaptation"
        frame.loc[order[adapt_n:holdout_start], "stage"] = "validation_buffer"
        frame.loc[order[holdout_start:], "stage"] = "holdout"
        if not (frame.loc[order, "stage"] == "holdout").any():
            raise RuntimeError(f"no holdout for patient {patient}")
    return frame


def score_predictions(
    predictions: list[Event],
    truths: list[tuple[float, float]],
    intervals: list[tuple[float, float]],
    patient: str,
    recording: str,
    method: str,
    stage: str,
) -> dict[str, object]:
    predictions = sorted(predictions, key=lambda event: (event.start_s, event.end_s))
    matching = match_events(predictions, truths)
    delays = [
        max(0.0, float(predictions[pair.prediction_index].start_s) - float(truths[pair.truth_index][0]))
        for pair in matching.pairs
    ]
    seizure_seconds = sum(
        max(0.0, min(end, interval_end) - max(start, interval_start))
        for start, end in truths
        for interval_start, interval_end in intervals
        if end > interval_start and start < interval_end
    )
    observed_seconds = sum(end - start for start, end in intervals)
    return {
        "patient_id": patient,
        "recording": recording,
        "method": method,
        "stage": stage,
        "truth_events": int(len(truths)),
        "tp": int(len(matching.pairs)),
        "fp": int(len(matching.unmatched_predictions)),
        "fn": int(len(matching.unmatched_truths)),
        "delay_count": int(len(delays)),
        "delay_sum_s": float(sum(delays)),
        "nonseizure_s": float(max(0.0, observed_seconds - seizure_seconds)),
        "interval_seconds": float(observed_seconds),
    }


def score_segment(
    record: dict[str, object],
    event_cache: dict[str, dict[str, list[Event]]],
    rows: pd.DataFrame,
    model: str,
    method: str,
    stage: str,
) -> dict[str, object]:
    recording = str(record["relative_edf"])
    intervals = merge_intervals([(float(r.future_start_s), float(r.future_end_s)) for r in rows.itertuples(index=False)])
    if not intervals:
        return score_predictions([], [], [], str(record["patient_id"]), recording, method, stage)
    truths = [
        truth
        for truth in truths_for_record(record)
        if any(truth[1] > start and truth[0] < end for start, end in intervals)
    ]
    predictions = [
        event
        for event in event_cache[recording][model]
        if any(event.end_s > start and event.start_s < end for start, end in intervals)
    ]
    return score_predictions(predictions, truths, intervals, str(record["patient_id"]), recording, method, stage)


def dedupe_events(events: list[Event]) -> list[Event]:
    selected: dict[tuple[float, float], Event] = {}
    for event in events:
        key = (round(float(event.start_s), 3), round(float(event.end_s), 3))
        previous = selected.get(key)
        if previous is None or float(event.peak_probability) > float(previous.peak_probability):
            selected[key] = event
    return sorted(selected.values(), key=lambda event: (event.start_s, event.end_s))


def score_interval_oracle(
    record: dict[str, object],
    event_cache: dict[str, dict[str, list[Event]]],
    rows: pd.DataFrame,
) -> dict[str, object]:
    recording = str(record["relative_edf"])
    intervals = merge_intervals([(float(r.future_start_s), float(r.future_end_s)) for r in rows.itertuples(index=False)])
    truths = [
        truth
        for truth in truths_for_record(record)
        if any(truth[1] > start and truth[0] < end for start, end in intervals)
    ]
    selected: list[Event] = []
    for row in rows.sort_values(["future_start_s", "cache_row"]).itertuples(index=False):
        candidates = [model for model in MODELS if int(getattr(row, f"correct_{model}")) == 1]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda model: (float(getattr(row, f"target_weight_{model}")), model))
        start, end = float(row.future_start_s), float(row.future_end_s)
        selected.extend(
            event
            for event in event_cache[recording][chosen]
            if event.end_s > start and event.start_s < end
        )
    return score_predictions(
        dedupe_events(selected), truths, intervals, str(record["patient_id"]), recording, "interval_oracle", "holdout"
    )


def aggregate(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {
            "patients": 0,
            "records": 0,
            "truth_events": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sensitivity": np.nan,
            "fa_per_24h": np.nan,
            "mean_delay_s": np.nan,
            "patient_mean_sensitivity": np.nan,
            "patient_median_sensitivity": np.nan,
            "patient_mean_fa_per_24h": np.nan,
            "patient_median_fa_per_24h": np.nan,
        }
    grouped = (
        rows.groupby("patient_id", sort=True)
        .agg(truth_events=("truth_events", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), delay_count=("delay_count", "sum"), delay_sum_s=("delay_sum_s", "sum"), nonseizure_s=("nonseizure_s", "sum"))
        .reset_index()
    )
    grouped["sensitivity"] = grouped["tp"] / grouped["truth_events"].replace(0, np.nan)
    grouped["fa_per_24h"] = grouped["fp"] / (grouped["nonseizure_s"] / 86400.0).replace(0, np.nan)
    return {
        "patients": int(grouped.patient_id.nunique()),
        "records": int(len(rows)),
        "truth_events": int(rows.truth_events.sum()),
        "tp": int(rows.tp.sum()),
        "fp": int(rows.fp.sum()),
        "fn": int(rows.fn.sum()),
        "sensitivity": float(rows.tp.sum() / rows.truth_events.sum()) if rows.truth_events.sum() else np.nan,
        "fa_per_24h": float(rows.fp.sum() / (rows.nonseizure_s.sum() / 86400.0)) if rows.nonseizure_s.sum() > 0 else np.nan,
        "mean_delay_s": float(rows.delay_sum_s.sum() / rows.delay_count.sum()) if rows.delay_count.sum() else np.nan,
        "patient_mean_sensitivity": float(grouped.sensitivity.mean()),
        "patient_median_sensitivity": float(grouped.sensitivity.median()),
        "patient_mean_fa_per_24h": float(grouped.fa_per_24h.mean()),
        "patient_median_fa_per_24h": float(grouped.fa_per_24h.median()),
    }


def choose_safe(metrics: dict[str, dict[str, float | int]]) -> str:
    """Choose lowest FA among models within 3 pp of the best sensitivity."""
    best_sensitivity = max(float(metrics[m]["sensitivity"]) if np.isfinite(metrics[m]["sensitivity"]) else -np.inf for m in MODELS)
    safe = [m for m in MODELS if np.isfinite(metrics[m]["sensitivity"]) and float(metrics[m]["sensitivity"]) >= best_sensitivity - 0.03]
    if not safe:
        safe = list(MODELS)
    return sorted(
        safe,
        key=lambda m: (
            float(metrics[m]["fa_per_24h"]) if np.isfinite(metrics[m]["fa_per_24h"]) else np.inf,
            -float(metrics[m]["sensitivity"]) if np.isfinite(metrics[m]["sensitivity"]) else np.inf,
            m,
        ),
    )[0]


def run_partition(partition: str, adaptation_fraction: float, scorer) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source = ROOT / "outputs/reports" / SOURCE_NAMESPACE
    labels_path = source / "labels" / f"{partition}_formal_fold0_seed17.parquet"
    manifest_path = source / "manifests" / f"{partition}_manifest.json"
    expert_root = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1" / partition
    labels = pd.read_parquet(labels_path)
    manifest = json.loads(manifest_path.read_text())
    event_cache, thresholds, _ = scorer._read_event_cache(manifest, expert_root)
    assigned = stage_assignments(labels, adaptation_fraction)
    record_by_path = {str(record["relative_edf"]): record for record in manifest["records"]}

    adaptation_records: list[dict[str, object]] = []
    holdout_records: list[dict[str, object]] = []
    patient_choices: list[dict[str, object]] = []
    adapt_by_patient_model: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    adapt_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    rows_by_record = {str(recording): group for recording, group in assigned.groupby("recording", sort=False)}

    for recording, group in rows_by_record.items():
        record = record_by_path[recording]
        adapt = group[group.stage == "adaptation"]
        for model in MODELS:
            scored = score_segment(record, event_cache, adapt, model, f"fixed_{model}", "adaptation")
            adaptation_records.append(scored)
            adapt_by_patient_model[(str(record["patient_id"]), model)].append(scored)
            adapt_by_model[model].append(scored)

    adapt_metric_by_model = {model: aggregate(pd.DataFrame(rows)) for model, rows in adapt_by_model.items()}
    global_best = choose_safe(adapt_metric_by_model)
    patient_best: dict[str, str] = {}
    for patient in sorted(assigned.patient_id.unique()):
        patient_metrics = {
            model: aggregate(pd.DataFrame(adapt_by_patient_model.get((str(patient), model), [])))
            for model in MODELS
        }
        patient_best[str(patient)] = choose_safe(patient_metrics)
        patient_choices.append(
            {
                "partition": partition,
                "adaptation_fraction": adaptation_fraction,
                "patient_id": str(patient),
                "history_best_fixed": patient_best[str(patient)],
                "global_best_fixed": global_best,
                "history_truth_events": int(sum(int(r["truth_events"]) for model in MODELS for r in adapt_by_patient_model.get((str(patient), model), []))),
            }
        )

    # Evaluate the final 20% with no selection based on its outcome, except for
    # the explicitly labelled oracle ceilings.  The patient oracle must choose
    # one expert for the entire patient, so first collect all fixed holdout
    # scores and only then make one choice per patient.
    fixed_holdout_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    holdout_groups: dict[str, list[tuple[dict[str, object], pd.DataFrame]]] = defaultdict(list)
    for recording, group in rows_by_record.items():
        record = record_by_path[recording]
        holdout = group[group.stage == "holdout"]
        if holdout.empty:
            continue
        holdout_groups[str(record["patient_id"])].append((record, holdout))
        for model in MODELS:
            scored = score_segment(record, event_cache, holdout, model, f"fixed_{model}", "holdout")
            holdout_records.append(scored)
            fixed_holdout_records[(str(record["patient_id"]), model)].append(scored)
        holdout_records.append(score_segment(record, event_cache, holdout, global_best, "global_best_fixed", "holdout"))
        holdout_records.append(score_segment(record, event_cache, holdout, patient_best[str(record["patient_id"])], "history_best_fixed", "holdout"))
        holdout_records.append(score_interval_oracle(record, event_cache, holdout))

    patient_oracle_choice: dict[str, str] = {}
    for patient, _groups in holdout_groups.items():
        patient_metrics = {
            model: aggregate(pd.DataFrame(fixed_holdout_records.get((patient, model), [])))
            for model in MODELS
        }
        patient_oracle_choice[patient] = choose_safe(patient_metrics)
    for patient, groups in holdout_groups.items():
        chosen = patient_oracle_choice[patient]
        for record, holdout in groups:
            holdout_records.append(score_segment(record, event_cache, holdout, chosen, "patient_oracle_fixed", "holdout"))
    for row in patient_choices:
        patient = str(row["patient_id"])
        row["patient_oracle_fixed"] = patient_oracle_choice.get(patient)

    methods = [f"fixed_{model}" for model in MODELS] + ["global_best_fixed", "history_best_fixed", "patient_oracle_fixed", "interval_oracle"]
    summary_rows: list[dict[str, object]] = []
    holdout_frame = pd.DataFrame(holdout_records)
    for method in methods:
        metrics = aggregate(holdout_frame[holdout_frame.method == method].copy())
        summary_rows.append({"partition": partition, "adaptation_fraction": adaptation_fraction, "method": method, **metrics})
    for row in patient_choices:
        row["patient_oracle_definition"] = "holdout outcome used only for upper-bound calculation"
    audit = {
        "partition": partition,
        "adaptation_fraction": adaptation_fraction,
        "holdout_fraction": HOLDOUT_FRACTION,
        "patient_count": int(assigned.patient_id.nunique()),
        "decision_rows": int(len(assigned)),
        "adaptation_rows": int((assigned.stage == "adaptation").sum()),
        "validation_buffer_rows": int((assigned.stage == "validation_buffer").sum()),
        "holdout_rows": int((assigned.stage == "holdout").sum()),
        "global_best_fixed_from_adaptation": global_best,
        "thresholds": thresholds,
        "threshold_source": "frozen expert metadata; no retuning",
        "event_cache_source": str(expert_root.relative_to(ROOT)),
        "test_labels_used_for_oracle": True,
        "model_training": False,
        "selection_on_holdout": "patient_oracle_fixed and interval_oracle only; explicitly non-deployable",
        "ordering": "recording path lexicographic, then future_start_s, then cache_row",
    }
    return summary_rows, holdout_records, patient_choices, {"assigned": assigned, "audit": audit, "labels_path": labels_path, "manifest_path": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", default="dev,eval")
    parser.add_argument("--fractions", default="0.5,0.6,0.7")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = (args.output_dir or OUT).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing namespace: {output_dir}")
    partitions = [value.strip() for value in args.partitions.split(",") if value.strip()]
    fractions = tuple(float(value) for value in args.fractions.split(",") if value.strip())
    if any(value not in ADAPTATION_FRACTIONS for value in fractions):
        raise ValueError(f"fractions must be from {ADAPTATION_FRACTIONS}: {fractions}")
    scorer = load_scorer()
    summaries: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    assignments: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    for partition in partitions:
        for fraction in fractions:
            result_summary, result_holdout, result_choices, metadata = run_partition(partition, fraction, scorer)
            summaries.extend(result_summary)
            holdout_rows.extend(
                [{**row, "partition": partition, "adaptation_fraction": fraction} for row in result_holdout]
            )
            choices.extend(result_choices)
            assigned = metadata["assigned"].copy()
            assigned["adaptation_fraction"] = fraction
            assignments.append(assigned[["sample_id", "partition", "patient_id", "recording", "future_start_s", "future_end_s", "stage", "patient_order", "adaptation_fraction"]])
            audits.append(metadata["audit"])
            for path in (metadata["labels_path"], metadata["manifest_path"]):
                input_hashes[str(path.relative_to(ROOT))] = sha256(path)
            print(f"PERSONALIZED_ORACLE_DONE partition={partition} adaptation_fraction={fraction}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(output_dir / "method_summary.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(output_dir / "holdout_record_metrics.csv", index=False)
    pd.DataFrame(choices).to_csv(output_dir / "patient_choices.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(output_dir / "split_assignments.csv", index=False)
    (output_dir / "partition_audits.json").write_text(json.dumps(audits, indent=2, sort_keys=True, default=float) + "\n")
    summary_payload = {
        "release_id": NAMESPACE,
        "source_release": SOURCE_NAMESPACE,
        "status": "audit_only_non_deployable",
        "dataset": "TUSZ_v2.0.6 official train/dev/eval; frozen expert outputs",
        "partitions": partitions,
        "adaptation_fractions": fractions,
        "holdout_fraction": HOLDOUT_FRACTION,
        "protocol": "per-patient chronological adaptation history, validation buffer, final holdout",
        "methods": {
            "history_best_fixed": "one expert selected per patient from adaptation history under sensitivity-within-3pp then lowest-FA rule",
            "global_best_fixed": "one expert selected from pooled adaptation history under the same rule",
            "patient_oracle_fixed": "one expert selected with holdout outcome; upper bound only",
            "interval_oracle": "expert selected per holdout 60-second action interval from offline correctness labels; upper bound only",
        },
        "test_labels_used_for_selection": True,
        "threshold_retuning": False,
        "model_training": False,
        "raw_eeg_used": False,
        "source_hashes": input_hashes,
        "analysis_script_sha256": sha256(ROOT / "scripts/160_compute_h6_personalized_oracle.py"),
        "outputs": {
            "method_summary": "method_summary.csv",
            "holdout_record_metrics": "holdout_record_metrics.csv",
            "patient_choices": "patient_choices.csv",
            "split_assignments": "split_assignments.csv",
            "partition_audits": "partition_audits.json",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True, default=float) + "\n")
    manifest = {
        "release_id": NAMESPACE,
        "status": "complete",
        "summary_sha256": sha256(output_dir / "summary.json"),
        "method_summary_sha256": sha256(output_dir / "method_summary.csv"),
        "holdout_record_metrics_sha256": sha256(output_dir / "holdout_record_metrics.csv"),
        "patient_choices_sha256": sha256(output_dir / "patient_choices.csv"),
        "split_assignments_sha256": sha256(output_dir / "split_assignments.csv"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.sha256").write_text(sha256(output_dir / "manifest.json") + "\n")
    print(json.dumps(summary_payload, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
