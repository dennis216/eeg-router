"""Corrected phase-intersection scoring for completed H7 pilot checkpoints.

The first immediate diagnostic scorer treated the patient-level phase length as
if it belonged to every EDF.  This script preserves those outputs and performs
a read-only re-score with the actual intersection of each EDF and the global
training/validation/test interval.  It does not retrain, select from test, or
overwrite the earlier diagnostic JSON files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root/b_false_alarm_atlas")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corrected_truths_for(record: dict[str, Any], offset: float, bounds: tuple[float, float]) -> tuple[list[tuple[float, float]], float]:
    """Return only the part of one EDF that intersects the global phase."""
    lo, hi = bounds
    duration = float(record["duration_s"])
    rec_lo, rec_hi = float(offset), float(offset) + duration
    overlap_lo, overlap_hi = max(lo, rec_lo), min(hi, rec_hi)
    if overlap_hi <= overlap_lo:
        return [], 0.0
    local_lo, local_hi = overlap_lo - float(offset), overlap_hi - float(offset)
    intervals: list[tuple[float, float]] = []
    for event in record.get("seizure_events", []):
        start, end = float(event["start_s"]), float(event["end_s"])
        center = float(offset) + (start + end) / 2.0
        if lo <= center < hi:
            clipped_start = max(start, local_lo)
            clipped_end = min(end, local_hi)
            if clipped_end > clipped_start:
                intervals.append((clipped_start, clipped_end))
    covered = 0.0
    for start, end in sorted(intervals):
        if not covered:
            covered = end - start
            previous_end = end
        elif start > previous_end:
            covered += end - start
            previous_end = end
        else:
            covered += max(0.0, end - previous_end)
            previous_end = max(previous_end, end)
    return intervals, max(0.0, overlap_hi - overlap_lo - covered) / 3600.0


def corrected_score_table(scorer, table: pd.DataFrame, records: dict[str, dict[str, Any]], partition: str, patient: str, phase_name: str, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    offsets = scorer.recording_global_offset(records, partition, patient)
    bounds = scorer.phase_bounds(records, partition, patient)[phase_name]
    detail: list[dict[str, Any]] = []
    tp = fp = truth_n = 0
    nonseizure_h = 0.0
    for recording, group in table.groupby("recording", sort=True):
        ordered = group.sort_values("context_end_s", kind="stable")
        if len(ordered) < 2:
            continue
        record = records[str(recording)]
        truths, hours = corrected_truths_for(record, offsets[str(recording)], bounds)
        predictions = scorer.eventize(ordered.context_end_s.to_numpy(float), ordered.probability.to_numpy(float), threshold=threshold)
        matched = scorer.match_events(predictions, truths, minimum_overlap_s=1.0)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        truth_n += len(truths)
        nonseizure_h += hours
        detail.append({
            "recording": recording,
            "patient_id": patient,
            "phase": phase_name,
            "truth_events": len(truths),
            "true_positive_events": len(matched.pairs),
            "false_alarm_events": len(matched.unmatched_predictions),
            "nonseizure_hours": hours,
        })
    metric = {
        "threshold": float(threshold),
        "true_positive_events": int(tp),
        "false_alarm_events": int(fp),
        "truth_events": int(truth_n),
        "event_sensitivity": float(tp / truth_n) if truth_n else 1.0,
        "nonseizure_hours": float(nonseizure_h),
        "fa_per_24h": float(fp * 24.0 / nonseizure_h) if nonseizure_h > 0 else float("nan"),
    }
    return metric, pd.DataFrame(detail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    out_root = ROOT / "outputs/reports" / args.namespace / "per_run_evaluations"
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Refusing overwrite of non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    scorer = load_module("h7_corrected_scorer", ROOT / "scripts/200_h7_score_and_gate.py")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = scorer.source_records()
    run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir() and (p / "checkpoint.pt").exists())
    if not run_dirs:
        raise RuntimeError(f"No completed checkpoints under {run_root}")
    summary_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        parts = run_dir.name.split("__")
        if len(parts) != 5 or not parts[-1].startswith("seed"):
            continue
        model, method, partition, patient, seed_text = parts
        seed = int(seed_text.removeprefix("seed"))
        validation = scorer.predict_run(run_dir, "dev", device, batch_size=args.batch_size)
        test = scorer.predict_run(run_dir, "holdout", device, batch_size=args.batch_size)
        source = scorer.source_checkpoint(model)
        frozen_validation = scorer.predict_unit(model, "head_only", partition, patient, "dev", device, source, batch_size=args.batch_size, output_method="frozen_original", seed=seed)
        frozen_metric, _ = corrected_score_table(scorer, frozen_validation, records, partition, patient, "dev", scorer.source_threshold(model))

        candidates: list[dict[str, Any]] = []
        for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
            metric, _ = corrected_score_table(scorer, validation, records, partition, patient, "dev", float(threshold))
            if metric["event_sensitivity"] >= frozen_metric["event_sensitivity"] - 0.03:
                candidates.append(metric)
        selected = min(candidates, key=lambda row: (row["fa_per_24h"], -row["threshold"])) if candidates else None
        threshold = float(selected["threshold"] if selected is not None else scorer.source_threshold(model))
        validation_metric, _ = corrected_score_table(scorer, validation, records, partition, patient, "dev", threshold)
        test_metric, _ = corrected_score_table(scorer, test, records, partition, patient, "holdout", threshold)
        payload = {
            "release_id": args.namespace,
            "run": run_dir.name,
            "model": model,
            "method": method,
            "partition": partition,
            "patient_id": patient,
            "seed": seed,
            "evaluation_type": "corrected_phase_intersection_rescore",
            "correction": "phase duration is counted only on the intersection between each EDF and the patient-level phase interval",
            "prior_immediate_diagnostic_invalid_for_fa24": True,
            "checkpoint": str((run_dir / "checkpoint.pt").relative_to(ROOT)),
            "checkpoint_sha256": sha256(run_dir / "checkpoint.pt"),
            "phase_names": {"fit": "training_set", "dev": "validation_set", "holdout": "test_set"},
            "threshold": threshold,
            "threshold_source": "validation_set_only" if selected is not None else "frozen_source_validation_only_fallback",
            "test_used_for_selection": False,
            "test_evaluation_count_in_corrected_release": 1,
            "frozen_validation": frozen_metric,
            "validation": validation_metric,
            "test": test_metric,
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "created_utc": pd.Timestamp.utcnow().isoformat(),
        }
        out = out_root / f"{run_dir.name}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        summary_rows.append({
            "run": run_dir.name,
            "model": model,
            "method": method,
            "patient_id": patient,
            "threshold": threshold,
            "frozen_validation_sensitivity": frozen_metric["event_sensitivity"],
            "validation_sensitivity": validation_metric["event_sensitivity"],
            "validation_fa24": validation_metric["fa_per_24h"],
            "test_sensitivity": test_metric["event_sensitivity"],
            "test_fa24": test_metric["fa_per_24h"],
            "test_tp": test_metric["true_positive_events"],
            "test_fp": test_metric["false_alarm_events"],
            "test_truth": test_metric["truth_events"],
            "test_nonseizure_hours": test_metric["nonseizure_hours"],
        })
        print(json.dumps({"run": run_dir.name, "status": "complete", "validation": validation_metric, "test": test_metric}, sort_keys=True), flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root.parent / "corrected_summary.csv", index=False)
    manifest = {
        "release_id": args.namespace,
        "status": "completed_corrected_rescore",
        "source_run_root": str(run_root.relative_to(ROOT)),
        "completed_runs": int(len(summary)),
        "source_code": "scripts/211_rescore_h7_completed_runs_corrected.py",
        "test_used_for_selection": False,
        "prior_scores_preserved": True,
        "correction": "EDF-phase intersection prevents repeating the global phase duration once per EDF",
        "summary_sha256": sha256(out_root.parent / "corrected_summary.csv"),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (out_root.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
