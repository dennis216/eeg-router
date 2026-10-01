"""Score the completed standard GroupKFold TUSZ external inference.

All thresholds and checkpoints are frozen before this script runs.  TUSZ
annotations are read only for post-inference event scoring; they cannot change
the threshold, model, cohort, or selection.  The script reports per-run and
model-level sensitivity, false alarms per 24 hours, detection delay, and
patient-level bootstrap intervals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
EXTERNAL_ROOT = ROOT / "outputs" / "tuh_external" / "tusz_groupkfold_external_v1"
COHORT_MANIFEST = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
OUTPUT_RUNS = REPORT_DIR / "tusz_groupkfold_external_v1_run_metrics.csv"
OUTPUT_RECORDS = REPORT_DIR / "tusz_groupkfold_external_v1_record_metrics.parquet"
OUTPUT_EVENTS = REPORT_DIR / "tusz_groupkfold_external_v1_events.parquet"
OUTPUT_BOOTSTRAP = REPORT_DIR / "tusz_groupkfold_external_v1_patient_bootstrap.parquet"
OUTPUT_MODEL_SUMMARY = REPORT_DIR / "tusz_groupkfold_external_v1_model_summary.csv"
OUTPUT_MANIFEST = REPORT_DIR / "tusz_groupkfold_external_v1_scoring_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260823


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_record(record: dict[str, object], probability_path: Path, metadata_path: Path, model: str, fold: int, seed: int):
    table = pd.read_parquet(probability_path)
    metadata = json.loads(metadata_path.read_text())
    threshold = float(metadata["threshold"])
    if len(table) >= 2:
        predictions = eventize(
            table.window_end_s.to_numpy(dtype=float),
            table.probability.to_numpy(dtype=float),
            threshold=threshold,
        )
        first_prediction_end = float(table.window_end_s.iloc[0])
    else:
        predictions = []
        first_prediction_end = min(70.0, float(record["duration_s"]))
    truths = [
        (max(float(event["start_s"]), first_prediction_end), float(event["end_s"]))
        for event in record["seizure_events"]
        if float(event["end_s"]) > first_prediction_end
    ]
    matching = match_events(predictions, truths)
    seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
    nonseizure_seconds = max(0.0, float(record["duration_s"]) - first_prediction_end - seizure_seconds)
    delays = [predictions[pair.prediction_index].start_s - truths[pair.truth_index][0] for pair in matching.pairs]
    record_row = {
        "model": model,
        "fold": fold,
        "seed": seed,
        "patient_id": str(record["patient_id"]),
        "recording": str(record["relative_edf"]),
        "threshold": threshold,
        "threshold_source": str(metadata["threshold_source"]),
        "tp": len(matching.pairs),
        "fp": len(matching.unmatched_predictions),
        "fn": len(matching.unmatched_truths),
        "truth_events": len(truths),
        "nonseizure_s": nonseizure_seconds,
        "delay_count": len(delays),
        "delay_sum_s": float(sum(delays)) if delays else 0.0,
        "median_delay_s": float(np.median(delays)) if delays else np.nan,
        "probability_path": str(probability_path.relative_to(ROOT)),
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "checkpoint_update": int(metadata["checkpoint_update"]),
        "model_training": bool(metadata["model_training"]),
    }
    event_rows = []
    matched_by_prediction = {pair.prediction_index: pair for pair in matching.pairs}
    for index, event in enumerate(predictions):
        pair = matched_by_prediction.get(index)
        event_rows.append(
            {
                "model": model,
                "fold": fold,
                "seed": seed,
                "patient_id": str(record["patient_id"]),
                "recording": str(record["relative_edf"]),
                "event_index": index,
                "start_s": float(event.start_s),
                "end_s": float(event.end_s),
                "peak_probability": float(event.peak_probability),
                "matched_truth_index": int(pair.truth_index) if pair is not None else None,
                "overlap_s": float(pair.overlap_s) if pair is not None else 0.0,
                "is_true_positive": pair is not None,
            }
        )
    return record_row, event_rows


def _summarize_records(records: pd.DataFrame) -> dict[str, object]:
    tp = int(records.tp.sum())
    fp = int(records.fp.sum())
    truth_events = int(records.truth_events.sum())
    nonseizure_seconds = float(records.nonseizure_s.sum())
    delays = float(records.delay_sum_s.sum())
    delay_count = int(records.delay_count.sum())
    return {
        "patients": int(records.patient_id.nunique()),
        "records": int(len(records)),
        "truth_events": truth_events,
        "tp": tp,
        "fp": fp,
        "fn": truth_events - tp,
        "sensitivity": tp / truth_events if truth_events else np.nan,
        "fa_per_24h": fp / (nonseizure_seconds / 86400.0) if nonseizure_seconds else np.nan,
        "mean_delay_s": delays / delay_count if delay_count else np.nan,
        "median_record_delay_s": float(records.median_delay_s.median()) if records.delay_count.any() else np.nan,
        "nonseizure_h": nonseizure_seconds / 3600.0,
    }


def _bootstrap_run(records: pd.DataFrame, model: str, fold: int, seed: int, rng: np.random.Generator):
    patients = sorted(records.patient_id.unique().tolist())
    rows = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(patients, len(patients), replace=True)
        boot = pd.concat([records[records.patient_id == patient] for patient in chosen], ignore_index=True)
        summary = _summarize_records(boot)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "seed": seed,
                "replicate": replicate,
                "sensitivity": summary["sensitivity"],
                "fa_per_24h": summary["fa_per_24h"],
                "mean_delay_s": summary["mean_delay_s"],
                "patients_sampled": len(patients),
            }
        )
    return rows


def main() -> None:
    manifest = json.loads(COHORT_MANIFEST.read_text())
    cohort_hash = _sha256(COHORT_MANIFEST)
    records_manifest = manifest["records"]
    record_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_root = EXTERNAL_ROOT / model / f"fold{fold}_seed{seed}"
                progress = json.loads((run_root / "progress.json").read_text())
                if progress.get("status") != "complete" or progress.get("records_completed") != len(records_manifest) or progress.get("records_failed", 0):
                    raise RuntimeError(f"inference run is not complete: {model} fold={fold} seed={seed}")
                if progress.get("cohort_manifest_sha256") != cohort_hash:
                    raise RuntimeError(f"cohort hash mismatch: {model} fold={fold} seed={seed}")
                run_records: list[dict[str, object]] = []
                run_events: list[dict[str, object]] = []
                for record in records_manifest:
                    relative = Path(record["relative_edf"])
                    probability_path = run_root / "probabilities" / relative.with_suffix(".parquet")
                    metadata_path = run_root / "metadata" / relative.with_suffix(".json")
                    if not probability_path.exists() or not metadata_path.exists():
                        raise FileNotFoundError(probability_path)
                    row, events = _score_record(record, probability_path, metadata_path, model, fold, seed)
                    run_records.append(row)
                    run_events.extend(events)
                frame = pd.DataFrame(run_records)
                summary = _summarize_records(frame)
                run_rows.append({"model": model, "fold": fold, "seed": seed, **summary, "threshold_values": json.dumps(sorted(frame.threshold.unique().tolist()))})
                record_rows.extend(run_records)
                event_rows.extend(run_events)
                bootstrap_rows.extend(_bootstrap_run(frame, model, fold, seed, rng))
                print(f"TUSZ_GROUPKFOLD_SCORE_DONE model={model} fold={fold} seed={seed} sensitivity={summary['sensitivity']:.4f} fa_per_24h={summary['fa_per_24h']:.3f}", flush=True)

    run_frame = pd.DataFrame(run_rows)
    record_frame = pd.DataFrame(record_rows)
    event_frame = pd.DataFrame(event_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    model_rows = []
    for model, group in run_frame.groupby("model", sort=True):
        model_rows.append(
            {
                "model": model,
                "runs": int(len(group)),
                "sensitivity_mean": float(group.sensitivity.mean()),
                "sensitivity_sd": float(group.sensitivity.std(ddof=1)),
                "sensitivity_q025": float(group.sensitivity.quantile(0.025)),
                "sensitivity_q975": float(group.sensitivity.quantile(0.975)),
                "fa_per_24h_mean": float(group.fa_per_24h.mean()),
                "fa_per_24h_median": float(group.fa_per_24h.median()),
                "fa_per_24h_q025": float(group.fa_per_24h.quantile(0.025)),
                "fa_per_24h_q975": float(group.fa_per_24h.quantile(0.975)),
                "mean_delay_s_mean": float(group.mean_delay_s.mean()),
                "mean_delay_s_median": float(group.mean_delay_s.median()),
                "truth_events_total_across_runs": int(group.truth_events.sum()),
                "tp_total_across_runs": int(group.tp.sum()),
                "fp_total_across_runs": int(group.fp.sum()),
            }
        )
    model_frame = pd.DataFrame(model_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(OUTPUT_RUNS, index=False)
    record_frame.to_parquet(OUTPUT_RECORDS, index=False)
    event_frame.to_parquet(OUTPUT_EVENTS, index=False)
    bootstrap_frame.to_parquet(OUTPUT_BOOTSTRAP, index=False)
    model_frame.to_csv(OUTPUT_MODEL_SUMMARY, index=False)
    scoring_manifest = {
        "release_id": RUN_ID,
        "external_cohort_id": manifest["external_cohort_id"],
        "status": "external_event_scoring_complete",
        "cohort_manifest_sha256": cohort_hash,
        "inference_runs_scored": int(len(run_frame)),
        "expected_inference_runs": 45,
        "record_rows": int(len(record_frame)),
        "event_rows": int(len(event_frame)),
        "bootstrap_rows": int(len(bootstrap_frame)),
        "patient_bootstrap_replicates_per_run": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eventization": {"ema_alpha": 1 / 3, "min_duration_s": 4.0, "merge_gap_s": 10.0, "refractory_s": 30.0},
        "matching_minimum_overlap_s": 1.0,
        "threshold_source": "each frozen CHB validation-only threshold; no external retuning",
        "external_labels_used_for_selection": False,
        "model_training": False,
        "threshold_retuning": False,
        "outputs": {
            "run_metrics_sha256": _sha256(OUTPUT_RUNS),
            "record_metrics_sha256": _sha256(OUTPUT_RECORDS),
            "events_sha256": _sha256(OUTPUT_EVENTS),
            "bootstrap_sha256": _sha256(OUTPUT_BOOTSTRAP),
            "model_summary_sha256": _sha256(OUTPUT_MODEL_SUMMARY),
        },
    }
    OUTPUT_MANIFEST.write_text(json.dumps(scoring_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(scoring_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
