"""Build standard GroupKFold out-of-fold event and uncertainty reports.

This script is intentionally downstream of ``72_audit_groupkfold_confirmatory_v1``.
It reads the frozen validation-selected threshold recorded by each run, never
selects a threshold from test data, and writes all outputs under the new
``v3-groupkfold-confirmatory-v1`` namespace.  Seed-specific OOF results are
retained; model-level patient bootstrap summaries average the three registered
seeds within each patient before resampling patients.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "v3-groupkfold-confirmatory-v1"
RUN_ROOT = ROOT / "runs" / RUN_ID
REPORT_ROOT = ROOT / "outputs" / "reports" / RUN_ID
MANIFEST_ROOT = ROOT / "manifests" / "groupkfold_cv_v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
N_BOOTSTRAP = 2000
CURVE_THRESHOLDS = tuple([0.01] + [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)] + [0.99])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def _recording_score(
    ordered: pd.DataFrame,
    *,
    threshold: float,
    seizures: pd.DataFrame,
    duration_by_recording: dict[str, float],
    model: str,
    fold: int,
    seed: int,
    emit_events: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    recording = str(ordered.recording.iloc[0])
    patient = str(ordered.patient.iloc[0])
    evaluation_start_s = float(ordered.end.iloc[0])
    duration_s = float(duration_by_recording[recording])
    truths = _truths(seizures, recording, evaluation_start_s)
    predictions = eventize(
        ordered.end.to_numpy(dtype=float),
        ordered.probability.to_numpy(dtype=float),
        threshold=float(threshold),
    )
    matched = match_events(predictions, truths)
    signed_deltas = [
        float(predictions[pair.prediction_index].start_s - truths[pair.truth_index][0])
        for pair in matched.pairs
    ]
    # A prediction may begin before seizure onset and continue into the
    # annotated seizure interval. Preserve that signed onset difference as
    # an audit field, but report clinical detection delay as a non-negative
    # latency, consistent with the router and external-evaluation scorers.
    delays = [max(0.0, value) for value in signed_deltas]
    seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
    row = {
        "model": model,
        "fold": fold,
        "seed": seed,
        "patient": patient,
        "recording": recording,
        "threshold": float(threshold),
        "tp": int(len(matched.pairs)),
        "fp": int(len(matched.unmatched_predictions)),
        "truth": int(len(truths)),
        "nonseizure_s": float(max(0.0, duration_s - evaluation_start_s - seizure_seconds)),
        "delay_sum_s": float(np.sum(delays)) if delays else 0.0,
        "delay_count": int(len(delays)),
    }
    events: list[dict[str, object]] = []
    if emit_events:
        matched_by_prediction = {pair.prediction_index: pair for pair in matched.pairs}
        for prediction_index, event in enumerate(predictions):
            pair = matched_by_prediction.get(prediction_index)
            truth_start = truths[pair.truth_index][0] if pair is not None else np.nan
            truth_end = truths[pair.truth_index][1] if pair is not None else np.nan
            events.append(
                {
                    "model": model,
                    "fold": fold,
                    "seed": seed,
                    "patient": patient,
                    "recording": recording,
                    "start_s": float(event.start_s),
                    "end_s": float(event.end_s),
                    "peak_probability": float(event.peak_probability),
                    "event_type": "tp" if pair is not None else "fp",
                    "truth_index": int(pair.truth_index) if pair is not None else None,
                    "truth_start_s": float(truth_start),
                    "truth_end_s": float(truth_end),
                    "signed_onset_delta_s": float(event.start_s - truth_start) if pair is not None else np.nan,
                    "detection_delay_s": max(0.0, float(event.start_s - truth_start)) if pair is not None else np.nan,
                    "threshold": float(threshold),
                }
            )
    return row, events


def _score_table(
    table: pd.DataFrame,
    *,
    threshold: float,
    seizures: pd.DataFrame,
    duration_by_recording: dict[str, float],
    model: str,
    fold: int,
    seed: int,
    emit_events: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for _, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if len(ordered) < 2:
            continue
        row, event_rows = _recording_score(
            ordered,
            threshold=threshold,
            seizures=seizures,
            duration_by_recording=duration_by_recording,
            model=model,
            fold=fold,
            seed=seed,
            emit_events=emit_events,
        )
        rows.append(row)
        events.extend(event_rows)
    return pd.DataFrame(rows), events


def _aggregate_metrics(recordings: pd.DataFrame) -> dict[str, object]:
    if recordings.empty:
        return {"tp": 0, "fp": 0, "truth": 0, "nonseizure_s": 0.0, "event_sensitivity": np.nan, "fa_per_24h": np.nan, "mean_detection_delay_s": np.nan, "matched_events": 0}
    tp = int(recordings.tp.sum())
    fp = int(recordings.fp.sum())
    truth = int(recordings.truth.sum())
    nonseizure_s = float(recordings.nonseizure_s.sum())
    delay_count = int(recordings.delay_count.sum())
    delay_sum = float(recordings.delay_sum_s.sum())
    return {
        "tp": tp,
        "fp": fp,
        "truth": truth,
        "nonseizure_s": nonseizure_s,
        "event_sensitivity": tp / truth if truth else np.nan,
        "fa_per_24h": fp * 86400.0 / nonseizure_s if nonseizure_s else np.nan,
        "mean_detection_delay_s": delay_sum / delay_count if delay_count else np.nan,
        "matched_events": delay_count,
    }


def _bootstrap_frame(frame: pd.DataFrame, *, rng_seed: int, model: str, seed: int | None) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.empty:
        raise RuntimeError("cannot bootstrap an empty patient frame")
    rng = np.random.default_rng(rng_seed)
    n = len(frame)
    replicates: list[dict[str, object]] = []
    for replicate in range(N_BOOTSTRAP):
        chosen = rng.integers(0, n, size=n)
        subset = frame.iloc[chosen]
        tp = float(subset.tp.sum())
        fp = float(subset.fp.sum())
        truth = float(subset.truth.sum())
        nonseizure_s = float(subset.nonseizure_s.sum())
        delay_count = float(subset.delay_count.sum())
        delay_sum = float(subset.delay_sum_s.sum())
        replicates.append(
            {
                "model": model,
                "seed": seed,
                "replicate": replicate,
                "event_sensitivity": tp / truth if truth else np.nan,
                "fa_per_24h": fp * 86400.0 / nonseizure_s if nonseizure_s else np.nan,
                "mean_detection_delay_s": delay_sum / delay_count if delay_count else np.nan,
            }
        )
    boot = pd.DataFrame(replicates)
    point = _aggregate_metrics(frame)
    summary = {
        "model": model,
        "seed": seed,
        "patients": n,
        "bootstrap_replicates": N_BOOTSTRAP,
        "point_event_sensitivity": point["event_sensitivity"],
        "sensitivity_ci_low": float(boot.event_sensitivity.quantile(0.025)),
        "sensitivity_ci_high": float(boot.event_sensitivity.quantile(0.975)),
        "point_fa_per_24h": point["fa_per_24h"],
        "fa_per_24h_ci_low": float(boot.fa_per_24h.quantile(0.025)),
        "fa_per_24h_ci_high": float(boot.fa_per_24h.quantile(0.975)),
        "point_mean_detection_delay_s": point["mean_detection_delay_s"],
        "delay_ci_low": float(boot.mean_detection_delay_s.quantile(0.025)),
        "delay_ci_high": float(boot.mean_detection_delay_s.quantile(0.975)),
    }
    return boot, summary


def main() -> None:
    audit_path = RUN_ROOT / "groupkfold_confirmatory_audit.json"
    if not audit_path.exists():
        raise RuntimeError("run scripts/72_audit_groupkfold_confirmatory_v1.py first")
    audit = json.loads(audit_path.read_text())
    if not audit.get("matrix_complete"):
        raise RuntimeError("confirmatory matrix audit is not complete")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    duration_by_recording = {
        str(row.recording_id): float(row.duration_s)
        for row in recordings.itertuples(index=False)
    }

    run_metrics: list[dict[str, object]] = []
    recording_metrics: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    probability_path = REPORT_ROOT / "oof_probabilities_seed.parquet"
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    probability_writer: pq.ParquetWriter | None = None

    for fold in FOLDS:
        split_path = MANIFEST_ROOT / f"fold_{fold}.json"
        split = json.loads(split_path.read_text())
        expected_test = set(split["test"])
        for model in MODELS:
            for seed in SEEDS:
                run_dir = RUN_ROOT / model / f"split{fold}_seed{seed}_main"
                completed = json.loads((run_dir / "completed.json").read_text())
                test_metrics = completed["test"]
                threshold = float(test_metrics["score"]["threshold"])
                table = pd.read_parquet(run_dir / "test" / "probabilities.parquet")
                observed_test = set(table.patient.astype(str).unique())
                if observed_test != expected_test:
                    raise RuntimeError(
                        f"{model} fold {fold} seed {seed}: test patient mismatch "
                        f"expected={sorted(expected_test)} observed={sorted(observed_test)}"
                    )
                augmented = table.copy()
                augmented["model"] = model
                augmented["fold"] = fold
                augmented["seed"] = seed
                augmented["validation_locked_threshold"] = threshold
                probability_arrow = pa.Table.from_pandas(augmented, preserve_index=False)
                if probability_writer is None:
                    probability_writer = pq.ParquetWriter(
                        probability_path, probability_arrow.schema, compression="zstd"
                    )
                probability_writer.write_table(probability_arrow)

                rec_frame, events = _score_table(
                    table,
                    threshold=threshold,
                    seizures=seizures,
                    duration_by_recording=duration_by_recording,
                    model=model,
                    fold=fold,
                    seed=seed,
                )
                recording_metrics.append(rec_frame)
                event_rows.extend(events)
                aggregate = _aggregate_metrics(rec_frame)
                run_metrics.append(
                    {
                        "model": model,
                        "fold": fold,
                        "seed": seed,
                        "threshold": threshold,
                        **aggregate,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                    }
                )

                for curve_threshold in CURVE_THRESHOLDS:
                    curve_rec, _ = _score_table(
                        table,
                        threshold=float(curve_threshold),
                        seizures=seizures,
                        duration_by_recording=duration_by_recording,
                        model=model,
                        fold=fold,
                        seed=seed,
                        emit_events=False,
                    )
                    curve_rows.append(
                        {
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "threshold": float(curve_threshold),
                            **_aggregate_metrics(curve_rec),
                            "threshold_source": "descriptive_test_sweep; no test-based selection",
                        }
                    )

    if probability_writer is not None:
        probability_writer.close()
    else:
        pd.DataFrame().to_parquet(probability_path, index=False)
    recording_frame = pd.concat(recording_metrics, ignore_index=True)
    event_frame = pd.DataFrame(event_rows)
    run_frame = pd.DataFrame(run_metrics)
    curve_frame = pd.DataFrame(curve_rows)
    recording_frame.to_parquet(REPORT_ROOT / "oof_recording_metrics.parquet", index=False)
    event_frame.to_parquet(REPORT_ROOT / "oof_events.parquet", index=False)
    run_frame.to_csv(REPORT_ROOT / "oof_run_metrics.csv", index=False)
    curve_frame.to_parquet(REPORT_ROOT / "sensitivity_fa_curves.parquet", index=False)

    seed_rows: list[dict[str, object]] = []
    for (model, seed), group in recording_frame.groupby(["model", "seed"], sort=True):
        seed_rows.append({"model": model, "seed": int(seed), **_aggregate_metrics(group)})
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(REPORT_ROOT / "oof_seed_metrics.csv", index=False)

    patient_frame = (
        recording_frame.groupby(["model", "seed", "patient"], as_index=False)
        .agg(
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            truth=("truth", "sum"),
            nonseizure_s=("nonseizure_s", "sum"),
            delay_sum_s=("delay_sum_s", "sum"),
            delay_count=("delay_count", "sum"),
        )
    )
    patient_frame["event_sensitivity"] = patient_frame.tp / patient_frame.truth.replace(0, np.nan)
    patient_frame["fa_per_24h"] = patient_frame.fp * 86400.0 / patient_frame.nonseizure_s.replace(0, np.nan)
    patient_frame["mean_detection_delay_s"] = patient_frame.delay_sum_s / patient_frame.delay_count.replace(0, np.nan)
    patient_frame.to_parquet(REPORT_ROOT / "oof_patient_metrics.parquet", index=False)
    waterfall = patient_frame.sort_values(["model", "seed", "event_sensitivity"], ascending=[True, True, False])
    waterfall.to_csv(REPORT_ROOT / "patient_waterfall.csv", index=False)

    bootstrap_frames: list[pd.DataFrame] = []
    bootstrap_summaries: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(SEEDS):
            frame = patient_frame[(patient_frame.model == model) & (patient_frame.seed == seed)].copy()
            boot, summary = _bootstrap_frame(frame, rng_seed=20260819 + model_index * 100 + seed_index, model=model, seed=seed)
            bootstrap_frames.append(boot)
            bootstrap_summaries.append(summary)
        averaged = (
            patient_frame[patient_frame.model == model]
            .groupby("patient", as_index=False)[["tp", "fp", "truth", "nonseizure_s", "delay_sum_s", "delay_count"]]
            .mean()
        )
        boot, summary = _bootstrap_frame(averaged, rng_seed=20260819 + model_index * 1000, model=model, seed=None)
        summary["seed_aggregation"] = "mean across registered seeds within patient"
        bootstrap_frames.append(boot)
        bootstrap_summaries.append(summary)
    pd.concat(bootstrap_frames, ignore_index=True).to_parquet(REPORT_ROOT / "patient_bootstrap.parquet", index=False)
    bootstrap_summary = pd.DataFrame(bootstrap_summaries)
    bootstrap_summary.to_csv(REPORT_ROOT / "patient_bootstrap_summary.csv", index=False)

    manifest = {
        "namespace": RUN_ID,
        "source_audit": str(audit_path.relative_to(ROOT)),
        "source_audit_sha256": _sha256(audit_path),
        "split_manifest": str((MANIFEST_ROOT / "cv_manifest.json").relative_to(ROOT)),
        "split_manifest_sha256": _sha256(MANIFEST_ROOT / "cv_manifest.json"),
        "models": list(MODELS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "run_count": len(run_frame),
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "threshold_policy": "each run's validation-selected threshold; no test threshold selection",
        "curve_thresholds": list(CURVE_THRESHOLDS),
        "oof_probability_file": str(probability_path.relative_to(ROOT)),
        "oof_events_file": str((REPORT_ROOT / "oof_events.parquet").relative_to(ROOT)),
        "oof_seed_metrics_file": str((REPORT_ROOT / "oof_seed_metrics.csv").relative_to(ROOT)),
        "patient_bootstrap_summary_file": str((REPORT_ROOT / "patient_bootstrap_summary.csv").relative_to(ROOT)),
    }
    for key in ("oof_probability_file", "oof_events_file", "oof_seed_metrics_file", "patient_bootstrap_summary_file"):
        manifest[key + "_sha256"] = _sha256(ROOT / manifest[key])
    (REPORT_ROOT / "oof_release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(bootstrap_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
