"""Validation-only alarm-time work-point selection for frozen final detectors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_ID = "v3-groupkfold-confirmatory-v1"
OUT = ROOT / "outputs" / "reports" / "h4-router-revision-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = range(5)
SEEDS = (17, 42, 3407)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def events_from_probabilities(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording, group in frame.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        for event in eventize(
            ordered.end.to_numpy(float),
            ordered.probability.to_numpy(float),
            threshold=float(threshold),
        ):
            rows.append(
                {
                    "patient": str(ordered.patient.iloc[0]),
                    "recording": str(recording),
                    "event_start": float(event.start_s),
                    "event_end": float(event.end_s),
                    "peak_probability": float(event.peak_probability),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["patient", "recording", "event_start", "event_end", "peak_probability"],
    )


def main() -> None:
    revised = load_module(ROOT / "scripts" / "251_run_h4_router_revision_diagnostics.py", "h4_revised")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    durations = recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    sweep_rows: list[dict[str, object]] = []
    lock_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for fold in FOLDS:
        for seed in SEEDS:
            per_model_validation: dict[str, dict[str, float]] = {}
            per_model_test: dict[str, dict[str, float]] = {}
            for model in MODELS:
                run = ROOT / "runs" / SOURCE_ID / model / f"split{fold}_seed{seed}_main"
                validation = pd.read_parquet(run / "validation" / "probabilities_step_05000.parquet")
                test = pd.read_parquet(run / "test" / "probabilities.parquet")
                val_starts, val_patients = revised.partition_metadata(fold, "validation", recordings, validation)
                test_starts, test_patients = revised.partition_metadata(fold, "test", recordings, test)
                candidates: list[dict[str, float]] = []
                frozen_sweep = pd.read_parquet(
                    run / "validation" / "threshold_sweep_step_05000.parquet"
                )
                frozen_max_sensitivity = float(frozen_sweep.event_sensitivity.max())
                eligible_thresholds = frozen_sweep.loc[
                    frozen_sweep.event_sensitivity >= frozen_max_sensitivity - 0.03,
                    "threshold",
                ].to_numpy(float)
                for threshold in eligible_thresholds:
                    events = events_from_probabilities(validation, float(threshold))
                    score = revised.score_events(events, val_starts, val_patients, seizures, durations)
                    row = {"fold": fold, "seed": seed, "model": model, "threshold": float(threshold), **score}
                    candidates.append(row)
                    sweep_rows.append(row)
                frame = pd.DataFrame(candidates)
                max_sensitivity = float(frame.sensitivity.max())
                eligible = frame[frame.sensitivity >= max_sensitivity - 0.03].copy()
                selected = eligible.sort_values(
                    ["false_alarm_minutes_per_24h", "fa_per_24h", "mean_delay_s", "threshold"],
                    ascending=[True, True, True, False],
                    na_position="last",
                    kind="stable",
                ).iloc[0]
                threshold = float(selected.threshold)
                val_score = selected.drop(labels=["fold", "seed", "model", "threshold"]).to_dict()
                per_model_validation[model] = val_score
                lock_rows.append(
                    {"fold": fold, "seed": seed, "model": model, "threshold": threshold, "max_validation_sensitivity": max_sensitivity, **val_score}
                )
                test_events = events_from_probabilities(test, threshold)
                test_score = revised.score_events(test_events, test_starts, test_patients, seizures, durations)
                per_model_test[model] = test_score
                test_rows.append({"fold": fold, "seed": seed, "method": model, "selected_model": model, "threshold": threshold, **test_score})
                print(f"ALARM_TIME_LOCK_DONE fold={fold} seed={seed} model={model} threshold={threshold}", flush=True)

            target = max(score["sensitivity"] for score in per_model_validation.values())
            eligible_models = [
                model for model, score in per_model_validation.items()
                if score["sensitivity"] >= target - 0.03
            ]
            best_model = sorted(
                eligible_models,
                key=lambda model: (
                    per_model_validation[model]["false_alarm_minutes_per_24h"],
                    per_model_validation[model]["fa_per_24h"],
                    per_model_validation[model]["mean_delay_s"] if np.isfinite(per_model_validation[model]["mean_delay_s"]) else np.inf,
                ),
            )[0]
            best_lock = next(
                row for row in reversed(lock_rows)
                if row["fold"] == fold and row["seed"] == seed and row["model"] == best_model
            )
            test_rows.append(
                {"fold": fold, "seed": seed, "method": "best_single_alarm_time_selected", "selected_model": best_model, "threshold": best_lock["threshold"], **per_model_test[best_model]}
            )

    pd.DataFrame(sweep_rows).to_csv(OUT / "final_checkpoint_validation_alarm_time_sweeps.csv", index=False)
    pd.DataFrame(lock_rows).to_csv(OUT / "final_checkpoint_validation_alarm_time_locks.csv", index=False)
    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "final_checkpoint_alarm_time_locked_test.csv", index=False)
    summary = tests.groupby("method", as_index=False).agg(
        runs=("fold", "size"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_fa_per_24h=("fa_per_24h", "mean"),
        mean_false_alarm_minutes_per_24h=("false_alarm_minutes_per_24h", "mean"),
        mean_alarm_minutes_per_24h=("alarm_minutes_per_24h", "mean"),
        mean_delay_s=("mean_delay_s", "mean"),
    )
    summary.to_csv(OUT / "final_checkpoint_alarm_time_locked_test_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
