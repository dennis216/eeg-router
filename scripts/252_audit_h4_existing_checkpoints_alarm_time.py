"""Audit all saved detector checkpoints on validation under the revised H4 scorer.

No model inference or training is performed.  Existing validation probability
tables and their validation-selected event thresholds are reused.
"""

from __future__ import annotations

import importlib.util
import json
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
STEPS = (1000, 2000, 3000, 4000, 5000)


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
        detected = eventize(
            ordered.end.to_numpy(float),
            ordered.probability.to_numpy(float),
            threshold=float(threshold),
        )
        patient = str(ordered.patient.iloc[0])
        rows.extend(
            {
                "patient": patient,
                "recording": str(recording),
                "event_start": float(event.start_s),
                "event_end": float(event.end_s),
                "peak_probability": float(event.peak_probability),
            }
            for event in detected
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
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                run = ROOT / "runs" / SOURCE_ID / model / f"split{fold}_seed{seed}_main"
                for step in STEPS:
                    probability_path = run / "validation" / f"probabilities_step_{step:05d}.parquet"
                    metrics_path = run / "validation" / f"metrics_step_{step:05d}.json"
                    if not probability_path.exists() or not metrics_path.exists():
                        continue
                    probabilities = pd.read_parquet(probability_path)
                    starts, patients = revised.partition_metadata(
                        fold, "validation", recordings, probabilities
                    )
                    metrics = json.loads(metrics_path.read_text())
                    threshold = float(metrics["selected_event_operating_point"]["threshold"])
                    events = events_from_probabilities(probabilities, threshold)
                    score = revised.score_events(events, starts, patients, seizures, durations)
                    rows.append(
                        {
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "step": step,
                            "threshold": threshold,
                            "window_auroc": float(metrics.get("window_auroc", np.nan)),
                            "window_average_precision": float(metrics.get("window_average_precision", np.nan)),
                            **score,
                        }
                    )
                print(f"CHECKPOINT_AUDIT_DONE model={model} fold={fold} seed={seed}", flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "existing_checkpoint_validation_alarm_time.csv", index=False)

    selections: list[dict[str, object]] = []
    for (model, fold, seed), group in audit.groupby(["model", "fold", "seed"], sort=True):
        max_sensitivity = float(group.sensitivity.max())
        eligible = group[group.sensitivity >= max_sensitivity - 0.03].copy()
        clinical = eligible.sort_values(
            ["false_alarm_minutes_per_24h", "fa_per_24h", "mean_delay_s", "window_average_precision", "step"],
            ascending=[True, True, True, False, True],
            na_position="last",
            kind="stable",
        ).iloc[0]
        ap = group.sort_values(
            ["window_average_precision", "window_auroc", "step"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]
        final = group.sort_values("step").iloc[-1]
        selections.append(
            {
                "model": model,
                "fold": int(fold),
                "seed": int(seed),
                "clinical_selected_step": int(clinical.step),
                "ap_selected_step": int(ap.step),
                "final_step": int(final.step),
                "clinical_selected_sensitivity": float(clinical.sensitivity),
                "clinical_selected_fa_per_24h": float(clinical.fa_per_24h),
                "clinical_selected_false_alarm_minutes_per_24h": float(clinical.false_alarm_minutes_per_24h),
                "final_sensitivity": float(final.sensitivity),
                "final_fa_per_24h": float(final.fa_per_24h),
                "final_false_alarm_minutes_per_24h": float(final.false_alarm_minutes_per_24h),
            }
        )
    selections_frame = pd.DataFrame(selections)
    selections_frame.to_csv(OUT / "existing_checkpoint_validation_selections.csv", index=False)
    summary = selections_frame.groupby("model", as_index=False).agg(
        runs=("fold", "size"),
        clinical_final_fraction=("clinical_selected_step", lambda values: float(np.mean(np.asarray(values) == 5000))),
        ap_final_fraction=("ap_selected_step", lambda values: float(np.mean(np.asarray(values) == 5000))),
        mean_clinical_sensitivity=("clinical_selected_sensitivity", "mean"),
        mean_final_sensitivity=("final_sensitivity", "mean"),
        mean_clinical_fa_per_24h=("clinical_selected_fa_per_24h", "mean"),
        mean_final_fa_per_24h=("final_fa_per_24h", "mean"),
        mean_clinical_false_alarm_minutes_per_24h=("clinical_selected_false_alarm_minutes_per_24h", "mean"),
        mean_final_false_alarm_minutes_per_24h=("final_false_alarm_minutes_per_24h", "mean"),
    )
    summary.to_csv(OUT / "existing_checkpoint_validation_selection_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
