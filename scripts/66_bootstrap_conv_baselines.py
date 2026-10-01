from __future__ import annotations

"""Compute patient-level bootstrap intervals for convolutional baseline tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "runs" / "baselines" / "v3-compact-conv-baselines"
N_BOOTSTRAP = 2000


def truth_intervals(seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def patient_rows(
    table: pd.DataFrame,
    threshold: float,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recording_id, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if len(ordered) < 2:
            continue
        evaluation_start_s = float(ordered.end.iloc[0])
        recording_rows = recordings[recordings.recording_id == recording_id]
        if recording_rows.empty:
            continue
        duration_s = float(recording_rows.duration_s.iloc[0])
        truths = truth_intervals(seizures, str(recording_id), evaluation_start_s)
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=threshold,
        )
        matched = match_events(predictions, truths)
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        delays = [
            predictions[pair.prediction_index].start_s - truths[pair.truth_index][0]
            for pair in matched.pairs
        ]
        rows.append(
            {
                "patient": str(ordered.patient.iloc[0]),
                "recording": str(recording_id),
                "tp": len(matched.pairs),
                "fp": len(matched.unmatched_predictions),
                "truth": len(truths),
                "nonseizure_s": max(0.0, duration_s - evaluation_start_s - seizure_seconds),
                "median_delay_s": float(np.median(delays)) if delays else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no event rows were produced for a convolutional baseline test table")
    return frame.groupby("patient", as_index=False).agg(
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        truth=("truth", "sum"),
        nonseizure_s=("nonseizure_s", "sum"),
        median_delay_s=("median_delay_s", "mean"),
    )


def main() -> None:
    config = json.loads((RUN_ROOT / "baseline_audit.json").read_text())
    if not config.get("matrix_complete"):
        raise RuntimeError("convolutional baseline audit is not complete; refusing bootstrap")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    bootstrap_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model in ("deepconvnet", "shallowconvnet"):
        for split in (17, 42):
            for seed in (17, 42, 3407):
                run_dir = RUN_ROOT / model / f"{model}_split{split}_seed{seed}"
                table = pd.read_parquet(run_dir / "test" / "probabilities.parquet")
                lock = json.loads((run_dir / "threshold_lock.json").read_text())
                patient_frame = patient_rows(table, float(lock["threshold"]), seizures, recordings)
                total_tp = int(patient_frame.tp.sum())
                total_fp = int(patient_frame.fp.sum())
                total_truth = int(patient_frame.truth.sum())
                total_nonseizure_s = float(patient_frame.nonseizure_s.sum())
                point_sensitivity = total_tp / total_truth if total_truth else np.nan
                point_fa = total_fp / (total_nonseizure_s / 86400.0) if total_nonseizure_s else np.nan
                rng = np.random.default_rng(20260815 + split * 100 + seed)
                patients = np.arange(len(patient_frame))
                sampled_sensitivity = np.empty(N_BOOTSTRAP, dtype=float)
                sampled_fa = np.empty(N_BOOTSTRAP, dtype=float)
                for replicate in range(N_BOOTSTRAP):
                    chosen = rng.choice(patients, size=len(patients), replace=True)
                    subset = patient_frame.iloc[chosen]
                    boot_truth = int(subset.truth.sum())
                    boot_nonseizure = float(subset.nonseizure_s.sum())
                    sampled_sensitivity[replicate] = subset.tp.sum() / boot_truth if boot_truth else np.nan
                    sampled_fa[replicate] = subset.fp.sum() / (boot_nonseizure / 86400.0) if boot_nonseizure else np.nan
                    bootstrap_rows.append(
                        {
                            "model": model,
                            "split": split,
                            "seed": seed,
                            "replicate": replicate,
                            "sensitivity": sampled_sensitivity[replicate],
                            "fa_per_24h": sampled_fa[replicate],
                        }
                    )
                summary_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "seed": seed,
                        "patients": len(patient_frame),
                        "point_sensitivity": point_sensitivity,
                        "point_fa_per_24h": point_fa,
                        "sensitivity_ci_low": float(np.nanquantile(sampled_sensitivity, 0.025)),
                        "sensitivity_ci_high": float(np.nanquantile(sampled_sensitivity, 0.975)),
                        "fa_per_24h_ci_low": float(np.nanquantile(sampled_fa, 0.025)),
                        "fa_per_24h_ci_high": float(np.nanquantile(sampled_fa, 0.975)),
                        "bootstrap_replicates": N_BOOTSTRAP,
                    }
                )
    bootstrap_path = RUN_ROOT / "baseline_patient_bootstrap.parquet"
    summary_path = RUN_ROOT / "baseline_patient_bootstrap_summary.csv"
    pd.DataFrame(bootstrap_rows).to_parquet(bootstrap_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    manifest = {
        "namespace": "v3-compact-conv-baselines",
        "patient_bootstrap_replicates": N_BOOTSTRAP,
        "run_count": len(summary_rows),
        "bootstrap_parquet": str(bootstrap_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
    }
    (RUN_ROOT / "baseline_patient_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
