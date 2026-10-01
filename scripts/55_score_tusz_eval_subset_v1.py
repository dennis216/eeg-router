"""Score frozen TUSZ probabilities with frozen CHB thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs/reports/v3-compact-extension-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPORT_DIR / "tusz_eval_subset_v1_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tuh_external/tusz_eval_subset_v1")
    parser.add_argument("--prefix", default="tusz_eval_subset_v1")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output_root = args.output_root
    per_record: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for model in MODELS:
        for record in manifest["records"]:
            relative = Path(record["relative_edf"])
            probability_path = output_root / model / "probabilities" / relative.with_suffix(".parquet")
            metadata_path = output_root / model / "metadata" / relative.with_suffix(".json")
            table = pd.read_parquet(probability_path)
            metadata = json.loads(metadata_path.read_text())
            threshold = float(metadata["threshold"])
            first_prediction_end = float(table.window_end_s.iloc[0]) if len(table) else min(70.0, float(record["duration_s"]))
            predictions = eventize(table.window_end_s.to_numpy(), table.probability.to_numpy(), threshold=threshold) if len(table) >= 2 else []
            truth = [(max(float(event["start_s"]), first_prediction_end), float(event["end_s"])) for event in record["seizure_events"] if float(event["end_s"]) > first_prediction_end]
            matching = match_events(predictions, truth)
            seizure_seconds = sum(max(0.0, end - start) for start, end in truth)
            nonseizure_seconds = max(0.0, float(record["duration_s"]) - first_prediction_end - seizure_seconds)
            delays = [predictions[pair.prediction_index].start_s - truth[pair.truth_index][0] for pair in matching.pairs]
            per_record.append({
                "model": model,
                "patient_id": record["patient_id"],
                "recording": relative.as_posix(),
                "threshold": threshold,
                "tp": len(matching.pairs),
                "fp": len(matching.unmatched_predictions),
                "fn": len(matching.unmatched_truths),
                "truth_events": len(truth),
                "nonseizure_s": nonseizure_seconds,
                "delays": delays,
            })
            for index, event in enumerate(predictions):
                event_rows.append({
                    "model": model,
                    "patient_id": record["patient_id"],
                    "recording": relative.as_posix(),
                    "event_index": index,
                    "start_s": event.start_s,
                    "end_s": event.end_s,
                    "peak_probability": event.peak_probability,
                    "matched_truth": any(pair.prediction_index == index for pair in matching.pairs),
                })

    event_path = REPORT_DIR / f"{args.prefix}_events.parquet"
    pd.DataFrame(event_rows).to_parquet(event_path, index=False)
    per_record_df = pd.DataFrame(per_record)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260812)
    for model, group in per_record_df.groupby("model", sort=True):
        tp = int(group.tp.sum())
        fp = int(group.fp.sum())
        truth_events = int(group.truth_events.sum())
        nonseizure_seconds = float(group.nonseizure_s.sum())
        delays = [value for values in group.delays for value in values]
        base = {
            "model": model,
            "patients": int(group.patient_id.nunique()),
            "records": len(group),
            "truth_events": truth_events,
            "tp": tp,
            "fp": fp,
            "fn": truth_events - tp,
            "sensitivity": tp / truth_events if truth_events else np.nan,
            "fa_per_24h": fp / (nonseizure_seconds / 86400.0) if nonseizure_seconds else np.nan,
            "median_delay_s": float(np.median(delays)) if delays else np.nan,
            "nonseizure_h": nonseizure_seconds / 3600.0,
        }
        patients = sorted(group.patient_id.unique())
        samples = []
        for replicate in range(2000):
            chosen = rng.choice(patients, len(patients), replace=True)
            boot = pd.concat([group[group.patient_id == patient] for patient in chosen])
            boot_truth = int(boot.truth_events.sum())
            boot_nonseizure = float(boot.nonseizure_s.sum())
            boot_sensitivity = float(boot.tp.sum() / boot_truth) if boot_truth else np.nan
            boot_fa = float(boot.fp.sum() / (boot_nonseizure / 86400.0)) if boot_nonseizure else np.nan
            samples.append((boot_sensitivity, boot_fa))
            bootstrap_rows.append({"model": model, "replicate": replicate, "sensitivity": boot_sensitivity, "fa_per_24h": boot_fa})
        sample_array = np.asarray(samples, dtype=float)
        base.update({
            "sensitivity_ci_low": float(np.nanquantile(sample_array[:, 0], 0.025)),
            "sensitivity_ci_high": float(np.nanquantile(sample_array[:, 0], 0.975)),
            "fa_per_24h_ci_low": float(np.nanquantile(sample_array[:, 1], 0.025)),
            "fa_per_24h_ci_high": float(np.nanquantile(sample_array[:, 1], 0.975)),
        })
        rows.append(base)

    results_path = REPORT_DIR / f"{args.prefix}_results.csv"
    bootstrap_path = REPORT_DIR / f"{args.prefix}_bootstrap.parquet"
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_path, index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(bootstrap_path, index=False)
    results_manifest = {
        "release_id": "v3-compact-extension-v1",
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": sha256(args.manifest),
        "status": "preliminary_frozen_external_eval_complete",
        "models": list(MODELS),
        "patient_bootstrap_replicates": 2000,
        "model_training": False,
        "threshold_retuning": False,
        "full_tusz_eval": args.prefix == "tusz_eval_full_v1",
        "results": rows,
        "events_sha256": sha256(event_path),
        "results_sha256": sha256(results_path),
    }
    results_manifest_path = REPORT_DIR / f"{args.prefix}_results_manifest.json"
    results_manifest_path.write_text(json.dumps(results_manifest, indent=2, sort_keys=True) + "\n")
    print(results_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
