"""Create a non-destructive audit of TUSZ detection-delay semantics.

The original external scorer defines delay as prediction-event onset minus
seizure onset.  Because a prediction event may begin before a seizure and
continue into it, this quantity can be negative.  This audit preserves the
original files and writes a separate corrected view where an already-active
alarm has onset latency zero:

    onset_latency_s = max(prediction_start_s, seizure_start_s) - seizure_start_s

Sensitivity, false-alarm counts, matching, thresholds, and checkpoints are
not changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-groupkfold-confirmatory-v1"
EVENTS = REPORT_DIR / "tusz_groupkfold_external_v1_events.parquet"
RECORDS = REPORT_DIR / "tusz_groupkfold_external_v1_record_metrics.parquet"
COHORT = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
OUT_EVENTS = REPORT_DIR / "tusz_groupkfold_external_v1_delay_audit_events.parquet"
OUT_RUNS = REPORT_DIR / "tusz_groupkfold_external_v1_delay_audit_run_summary.csv"
OUT_MODELS = REPORT_DIR / "tusz_groupkfold_external_v1_delay_audit_model_summary.csv"
OUT_MANIFEST = REPORT_DIR / "tusz_groupkfold_external_v1_delay_audit_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    events = pd.read_parquet(EVENTS)
    records = pd.read_parquet(RECORDS)
    cohort = json.loads(COHORT.read_text())
    # Reproduce the scorer's post-warmup truth list exactly.  The matcher's
    # truth index is relative to this filtered list, not the raw annotation
    # list.  One representative probability file per recording is sufficient
    # because the temporal grid is shared across all frozen runs.
    representative = records.drop_duplicates("recording").set_index("recording")
    first_prediction_end: dict[str, float] = {}
    for recording, row in representative.iterrows():
        probability_path = ROOT / str(row["probability_path"])
        probability_table = pd.read_parquet(probability_path, columns=["window_end_s"])
        if len(probability_table) >= 2:
            first_prediction_end[str(recording)] = float(probability_table.window_end_s.iloc[0])
        else:
            duration_s = float(
                next(item["duration_s"] for item in cohort["records"] if str(item["relative_edf"]) == str(recording))
            )
            first_prediction_end[str(recording)] = min(70.0, duration_s)
    truth_lookup: dict[tuple[str, int], float] = {}
    for record in cohort["records"]:
        recording = str(record["relative_edf"])
        warmup_end = first_prediction_end[recording]
        filtered_truths = [
            (max(float(seizure["start_s"]), warmup_end), float(seizure["end_s"]))
            for seizure in record["seizure_events"]
            if float(seizure["end_s"]) > warmup_end
        ]
        for index, (start_s, _end_s) in enumerate(filtered_truths):
            truth_lookup[(recording, index)] = start_s

    matched = events[events.is_true_positive].copy()
    matched["matched_truth_index"] = matched.matched_truth_index.astype(int)
    matched["truth_start_s"] = [
        truth_lookup[(str(recording), int(index))]
        for recording, index in zip(matched.recording, matched.matched_truth_index, strict=True)
    ]
    matched["raw_event_onset_minus_truth_s"] = matched.start_s - matched.truth_start_s
    matched["onset_latency_s"] = np.maximum(matched.start_s, matched.truth_start_s) - matched.truth_start_s
    matched["alarm_active_before_onset"] = matched.start_s < matched.truth_start_s

    run_rows = []
    for keys, group in matched.groupby(["model", "fold", "seed"], sort=True):
        model, fold, seed = keys
        run_rows.append(
            {
                "model": model,
                "fold": int(fold),
                "seed": int(seed),
                "matched_events": int(len(group)),
                "pre_onset_active_events": int(group.alarm_active_before_onset.sum()),
                "pre_onset_active_fraction": float(group.alarm_active_before_onset.mean()),
                "raw_mean_event_onset_minus_truth_s": float(group.raw_event_onset_minus_truth_s.mean()),
                "corrected_mean_onset_latency_s": float(group.onset_latency_s.mean()),
                "corrected_median_onset_latency_s": float(group.onset_latency_s.median()),
            }
        )
    runs = pd.DataFrame(run_rows)
    model_rows = []
    for model, group in matched.groupby("model", sort=True):
        model_rows.append(
            {
                "model": model,
                "matched_events": int(len(group)),
                "pre_onset_active_events": int(group.alarm_active_before_onset.sum()),
                "pre_onset_active_fraction": float(group.alarm_active_before_onset.mean()),
                "raw_mean_event_onset_minus_truth_s": float(group.raw_event_onset_minus_truth_s.mean()),
                "corrected_mean_onset_latency_s": float(group.onset_latency_s.mean()),
                "corrected_median_onset_latency_s": float(group.onset_latency_s.median()),
            }
        )
    models = pd.DataFrame(model_rows)

    matched.to_parquet(OUT_EVENTS, index=False)
    runs.to_csv(OUT_RUNS, index=False)
    models.to_csv(OUT_MODELS, index=False)
    manifest = {
        "analysis_id": "tusz-detection-delay-semantics-audit-v1",
        "status": "complete_non_destructive_correction",
        "original_definition": "prediction event onset minus seizure onset; may be negative when an alarm begins before and overlaps seizure onset",
        "corrected_definition": "max(prediction event onset, seizure onset) minus seizure onset; an already-active alarm has latency zero",
        "sensitivity_changed": False,
        "false_alarm_counts_changed": False,
        "event_matching_changed": False,
        "thresholds_or_models_changed": False,
        "source_hashes": {"events": sha256(EVENTS), "records": sha256(RECORDS), "cohort": sha256(COHORT)},
        "outputs": {"events": sha256(OUT_EVENTS), "runs": sha256(OUT_RUNS), "models": sha256(OUT_MODELS)},
        "matched_events": int(len(matched)),
        "pre_onset_active_events": int(matched.alarm_active_before_onset.sum()),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(models.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
