"""Freeze TP/FP/FN test events for v3-compact-extension-v1.

This script consumes only completed test probabilities and validation-locked
thresholds. It never searches thresholds or changes test predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
RUNS = (
    ("BSN-pilot-v1", 17, (17, 42, 3407)),
    ("BSN-pilot-v1-multisplit", 42, (17, 42)),
    ("v3-lite-2split", 42, (3407,)),
)
MODELS = ("singlem", "cbramod", "tcn_gat")
EVENT_DIR = ROOT / "outputs" / "events"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording: str, evaluation_start: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording]
    return [
        (max(float(row.start_s), evaluation_start), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start
    ]


def main() -> None:
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    rows: list[dict[str, object]] = []
    run_count = 0
    for namespace, split, seeds in RUNS:
        for seed in seeds:
            for model in MODELS:
                run = ROOT / "runs" / namespace / model / f"split{split}_seed{seed}_main"
                completed = json.loads((run / "completed.json").read_text())
                threshold = float(completed["test"]["score"]["threshold"])
                probability_path = run / "test" / "probabilities.parquet"
                table = pd.read_parquet(probability_path)
                for recording, group in table.groupby("recording", sort=False):
                    ordered = group.sort_values("end", kind="stable")
                    evaluation_start = float(ordered.end.iloc[0])
                    predictions = eventize(
                        ordered.end.to_numpy(float),
                        ordered.probability.to_numpy(float),
                        threshold=threshold,
                    )
                    truth = _truths(seizures, str(recording), evaluation_start)
                    matched = match_events(predictions, truth)
                    matched_by_prediction = {pair.prediction_index: pair for pair in matched.pairs}
                    matched_truth = {pair.truth_index for pair in matched.pairs}
                    patient = str(ordered.patient.iloc[0])
                    duration = float(
                        recordings.loc[recordings.recording_id == recording, "duration_s"].iloc[0]
                    )
                    common = {
                        "namespace": namespace,
                        "split": split,
                        "seed": seed,
                        "model": model,
                        "run_id": f"{namespace}/split{split}_seed{seed}_main",
                        "patient": patient,
                        "recording": str(recording),
                        "recording_duration_s": duration,
                        "threshold": threshold,
                        "threshold_source": "final_step_5000_validation_only",
                        "test_probability_sha256": completed["test"]["probability_sha256"],
                    }
                    for index, event in enumerate(predictions):
                        pair = matched_by_prediction.get(index)
                        rows.append(
                            {
                                **common,
                                "event_type": "tp" if pair is not None else "fp",
                                "start_s": event.start_s,
                                "end_s": event.end_s,
                                "peak_probability": event.peak_probability,
                                "truth_index": pair.truth_index if pair is not None else None,
                                "truth_start_s": truth[pair.truth_index][0] if pair is not None else None,
                                "truth_end_s": truth[pair.truth_index][1] if pair is not None else None,
                                "truth_overlap_s": pair.overlap_s if pair is not None else 0.0,
                            }
                        )
                    for truth_index, (truth_start, truth_end) in enumerate(truth):
                        if truth_index not in matched_truth:
                            rows.append(
                                {
                                    **common,
                                    "event_type": "fn",
                                    "start_s": truth_start,
                                    "end_s": truth_end,
                                    "peak_probability": None,
                                    "truth_index": truth_index,
                                    "truth_start_s": truth_start,
                                    "truth_end_s": truth_end,
                                    "truth_overlap_s": 0.0,
                                }
                            )
                run_count += 1
    events = pd.DataFrame(rows)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_path = EVENT_DIR / "v3-compact-extension-v1_test_events.parquet"
    events.to_parquet(event_path, index=False)
    counts = (
        events.groupby(["split", "model", "event_type"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    counts.to_csv(REPORT_DIR / "event_counts.csv", index=False)
    manifest = {
        "release_id": "v3-compact-extension-v1",
        "event_path": str(event_path),
        "event_table_sha256": _sha256(event_path),
        "runs": run_count,
        "rows": int(len(events)),
        "event_types": sorted(events.event_type.unique().tolist()),
        "threshold_source": "final_step_5000_validation_only",
        "event_definition": "causal EMA alpha=1/3; min duration 4s; merge gap 10s; refractory 30s; matcher as frozen evaluator",
        "counts": counts.to_dict(orient="records"),
    }
    (REPORT_DIR / "event_table_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
