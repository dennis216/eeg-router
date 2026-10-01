"""Build train/validation/test event tables for standard H2/H3.

Thresholds are the locked validation-selected thresholds recorded by each
formal run.  The outer-test rows are copied only for the final frozen
evaluation; no threshold or feature specification is selected from them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
TRAIN_DISCOVERY_DIR = os.environ.get("BFA_H2H3_TRAIN_DIR", "train_discovery")
EVENT_OUTPUT_NAME = os.environ.get("BFA_H2H3_EVENT_OUTPUT", "h2h3_events.parquet")
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truths(seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def _events_for_table(
    table: pd.DataFrame,
    *,
    partition: str,
    fold: int,
    seed: int,
    model: str,
    threshold: float,
    seizures: pd.DataFrame,
    relative_by_recording: dict[str, str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if len(ordered) < 2:
            continue
        recording_id = str(recording)
        evaluation_start_s = float(ordered.end.iloc[0])
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=float(threshold),
        )
        truth = _truths(seizures, recording_id, evaluation_start_s)
        matched = match_events(predictions, truth)
        matched_by_prediction = {pair.prediction_index: pair for pair in matched.pairs}
        patient = str(ordered.patient.iloc[0])
        for index, event in enumerate(predictions):
            pair = matched_by_prediction.get(index)
            truth_start = truth[pair.truth_index][0] if pair is not None else None
            truth_end = truth[pair.truth_index][1] if pair is not None else None
            signed = float(event.start_s - truth_start) if pair is not None else None
            output.append(
                {
                    "namespace": RUN_ID,
                    "partition": partition,
                    "fold": fold,
                    "seed": seed,
                    "model": model,
                    "patient": patient,
                    "recording": recording_id,
                    "relative_path": str(relative_by_recording[recording_id]),
                    "start_s": float(event.start_s),
                    "end_s": float(event.end_s),
                    "peak_probability": float(event.peak_probability),
                    "event_type": "tp" if pair is not None else "fp",
                    "truth_index": int(pair.truth_index) if pair is not None else None,
                    "truth_start_s": float(truth_start) if truth_start is not None else None,
                    "truth_end_s": float(truth_end) if truth_end is not None else None,
                    "signed_onset_delta_s": signed,
                    "threshold": float(threshold),
                }
            )
    return output


def main() -> None:
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    relative_by_recording = (
        windows.drop_duplicates("recording").set_index("recording")["relative_path"].astype(str).to_dict()
    )
    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
                completed = json.loads((run_dir / "completed.json").read_text())
                threshold = float(completed["test"]["score"]["threshold"])
                paths = {
                    "train": run_dir / TRAIN_DISCOVERY_DIR / "probabilities_step_05000.parquet",
                    "validation": run_dir / "validation" / "probabilities_step_05000.parquet",
                    "test": run_dir / "test" / "probabilities.parquet",
                }
                for partition, path in paths.items():
                    if not path.exists():
                        raise FileNotFoundError(path)
                    table = pd.read_parquet(path)
                    rows.extend(
                        _events_for_table(
                            table,
                            partition=partition,
                            fold=fold,
                            seed=seed,
                            model=model,
                            threshold=threshold,
                            seizures=seizures,
                            relative_by_recording=relative_by_recording,
                        )
                    )
                print(f"H2H3_EVENTS_DONE fold={fold} model={model} seed={seed}", flush=True)
    output = pd.DataFrame(rows)
    output_path = ROOT / "outputs" / "reports" / RUN_ID / EVENT_OUTPUT_NAME
    output.to_parquet(output_path, index=False)
    manifest = {
        "release_id": RUN_ID,
        "rows": int(len(output)),
        "partitions": {str(key): int(value) for key, value in output.partition.value_counts().to_dict().items()},
        "false_positive_rows": {
            str(key): int(value)
            for key, value in output[output.event_type == "fp"].partition.value_counts().to_dict().items()
        },
        "models": list(MODELS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "threshold_source": "each run's validation-selected threshold; test not used for selection",
        "output_sha256": _sha256(output_path),
    }
    manifest_name = output_path.with_name(output_path.stem + "_manifest.json")
    (manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
