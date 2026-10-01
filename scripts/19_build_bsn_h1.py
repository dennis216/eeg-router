"""Build the frozen BSN pilot false-alarm event table and H1 temporal nulls.

This script reads only completed test probabilities and their validation-locked
thresholds. It never reselects a threshold or changes eventization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "runs" / "BSN-pilot-v1"
EVENT_DIR = ROOT / "outputs" / "events"
REPORT_DIR = ROOT / "outputs" / "reports"
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)
PAIRS = (("singlem", "cbramod"), ("singlem", "tcn_gat"), ("cbramod", "tcn_gat"))
MAX_ONSET_DIFF_S = 30.0
MIN_NULL_SHIFT_S = 300.0
MIN_BLOCK_S = 720.0
N_PERMUTATIONS = 1000


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


def _extract_events(
    model: str,
    seed: int,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> list[dict[str, object]]:
    run_dir = RUN_ROOT / model / f"split17_seed{seed}_main"
    completed = json.loads((run_dir / "completed.json").read_text())
    threshold = float(completed["test"]["score"]["threshold"])
    table = pd.read_parquet(run_dir / "test" / "probabilities.parquet")
    rows: list[dict[str, object]] = []
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
        patient = str(ordered.patient.iloc[0])
        duration = float(
            recordings.loc[recordings.recording_id == recording, "duration_s"].iloc[0]
        )
        for index, event in enumerate(predictions):
            pair = matched_by_prediction.get(index)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "patient": patient,
                    "recording": str(recording),
                    "recording_duration_s": duration,
                    "start_s": event.start_s,
                    "end_s": event.end_s,
                    "peak_probability": event.peak_probability,
                    "event_type": "tp" if pair is not None else "fp",
                    "truth_index": pair.truth_index if pair is not None else None,
                    "truth_overlap_s": pair.overlap_s if pair is not None else 0.0,
                    "threshold": threshold,
                }
            )
    return rows


def _greedy_matches(left: list[Event], right: list[Event]) -> list[tuple[Event, Event]]:
    """Maximum-cardinality onset matching for sorted events within 30 seconds."""
    left = sorted(left, key=lambda event: event.start_s)
    right = sorted(right, key=lambda event: event.start_s)
    i = j = 0
    pairs: list[tuple[Event, Event]] = []
    while i < len(left) and j < len(right):
        delta = left[i].start_s - right[j].start_s
        if abs(delta) <= MAX_ONSET_DIFF_S:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif delta < -MAX_ONSET_DIFF_S:
            i += 1
        else:
            j += 1
    return pairs


def _event_from_row(row: pd.Series) -> Event:
    return Event(float(row.start_s), float(row.end_s), float(row.peak_probability))


EventIndex = dict[tuple[str, str], list[Event]]


def _event_index(frame: pd.DataFrame) -> EventIndex:
    index: EventIndex = {}
    for row in frame.itertuples(index=False):
        index.setdefault((str(row.patient), str(row.recording)), []).append(
            Event(float(row.start_s), float(row.end_s), float(row.peak_probability))
        )
    return index


def _pair_metric_index(left: EventIndex, right: EventIndex, patients: tuple[str, ...]) -> dict[str, float]:
    patient_f1: list[float] = []
    patient_iou: list[float] = []
    patient_onset: list[float] = []
    patient_match_rate: list[float] = []
    total_left = total_right = total_matches = 0
    for patient in patients:
        n_left = n_right = matches = 0
        ious: list[float] = []
        onsets: list[float] = []
        recordings = sorted(
            {recording for p, recording in left if p == patient}
            | {recording for p, recording in right if p == patient}
        )
        for recording in recordings:
            events_a = left.get((patient, recording), [])
            events_b = right.get((patient, recording), [])
            pairs = _greedy_matches(events_a, events_b)
            n_left += len(events_a)
            n_right += len(events_b)
            matches += len(pairs)
            for x, y in pairs:
                overlap = max(0.0, min(x.end_s, y.end_s) - max(x.start_s, y.start_s))
                union = max(x.end_s, y.end_s) - min(x.start_s, y.start_s)
                ious.append(overlap / union if union else 0.0)
                onsets.append(abs(x.start_s - y.start_s))
        total_left += n_left
        total_right += n_right
        total_matches += matches
        patient_f1.append(2 * matches / (n_left + n_right) if n_left + n_right else 0.0)
        patient_match_rate.append(matches / min(n_left, n_right) if min(n_left, n_right) else 0.0)
        if ious:
            patient_iou.append(float(np.mean(ious)))
            patient_onset.append(float(np.mean(onsets)))
        else:
            patient_iou.append(0.0)
            patient_onset.append(float("nan"))
    return {
        "patient_median_f1": float(np.median(patient_f1)),
        "patient_median_match_rate": float(np.median(patient_match_rate)),
        "patient_median_iou": float(np.median(patient_iou)),
        "patient_median_onset_diff_s": float(np.nanmedian(patient_onset)) if np.isfinite(patient_onset).any() else float("nan"),
        "pooled_f1": 2 * total_matches / (total_left + total_right) if total_left + total_right else 0.0,
        "left_events": float(total_left),
        "right_events": float(total_right),
        "matched_events": float(total_matches),
    }


def _patient_f1_values(left: EventIndex, right: EventIndex, patients: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for patient in patients:
        n_left = n_right = matches = 0
        recordings = sorted(
            {recording for p, recording in left if p == patient}
            | {recording for p, recording in right if p == patient}
        )
        for recording in recordings:
            events_a = left.get((patient, recording), [])
            events_b = right.get((patient, recording), [])
            matches += len(_greedy_matches(events_a, events_b))
            n_left += len(events_a)
            n_right += len(events_b)
        values[patient] = 2 * matches / (n_left + n_right) if n_left + n_right else 0.0
    return values


def _shift_index(index: EventIndex, offsets: dict[str, float], durations: dict[str, float]) -> EventIndex:
    shifted: EventIndex = {}
    for (patient, recording), events in index.items():
        duration = durations[recording]
        offset = offsets.get(recording, 0.0)
        shifted[(patient, recording)] = [
            Event(
                (event.start_s + offset) % duration,
                min(duration, (event.start_s + offset) % duration + min(event.end_s - event.start_s, duration)),
                event.peak_probability,
            )
            for event in events
        ]
    return shifted


def _bh_fdr(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank in range(len(order), 0, -1):
        index = int(order[rank - 1])
        running = min(running, p_values[index] * len(order) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    split = json.loads((ROOT / "manifests/splits/split_17.json").read_text())
    patients = tuple(split["test"])

    event_rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            event_rows.extend(_extract_events(model, seed, seizures, recordings))
    events = pd.DataFrame(event_rows)
    event_path = EVENT_DIR / "bsn_pilot_v1_test_events.parquet"
    events.to_parquet(event_path, index=False)

    pair_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260804)
    for seed in SEEDS:
        for left_model, right_model in PAIRS:
            left = events[(events.seed == seed) & (events.model == left_model) & (events.event_type == "fp")]
            right = events[(events.seed == seed) & (events.model == right_model) & (events.event_type == "fp")]
            left_index = _event_index(left)
            right_index = _event_index(right)
            observed = _pair_metric_index(left_index, right_index, patients)
            null_metrics: list[dict[str, float]] = []
            valid_recordings = sorted({recording for _, recording in left_index} | {recording for _, recording in right_index})
            duration_by_recording = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
            for permutation in range(N_PERMUTATIONS):
                offsets: dict[str, float] = {}
                for recording in valid_recordings:
                    if recording not in duration_by_recording:
                        continue
                    duration = duration_by_recording[recording]
                    if duration >= MIN_BLOCK_S:
                        offsets[recording] = float(rng.uniform(MIN_NULL_SHIFT_S, duration - MIN_NULL_SHIFT_S))
                    else:
                        offsets[recording] = 0.0
                shifted_right = _shift_index(right_index, offsets, duration_by_recording)
                metric = _pair_metric_index(left_index, shifted_right, patients)
                null_metrics.append(metric)
                null_rows.append(
                    {
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        "permutation": permutation,
                        **metric,
                    }
                )
            null_f1 = np.asarray([row["patient_median_f1"] for row in null_metrics], dtype=float)
            null_iou = np.asarray([row["patient_median_iou"] for row in null_metrics], dtype=float)
            p = (1 + int(np.sum(null_f1 >= observed["patient_median_f1"]))) / (N_PERMUTATIONS + 1)
            pair_rows.append(
                {
                    "seed": seed,
                    "left_model": left_model,
                    "right_model": right_model,
                    "observed_patient_median_f1": observed["patient_median_f1"],
                    "null_patient_median_f1": float(np.median(null_f1)),
                    "null_f1_q025": float(np.quantile(null_f1, 0.025)),
                    "null_f1_q975": float(np.quantile(null_f1, 0.975)),
                    "observed_minus_null_f1": observed["patient_median_f1"] - float(np.median(null_f1)),
                    "p_raw": p,
                    "observed_patient_median_iou": observed["patient_median_iou"],
                    "null_patient_median_iou": float(np.median(null_iou)),
                    "observed_patient_median_onset_diff_s": observed["patient_median_onset_diff_s"],
                    "observed_pooled_f1": observed["pooled_f1"],
                    "left_fp_events": int(observed["left_events"]),
                    "right_fp_events": int(observed["right_events"]),
                    "matched_fp_events": int(observed["matched_events"]),
                    "permutations": N_PERMUTATIONS,
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_df["q_fdr"] = _bh_fdr(pair_df.p_raw.tolist())
    pair_path = REPORT_DIR / "bsn_pilot_v1_h1_pairwise.csv"
    pair_df.to_csv(pair_path, index=False)
    null_path = REPORT_DIR / "bsn_pilot_v1_h1_nulls.parquet"
    pd.DataFrame(null_rows).to_parquet(null_path, index=False)

    summary = {
        "analysis": "H1_temporal_false_alarm_sharing",
        "pilot": "BSN-pilot-v1",
        "split": 17,
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "event_definition": "validation_locked_threshold; causal EMA alpha=1/3; min duration 4s; merge gap 10s; refractory 30s",
        "primary_metric": "patient median pairwise FP onset-match F1 within +/-30s",
        "null": "recording-block circular shift of right-model FP events; offsets >=300s; blocks <720s excluded",
        "permutations": N_PERMUTATIONS,
        "event_table_sha256": _sha256(event_path),
        "pairwise_csv_sha256": _sha256(pair_path),
        "nulls_sha256": _sha256(null_path),
        "pairwise": pair_rows,
    }
    (REPORT_DIR / "bsn_pilot_v1_h1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(pair_df.to_string(index=False))
    print(json.dumps({"events": len(events), "event_table": str(event_path), "pairwise": str(pair_path)}, indent=2))


if __name__ == "__main__":
    main()
