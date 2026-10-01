"""Run the frozen H1 temporal false-alarm sharing analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event


ROOT = Path(__file__).resolve().parent.parent
EVENT_PATH = ROOT / "outputs" / "events" / "v3-compact-extension-v1_test_events.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = (("singlem", "cbramod"), ("singlem", "tcn_gat"), ("cbramod", "tcn_gat"))
SPLITS = (17, 42)
SEEDS = (17, 42, 3407)
MAX_ONSET_DIFF_S = 30.0
MIN_NULL_SHIFT_S = 300.0
MIN_BLOCK_S = 720.0
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EventIndex = dict[tuple[str, str], list[Event]]


def _event_index(frame: pd.DataFrame) -> EventIndex:
    index: EventIndex = {}
    for row in frame.itertuples(index=False):
        index.setdefault((str(row.patient), str(row.recording)), []).append(
            Event(float(row.start_s), float(row.end_s), float(row.peak_probability))
        )
    return index


def _greedy_matches(left: list[Event], right: list[Event]) -> list[tuple[Event, Event]]:
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


def _patient_f1_values(left: EventIndex, right: EventIndex, patients: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for patient in patients:
        n_left = n_right = matches = 0
        recordings = sorted(
            {recording for p, recording in left if p == patient}
            | {recording for p, recording in right if p == patient}
        )
        for recording in recordings:
            events_left = left.get((patient, recording), [])
            events_right = right.get((patient, recording), [])
            matches += len(_greedy_matches(events_left, events_right))
            n_left += len(events_left)
            n_right += len(events_right)
        values[patient] = 2.0 * matches / (n_left + n_right) if n_left + n_right else 0.0
    return values


def _pair_metrics(left: EventIndex, right: EventIndex, patients: tuple[str, ...]) -> dict[str, float]:
    values = _patient_f1_values(left, right, patients)
    vector = np.asarray([values[patient] for patient in patients], dtype=float)
    return {
        "patient_median_f1": float(np.median(vector)),
        "pooled_f1": float(np.mean(vector)),
        "left_events": float(sum(len(events) for events in left.values())),
        "right_events": float(sum(len(events) for events in right.values())),
        "matched_events": float(
            sum(
                len(_greedy_matches(left.get((patient, recording), []), right.get((patient, recording), [])))
                for patient in patients
                for recording in sorted(
                    {recording for p, recording in left if p == patient}
                    | {recording for p, recording in right if p == patient}
                )
            )
        ),
    }


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
    adjusted = np.empty(len(order), dtype=float)
    running = 1.0
    for rank in range(len(order), 0, -1):
        index = int(order[rank - 1])
        running = min(running, p_values[index] * len(order) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    events = pd.read_parquet(EVENT_PATH)
    split_patients = {
        split: tuple(json.loads((ROOT / "manifests" / "splits" / f"split_{split}.json").read_text())["test"])
        for split in SPLITS
    }
    durations = (
        events[["recording", "recording_duration_s"]]
        .drop_duplicates("recording")
        .set_index("recording")["recording_duration_s"]
        .astype(float)
        .to_dict()
    )
    rng = np.random.default_rng(20260811)
    pair_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for split in SPLITS:
        patients = split_patients[split]
        for seed in SEEDS:
            for left_model, right_model in PAIRS:
                left_frame = events[
                    (events.split == split)
                    & (events.seed == seed)
                    & (events.model == left_model)
                    & (events.event_type == "fp")
                ]
                right_frame = events[
                    (events.split == split)
                    & (events.seed == seed)
                    & (events.model == right_model)
                    & (events.event_type == "fp")
                ]
                left_index = _event_index(left_frame)
                right_index = _event_index(right_frame)
                observed = _pair_metrics(left_index, right_index, patients)
                patient_values = _patient_f1_values(left_index, right_index, patients)
                vector = np.asarray([patient_values[patient] for patient in patients], dtype=float)
                sampled_medians = np.empty(N_BOOTSTRAP, dtype=float)
                for index in range(N_BOOTSTRAP):
                    sampled = rng.integers(0, len(vector), size=len(vector))
                    sampled_medians[index] = float(np.median(vector[sampled]))
                bootstrap_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        "observed_patient_median_f1": float(np.median(vector)),
                        "bootstrap_median_q025": float(np.quantile(sampled_medians, 0.025)),
                        "bootstrap_median_q975": float(np.quantile(sampled_medians, 0.975)),
                        "bootstrap_replicates": N_BOOTSTRAP,
                    }
                )
                valid_recordings = sorted(
                    recording
                    for recording in ({recording for _, recording in left_index} | {recording for _, recording in right_index})
                    if durations.get(recording, 0.0) >= MIN_BLOCK_S
                )
                null_values: list[float] = []
                for permutation in range(N_PERMUTATIONS):
                    offsets = {
                        recording: float(rng.uniform(MIN_NULL_SHIFT_S, durations[recording] - MIN_NULL_SHIFT_S))
                        for recording in valid_recordings
                    }
                    shifted_right = _shift_index(
                        {key: value for key, value in right_index.items() if key[1] in valid_recordings},
                        offsets,
                        durations,
                    )
                    null_metric = _pair_metrics(left_index, shifted_right, patients)
                    null_values.append(null_metric["patient_median_f1"])
                    null_rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "left_model": left_model,
                            "right_model": right_model,
                            "permutation": permutation,
                            "patient_median_f1": null_metric["patient_median_f1"],
                        }
                    )
                null_array = np.asarray(null_values, dtype=float)
                p_value = (1.0 + float(np.sum(null_array >= observed["patient_median_f1"]))) / (N_PERMUTATIONS + 1.0)
                pair_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        "observed_patient_median_f1": observed["patient_median_f1"],
                        "null_patient_median_f1": float(np.median(null_array)),
                        "null_f1_q025": float(np.quantile(null_array, 0.025)),
                        "null_f1_q975": float(np.quantile(null_array, 0.975)),
                        "observed_minus_null_f1": observed["patient_median_f1"] - float(np.median(null_array)),
                        "p_raw": p_value,
                        "observed_pooled_f1": observed["pooled_f1"],
                        "left_fp_events": int(observed["left_events"]),
                        "right_fp_events": int(observed["right_events"]),
                        "matched_fp_events": int(observed["matched_events"]),
                        "permutations": N_PERMUTATIONS,
                        "min_null_shift_s": MIN_NULL_SHIFT_S,
                        "min_block_s": MIN_BLOCK_S,
                    }
                )
    pair_df = pd.DataFrame(pair_rows)
    pair_df["q_fdr"] = _bh_fdr(pair_df.p_raw.tolist())
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    null_df = pd.DataFrame(null_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pair_path = REPORT_DIR / "h1_pairwise.csv"
    bootstrap_path = REPORT_DIR / "h1_bootstrap.csv"
    null_path = REPORT_DIR / "h1_nulls.parquet"
    pair_df.to_csv(pair_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    null_df.to_parquet(null_path, index=False)
    summary = {
        "release_id": "v3-compact-extension-v1",
        "analysis": "H1_temporal_false_alarm_sharing",
        "splits": list(SPLITS),
        "seeds": list(SEEDS),
        "pairs": [list(pair) for pair in PAIRS],
        "primary_metric": "patient median pairwise FP onset-match F1 within +/-30s",
        "null": "within-recording circular shift; offsets >=300s; blocks <720s excluded",
        "permutations": N_PERMUTATIONS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "pairwise_sha256": _sha256(pair_path),
        "bootstrap_sha256": _sha256(bootstrap_path),
        "nulls_sha256": _sha256(null_path),
        "pairwise": pair_rows,
    }
    (REPORT_DIR / "h1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(pair_df.to_string(index=False))
    print(json.dumps({"pairwise": str(pair_path), "bootstrap": str(bootstrap_path), "nulls": str(null_path)}, indent=2))


if __name__ == "__main__":
    main()
