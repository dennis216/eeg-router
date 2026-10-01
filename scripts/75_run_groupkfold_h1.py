"""Run H1 on the frozen standard GroupKFold out-of-fold event table.

The OOF table contains one outer-test prediction for every patient, model and
seed.  This script never reselects thresholds and never uses the old greedy
queue.  It compares false-positive onset sharing with within-recording
circular-shift nulls and uses patients (not windows or events) as the
bootstrap unit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event


ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "v3-groupkfold-confirmatory-v1"
EVENT_PATH = ROOT / "outputs" / "reports" / RUN_ID / "oof_events.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = (("singlem", "cbramod"), ("singlem", "tcn_gat"), ("cbramod", "tcn_gat"))
SEEDS = (17, 42, 3407)
MAX_ONSET_DIFF_S = 30.0
MIN_NULL_SHIFT_S = 300.0
MIN_BLOCK_S = 720.0
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
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


def _patient_f1_values(
    left: EventIndex, right: EventIndex, patients: tuple[str, ...]
) -> dict[str, float]:
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


def _pair_metrics(
    left: EventIndex, right: EventIndex, patients: tuple[str, ...]
) -> dict[str, float]:
    values = _patient_f1_values(left, right, patients)
    vector = np.asarray([values[patient] for patient in patients], dtype=float)
    matched = sum(
        len(_greedy_matches(left.get((patient, recording), []), right.get((patient, recording), [])))
        for patient in patients
        for recording in sorted(
            {recording for p, recording in left if p == patient}
            | {recording for p, recording in right if p == patient}
        )
    )
    return {
        "patient_median_f1": float(np.median(vector)),
        "patient_mean_f1": float(np.mean(vector)),
        "left_events": float(sum(len(events) for events in left.values())),
        "right_events": float(sum(len(events) for events in right.values())),
        "matched_events": float(matched),
    }


def _shift_index(index: EventIndex, offsets: dict[str, float], durations: dict[str, float]) -> EventIndex:
    shifted: EventIndex = {}
    for (patient, recording), events in index.items():
        duration = durations[recording]
        offset = offsets.get(recording, 0.0)
        shifted[(patient, recording)] = [
            Event(
                (event.start_s + offset) % duration,
                min(
                    duration,
                    (event.start_s + offset) % duration
                    + min(event.end_s - event.start_s, duration),
                ),
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
    fp = events[events.event_type == "fp"].copy()
    patients = tuple(sorted(events.patient.astype(str).unique()))
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    rng = np.random.default_rng(20260821)

    pair_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        for left_model, right_model in PAIRS:
            left_frame = fp[(fp.seed == seed) & (fp.model == left_model)]
            right_frame = fp[(fp.seed == seed) & (fp.model == right_model)]
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
                    "seed": seed,
                    "left_model": left_model,
                    "right_model": right_model,
                    "patients": len(patients),
                    "observed_patient_median_f1": observed["patient_median_f1"],
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
                    recording: float(
                        rng.uniform(MIN_NULL_SHIFT_S, durations[recording] - MIN_NULL_SHIFT_S)
                    )
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
                    "seed": seed,
                    "left_model": left_model,
                    "right_model": right_model,
                    "patients": len(patients),
                    "observed_patient_median_f1": observed["patient_median_f1"],
                    "null_patient_median_f1": float(np.median(null_array)),
                    "null_f1_q025": float(np.quantile(null_array, 0.025)),
                    "null_f1_q975": float(np.quantile(null_array, 0.975)),
                    "observed_minus_null_f1": observed["patient_median_f1"] - float(np.median(null_array)),
                    "p_raw": p_value,
                    "observed_patient_mean_f1": observed["patient_mean_f1"],
                    "left_fp_events": int(observed["left_events"]),
                    "right_fp_events": int(observed["right_events"]),
                    "matched_fp_events": int(observed["matched_events"]),
                    "permutations": N_PERMUTATIONS,
                    "min_null_shift_s": MIN_NULL_SHIFT_S,
                    "min_block_s": MIN_BLOCK_S,
                }
            )

            for fold in sorted(events.fold.unique()):
                fold_patients = tuple(sorted(events.loc[events.fold == fold, "patient"].astype(str).unique()))
                fold_left = _event_index(left_frame[left_frame.fold == fold])
                fold_right = _event_index(right_frame[right_frame.fold == fold])
                fold_metric = _pair_metrics(fold_left, fold_right, fold_patients)
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        "patients": len(fold_patients),
                        **fold_metric,
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pair_df["q_fdr"] = _bh_fdr(pair_df.p_raw.tolist())
    pair_df.to_csv(REPORT_DIR / "h1_groupkfold_pairwise.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REPORT_DIR / "h1_groupkfold_by_fold.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_DIR / "h1_groupkfold_bootstrap.csv", index=False)
    pd.DataFrame(null_rows).to_parquet(REPORT_DIR / "h1_groupkfold_nulls.parquet", index=False)
    summary = {
        "release_id": RUN_ID,
        "analysis": "H1_temporal_false_alarm_sharing_on_outer_test_OOF",
        "event_source": str(EVENT_PATH.relative_to(ROOT)),
        "event_source_sha256": _sha256(EVENT_PATH),
        "patients": list(patients),
        "seeds": list(SEEDS),
        "pairs": [list(pair) for pair in PAIRS],
        "primary_metric": "patient median pairwise false-positive onset-match F1 within +/-30s",
        "null": "within-recording circular shift; offsets >=300s; recordings <720s excluded",
        "permutations": N_PERMUTATIONS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "test_threshold_selection": "forbidden; OOF thresholds remain validation-locked",
        "outputs": [
            "h1_groupkfold_pairwise.csv",
            "h1_groupkfold_by_fold.csv",
            "h1_groupkfold_bootstrap.csv",
            "h1_groupkfold_nulls.parquet",
        ],
        "pairwise": pair_rows,
    }
    (REPORT_DIR / "h1_groupkfold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pair_df.to_string(index=False))
    print(json.dumps({"pairwise": str(REPORT_DIR / "h1_groupkfold_pairwise.csv"), "nulls": str(REPORT_DIR / "h1_groupkfold_nulls.parquet")}, indent=2))


if __name__ == "__main__":
    main()
