"""Frozen H1 robustness control that preserves local recording context.

The original H1 null circularly shifts every false-positive event through a
whole recording. This script keeps each event onset inside its original
20-minute recording block, so slow context (for example sleep or recording
quality changes) remains matched between observed and null event streams.

It only reads the frozen GroupKFold OOF event table. No model, threshold,
eventization parameter, split, or checkpoint is changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\root\b_false_alarm_atlas")
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "research-a-robustness-corrections-v1"
EVENT_PATH = SOURCE_ROOT / "outputs" / "reports" / SOURCE_RUN / "oof_events.parquet"
RECORDINGS_PATH = SOURCE_ROOT / "manifests" / "recordings.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = (("singlem", "cbramod"), ("singlem", "tcn_gat"), ("cbramod", "tcn_gat"))
SEEDS = (17, 42, 3407)
TOLERANCE_S = 30.0
BLOCK_S = 1200.0
MIN_BLOCK_S = 720.0
MIN_SHIFT_S = 300.0
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000
RNG_SEED = 20260826


@dataclass(frozen=True)
class Event:
    start: float
    end: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(values: list[float]) -> list[float]:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = int(order[rank - 1])
        running = min(running, values[index] * len(values) / rank)
        result[index] = running
    return result.tolist()


def index_events(frame: pd.DataFrame) -> dict[tuple[str, str], list[Event]]:
    result: dict[tuple[str, str], list[Event]] = {}
    for row in frame.itertuples(index=False):
        result.setdefault((str(row.patient), str(row.recording)), []).append(Event(float(row.start_s), float(row.end_s)))
    return result


def eligible_block(start: float, duration: float) -> tuple[int, float, float] | None:
    block_index = int(start // BLOCK_S)
    block_start = block_index * BLOCK_S
    block_end = min(block_start + BLOCK_S, duration)
    if block_end - block_start < MIN_BLOCK_S:
        return None
    return block_index, block_start, block_end


def filter_to_eligible_blocks(index: dict[tuple[str, str], list[Event]], durations: dict[str, float]) -> dict[tuple[str, str], list[Event]]:
    result: dict[tuple[str, str], list[Event]] = {}
    for key, events in index.items():
        duration = durations.get(key[1])
        if duration is None:
            continue
        kept = [event for event in events if eligible_block(event.start, duration) is not None]
        if kept:
            result[key] = kept
    return result


def onset_matches(left: list[Event], right: list[Event]) -> int:
    left, right = sorted(left, key=lambda x: x.start), sorted(right, key=lambda x: x.start)
    i = j = matches = 0
    while i < len(left) and j < len(right):
        difference = left[i].start - right[j].start
        if abs(difference) <= TOLERANCE_S:
            matches += 1
            i += 1
            j += 1
        elif difference < -TOLERANCE_S:
            i += 1
        else:
            j += 1
    return matches


def patient_values(left: dict[tuple[str, str], list[Event]], right: dict[tuple[str, str], list[Event]], patients: tuple[str, ...]) -> np.ndarray:
    values: list[float] = []
    for patient in patients:
        keys = sorted({key for key in left if key[0] == patient} | {key for key in right if key[0] == patient})
        n_left = n_right = n_matches = 0
        for key in keys:
            l_events, r_events = left.get(key, []), right.get(key, [])
            n_left += len(l_events)
            n_right += len(r_events)
            n_matches += onset_matches(l_events, r_events)
        values.append(2 * n_matches / (n_left + n_right) if n_left + n_right else 0.0)
    return np.asarray(values, dtype=float)


def shift_within_blocks(index: dict[tuple[str, str], list[Event]], durations: dict[str, float], rng: np.random.Generator) -> dict[tuple[str, str], list[Event]]:
    offsets: dict[tuple[str, int], float] = {}
    shifted: dict[tuple[str, str], list[Event]] = {}
    for (patient, recording), events in index.items():
        duration = durations[recording]
        target: list[Event] = []
        for event in events:
            block = eligible_block(event.start, duration)
            if block is None:
                continue
            block_index, block_start, block_end = block
            offset_key = (recording, block_index)
            if offset_key not in offsets:
                offsets[offset_key] = float(rng.uniform(MIN_SHIFT_S, (block_end - block_start) - MIN_SHIFT_S))
            local_start = event.start - block_start
            shifted_start = block_start + ((local_start + offsets[offset_key]) % (block_end - block_start))
            target.append(Event(shifted_start, min(block_end, shifted_start + (event.end - event.start))))
        if target:
            shifted[(patient, recording)] = target
    return shifted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "h1_local_context_null_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite formal result: {manifest_path}")
    events = pd.read_parquet(EVENT_PATH)
    fp = events.loc[events.event_type == "fp"].copy()
    patients = tuple(sorted(events.patient.astype(str).unique()))
    recordings = pd.read_parquet(RECORDINGS_PATH)
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for left_model, right_model in PAIRS:
            left = filter_to_eligible_blocks(index_events(fp[(fp.seed == seed) & (fp.model == left_model)]), durations)
            right = filter_to_eligible_blocks(index_events(fp[(fp.seed == seed) & (fp.model == right_model)]), durations)
            observed = patient_values(left, right, patients)
            null = np.empty(N_PERMUTATIONS, dtype=float)
            for replicate in range(N_PERMUTATIONS):
                null[replicate] = float(np.median(patient_values(left, shift_within_blocks(right, durations, rng), patients)))
            samples = np.empty(N_BOOTSTRAP, dtype=float)
            for replicate in range(N_BOOTSTRAP):
                samples[replicate] = float(np.median(observed[rng.integers(0, len(observed), size=len(observed))]))
            rows.append({
                "seed": seed, "left_model": left_model, "right_model": right_model, "patients": len(patients),
                "observed_patient_median_f1": float(np.median(observed)), "null_patient_median_f1": float(np.median(null)),
                "observed_minus_null_f1": float(np.median(observed) - np.median(null)),
                "null_f1_q025": float(np.quantile(null, .025)), "null_f1_q975": float(np.quantile(null, .975)),
                "p_raw": (1 + float(np.sum(null >= np.median(observed)))) / (N_PERMUTATIONS + 1),
                "eligible_left_fp_events": int(sum(len(v) for v in left.values())), "eligible_right_fp_events": int(sum(len(v) for v in right.values())),
            })
            bootstrap_rows.append({"seed": seed, "left_model": left_model, "right_model": right_model, "q025": float(np.quantile(samples, .025)), "q975": float(np.quantile(samples, .975)), "replicates": N_BOOTSTRAP})
            print(json.dumps({"analysis": "h1_local_context", "seed": seed, "pair": [left_model, right_model], "complete": True}), flush=True)
    table = pd.DataFrame(rows)
    table["q_fdr"] = bh_fdr(table.p_raw.tolist())
    table.to_csv(OUT / "h1_local_context_null_pairwise.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "h1_local_context_null_bootstrap.csv", index=False)
    manifest = {
        "analysis_id": "h1-local-context-null-v1", "status": "complete", "source_run": SOURCE_RUN,
        "event_source_sha256": sha256(EVENT_PATH), "recording_manifest_sha256": sha256(RECORDINGS_PATH),
        "models": list(MODELS), "seeds": list(SEEDS), "patients": list(patients),
        "primary_metric": "patient-median false-positive onset-match F1 within +/-30 seconds",
        "local_null": {"block_seconds": BLOCK_S, "minimum_block_seconds": MIN_BLOCK_S, "minimum_within_block_shift_seconds": MIN_SHIFT_S, "description": "right-model false-positive onset remains inside its original local recording block"},
        "permutations": N_PERMUTATIONS, "patient_bootstrap_replicates": N_BOOTSTRAP,
        "test_used_for_selection": False, "threshold_source": "existing validation-only frozen OOF thresholds", "model_training": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
