"""Frozen, post-hoc H1 temporal-tolerance sensitivity analysis.

Reads only the GroupKFold OOF false-positive event table.  It never changes
models, thresholds, splits, or eventization. All six matching rules are run
as a fixed family and their p-values are BH-FDR corrected jointly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from bfa.evaluation.eventize import Event


ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "h1-temporal-tolerance-sensitivity-v3"
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
EVENT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "oof_events.parquet"
OUT = ROOT / "outputs" / "reports" / RUN_ID
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
SEEDS = (17, 42, 3407)
TOLERANCES = (5.0, 10.0, 15.0, 30.0, 60.0)
RULES = tuple([f"onset_pm_{int(t)}s" for t in TOLERANCES] + ["interval_overlap_strict"])
MIN_NULL_SHIFT_S = 300.0
MIN_BLOCK_S = 720.0
N_BOOTSTRAP = 2000


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


def event_index(frame: pd.DataFrame) -> dict[tuple[str, str], list[Event]]:
    result: dict[tuple[str, str], list[Event]] = {}
    for row in frame.itertuples(index=False):
        result.setdefault((str(row.patient), str(row.recording)), []).append(
            Event(float(row.start_s), float(row.end_s), float(row.peak_probability))
        )
    return result


def onset_matches(left: list[Event], right: list[Event], tolerance_s: float) -> int:
    left = sorted(left, key=lambda event: event.start_s)
    right = sorted(right, key=lambda event: event.start_s)
    i = j = matched = 0
    while i < len(left) and j < len(right):
        delta = left[i].start_s - right[j].start_s
        if abs(delta) <= tolerance_s:
            matched += 1
            i += 1
            j += 1
        elif delta < -tolerance_s:
            i += 1
        else:
            j += 1
    return matched


def overlap_matches(left: list[Event], right: list[Event]) -> int:
    """Exact maximum-cardinality one-to-one matching for positive overlap."""
    if not left or not right:
        return 0
    rows, columns = zip(*[
        (i, j)
        for i, first in enumerate(left)
        for j, second in enumerate(right)
        if first.start_s < second.end_s and second.start_s < first.end_s
    ]) if any(
        first.start_s < second.end_s and second.start_s < first.end_s
        for first in left for second in right
    ) else ((), ())
    if not rows:
        return 0
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(len(left), len(right)))
    return int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))


def matches(left: list[Event], right: list[Event], rule: str) -> int:
    if rule == "interval_overlap_strict":
        return overlap_matches(left, right)
    tolerance_s = float(rule.removeprefix("onset_pm_").removesuffix("s"))
    return onset_matches(left, right, tolerance_s)


def metric(
    left: dict[tuple[str, str], list[Event]],
    right: dict[tuple[str, str], list[Event]],
    patients: tuple[str, ...],
    rule: str,
) -> tuple[float, float, int]:
    values: list[float] = []
    total = 0
    for patient in patients:
        recordings = sorted({recording for p, recording in left if p == patient} | {recording for p, recording in right if p == patient})
        n_left = n_right = n_matched = 0
        for recording in recordings:
            first = left.get((patient, recording), [])
            second = right.get((patient, recording), [])
            n_left += len(first)
            n_right += len(second)
            n_matched += matches(first, second, rule)
        total += n_matched
        values.append(2 * n_matched / (n_left + n_right) if n_left + n_right else 0.0)
    return float(np.median(values)), float(np.mean(values)), total


def shifted(
    index: dict[tuple[str, str], list[Event]], offsets: dict[str, float], durations: dict[str, float]
) -> dict[tuple[str, str], list[Event]]:
    result: dict[tuple[str, str], list[Event]] = {}
    for (patient, recording), events in index.items():
        if recording not in offsets:
            continue
        duration, offset = durations[recording], offsets[recording]
        result[(patient, recording)] = [
            Event(
                (event.start_s + offset) % duration,
                min(duration, (event.start_s + offset) % duration + min(event.end_s - event.start_s, duration)),
                event.peak_probability,
            )
            for event in events
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    permutations = 5 if args.smoke else args.permutations
    output_dir = OUT / "_smoke" if args.smoke else OUT
    if (output_dir / "manifest.json").exists() and not args.smoke:
        raise RuntimeError(f"refusing to overwrite existing formal output: {output_dir}")
    if not EVENT_PATH.exists():
        raise FileNotFoundError(EVENT_PATH)
    # The smoke output lives under the formal namespace; only the formal
    # manifest check above controls whether a formal run may proceed.
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(EVENT_PATH)
    fp = events.loc[events.event_type == "fp"].copy()
    patients = tuple(sorted(events.patient.astype(str).unique()))
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id")["duration_s"].astype(float).to_dict()
    # Reuse the original H1 stream so its ±30s result is exactly reproducible.
    rng = np.random.default_rng(20260821)
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for left_model, right_model in PAIRS:
            left = event_index(fp[(fp.seed == seed) & (fp.model == left_model)])
            right = event_index(fp[(fp.seed == seed) & (fp.model == right_model)])
            observed = {rule: metric(left, right, patients, rule) for rule in RULES}
            # Consume precisely the bootstrap random draws made by the original
            # H1 script before it begins this pair's circular-shift null loop.
            # We do not re-report bootstrap CIs here; this preserves the exact
            # fixed offset stream for the inherited ±30-second primary rule.
            patient_values = np.asarray([
                metric(left, right, (patient,), "onset_pm_30s")[0]
                for patient in patients
            ])
            for _ in range(N_BOOTSTRAP):
                rng.integers(0, len(patient_values), size=len(patient_values))
            null = {rule: [] for rule in RULES}
            valid = sorted({recording for _, recording in left} | {recording for _, recording in right})
            valid = [recording for recording in valid if durations.get(recording, 0.0) >= MIN_BLOCK_S]
            for permutation in range(permutations):
                offsets = {recording: float(rng.uniform(MIN_NULL_SHIFT_S, durations[recording] - MIN_NULL_SHIFT_S)) for recording in valid}
                shifted_right = shifted(right, offsets, durations)
                for rule in RULES:
                    value = metric(left, shifted_right, patients, rule)[0]
                    null[rule].append(value)
                    null_rows.append({"seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "permutation": permutation, "patient_median_f1": value})
            for rule in RULES:
                values = np.asarray(null[rule])
                median, mean, matched = observed[rule]
                rows.append({"seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "patients": len(patients), "observed_patient_median_f1": median, "observed_patient_mean_f1": mean, "matched_fp_events": matched, "null_patient_median_f1": float(np.median(values)), "observed_minus_null_f1": median - float(np.median(values)), "null_f1_q025": float(np.quantile(values, .025)), "null_f1_q975": float(np.quantile(values, .975)), "p_raw": (1 + float(np.sum(values >= median))) / (permutations + 1), "permutations": permutations, "min_null_shift_s": MIN_NULL_SHIFT_S, "min_block_s": MIN_BLOCK_S})
            print(json.dumps({"seed": seed, "pair": [left_model, right_model], "done": True}), flush=True)
    table = pd.DataFrame(rows)
    table["q_fdr_all_54"] = bh_fdr(table.p_raw.tolist())
    table["q_fdr_within_rule"] = table.groupby("rule", group_keys=False).p_raw.transform(lambda series: bh_fdr(series.tolist()))
    table.to_csv(output_dir / "h1_tolerance_sensitivity_pairwise.csv", index=False)
    pd.DataFrame(null_rows).to_parquet(output_dir / "h1_tolerance_sensitivity_nulls.parquet", index=False)
    manifest = {"release_id": RUN_ID, "status": "smoke" if args.smoke else "complete_posthoc_robustness_analysis", "source_run": SOURCE_RUN, "event_source": str(EVENT_PATH.relative_to(ROOT)), "event_source_sha256": sha256(EVENT_PATH), "models": list(MODELS), "seeds": list(SEEDS), "rules": {"onset_tolerances_s": list(TOLERANCES), "interval_overlap_strict": "exact maximum-cardinality one-to-one positive interval overlap"}, "primary_existing_rule": "onset_pm_30s", "analysis_family": "all 54 rule-pair-seed tests BH-FDR corrected jointly", "null": "within-recording circular shift; offsets >=300s; recordings <720s excluded", "permutations": permutations, "null_rng_seed": 20260821, "rng_replay": "replays the original 2,000 bootstrap draws before each pair's null offsets", "supersedes": "h1-temporal-tolerance-sensitivity-v1 and v2 (preserved; not used for reporting)", "model_training": False, "threshold_selection": "unchanged; validation-only frozen OOF thresholds", "created_at_unix": time.time()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
