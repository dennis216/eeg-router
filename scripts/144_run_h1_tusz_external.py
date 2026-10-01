"""Run the frozen H1 external descriptive replication on TUSZ.

This analysis reads the completed TUSZ external inference event table.  It
does not train models, change thresholds, or use TUSZ labels for selection.
Because every TUSZ patient was evaluated by all five frozen checkpoints, the
unit of analysis is pair x fold x seed (45 units), not an OOF concatenation.
The primary comparison is patient-median false-positive onset-match F1 within
/-30 s versus a within-recording circular-shift null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
SOURCE_RUN = "v3-groupkfold-confirmatory-v1"
OUT_ID = "h1-external-tusz-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / OUT_ID
SOURCE_EVENTS = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
SOURCE_RUN_METRICS = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_run_metrics.csv"
SOURCE_SCORING_MANIFEST = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_scoring_manifest.json"
COHORT_MANIFEST = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
RULES = ("onset_pm_5s", "onset_pm_15s", "onset_pm_30s", "onset_pm_60s", "interval_overlap_strict")
PRIMARY_RULE = "onset_pm_30s"
MIN_NULL_SHIFT_S = 300.0
MIN_BLOCK_S = 720.0
DEFAULT_PERMUTATIONS = 1000
DEFAULT_BOOTSTRAP = 2000
RNG_SEED = 20260828


class Event:
    __slots__ = ("start_s", "end_s", "peak_probability")

    def __init__(self, start_s: float, end_s: float, peak_probability: float = 0.0):
        self.start_s = float(start_s)
        self.end_s = float(end_s)
        self.peak_probability = float(peak_probability)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values, dtype=float))
    result = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = int(order[rank - 1])
        running = min(running, float(values[index]) * len(values) / rank)
        result[index] = running
    return result.tolist()


def onset_matches(left: list[Event], right: list[Event], tolerance_s: float) -> int:
    first = sorted(left, key=lambda event: event.start_s)
    second = sorted(right, key=lambda event: event.start_s)
    i = j = matched = 0
    while i < len(first) and j < len(second):
        delta = first[i].start_s - second[j].start_s
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
    edges = [
        (i, j)
        for i, first in enumerate(left)
        for j, second in enumerate(right)
        if first.start_s < second.end_s and second.start_s < first.end_s
    ]
    if not edges:
        return 0
    rows, cols = zip(*edges)
    graph = csr_matrix((np.ones(len(edges), dtype=np.int8), (rows, cols)), shape=(len(left), len(right)))
    return int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))


def match_count(left: list[Event], right: list[Event], rule: str) -> int:
    if rule == "interval_overlap_strict":
        return overlap_matches(left, right)
    tolerance = float(rule.removeprefix("onset_pm_").removesuffix("s"))
    return onset_matches(left, right, tolerance)


def patient_metric(
    left: dict[tuple[str, str], list[Event]],
    right: dict[tuple[str, str], list[Event]],
    patient_records: dict[str, list[tuple[str, str]]],
    patients: list[str],
    rule: str,
) -> tuple[float, float, int, int, int, np.ndarray]:
    values: list[float] = []
    matched_total = n_left_total = n_right_total = 0
    for patient in patients:
        n_left = n_right = matched = 0
        for key in patient_records[patient]:
            first = left.get(key, [])
            second = right.get(key, [])
            n_left += len(first)
            n_right += len(second)
            matched += match_count(first, second, rule)
        matched_total += matched
        n_left_total += n_left
        n_right_total += n_right
        values.append(2.0 * matched / (n_left + n_right) if n_left + n_right else 0.0)
    vector = np.asarray(values, dtype=float)
    return float(np.median(vector)), float(np.mean(vector)), matched_total, n_left_total, n_right_total, vector


def shift_index(
    index: dict[tuple[str, str], list[Event]], offsets: dict[tuple[str, str], float], durations: dict[tuple[str, str], float]
) -> dict[tuple[str, str], list[Event]]:
    shifted: dict[tuple[str, str], list[Event]] = {}
    for key, events in index.items():
        if key not in offsets:
            continue
        duration = durations[key]
        offset = offsets[key]
        out: list[Event] = []
        for event in events:
            start = (event.start_s + offset) % duration
            # Keep the same truncation convention as the frozen CHB H1 null.
            end = min(duration, start + min(event.end_s - event.start_s, duration))
            out.append(Event(start, end, event.peak_probability))
        shifted[key] = out
    return shifted


def event_index(frame: pd.DataFrame, model: str, fold: int, seed: int) -> dict[tuple[str, str], list[Event]]:
    subset = frame[(frame.model == model) & (frame.fold == fold) & (frame.seed == seed) & (~frame.is_true_positive)]
    result: dict[tuple[str, str], list[Event]] = {}
    for row in subset.itertuples(index=False):
        key = (str(row.patient_id), str(row.recording))
        result.setdefault(key, []).append(Event(row.start_s, row.end_s, row.peak_probability))
    for events in result.values():
        events.sort(key=lambda event: event.start_s)
    return result


def validate_sources(events: pd.DataFrame, cohort: dict, scoring: dict) -> dict[str, object]:
    required = {"model", "fold", "seed", "patient_id", "recording", "event_index", "start_s", "end_s", "is_true_positive"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise RuntimeError(f"TUSZ events missing required columns: {missing}")
    if int(scoring.get("inference_runs_scored", -1)) != 45:
        raise RuntimeError("TUSZ source scoring does not contain all 45 runs")
    if bool(scoring.get("threshold_retuning", True)) or bool(scoring.get("model_training", True)):
        raise RuntimeError("TUSZ source indicates threshold retuning or model training")
    if str(scoring.get("threshold_source", "")).find("validation-only") < 0:
        raise RuntimeError("TUSZ threshold source is not validation-only")
    cohort_records = cohort.get("records", [])
    duration_map = {(str(row["patient_id"]), str(row["relative_edf"])): float(row["duration_s"]) for row in cohort_records}
    event_keys = set(zip(events.patient_id.astype(str), events.recording.astype(str)))
    unknown = sorted(event_keys - set(duration_map))
    if unknown:
        raise RuntimeError(f"TUSZ events contain {len(unknown)} unknown cohort recording keys")
    duplicate_rows = int(events.duplicated(["model", "fold", "seed", "patient_id", "recording", "event_index"]).sum())
    if duplicate_rows:
        raise RuntimeError(f"TUSZ events contain {duplicate_rows} duplicate event keys")
    expected_patients = sorted(str(row["patient_id"]) for row in cohort.get("records", []))
    expected_patients = sorted(set(expected_patients))
    return {
        "event_rows": int(len(events)),
        "event_fp_rows": int((~events.is_true_positive).sum()),
        "event_patient_count": int(events.patient_id.astype(str).nunique()),
        "cohort_patient_count": len(expected_patients),
        "cohort_record_count": len(cohort_records),
        "eligible_record_count_duration_ge_720s": int(sum(duration >= MIN_BLOCK_S for duration in duration_map.values())),
        "unknown_record_keys": 0,
        "duplicate_event_keys": duplicate_rows,
    }


def make_plot(pairwise: pd.DataFrame, output: Path) -> None:
    primary = pairwise[pairwise.rule == PRIMARY_RULE].copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    colors = {"singlem|cbramod": "#2455A4", "singlem|tcn_gat": "#008C95", "cbramod|tcn_gat": "#B56B1F"}
    for _, row in primary.iterrows():
        key = f"{row.left_model}|{row.right_model}"
        significant = bool(row.q_fdr_primary_45 < 0.05)
        ax.scatter(row.null_patient_median_f1, row.observed_patient_median_f1, s=58, color=colors.get(key, "#4D4D4D"), alpha=0.95 if significant else 0.45, edgecolor="white", linewidth=0.5)
    lo = min(float(primary.null_patient_median_f1.min()), float(primary.observed_patient_median_f1.min()))
    hi = max(float(primary.null_patient_median_f1.max()), float(primary.observed_patient_median_f1.max()))
    pad = 0.03 * max(hi - lo, 0.1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#777777", linewidth=1.2, label="Observed = null")
    ax.set_xlabel("Circular-shift null patient-median F1")
    ax.set_ylabel("Observed patient-median F1 (±30 s)")
    ax.set_title("TUSZ H1 external descriptive replication\n45 fold × seed × model-pair comparisons")
    ax.grid(True, color="#E8EDF3", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    audit: dict[str, object],
    pairwise: pd.DataFrame,
    summary_pair_seed: pd.DataFrame,
    summary_pair: pd.DataFrame,
) -> None:
    primary = pairwise[pairwise.rule == PRIMARY_RULE]
    significant = int((primary.q_fdr_primary_45 < 0.05).sum())
    lines = [
        "# H1 External TUSZ Replication Report",
        "",
        "Status: complete external descriptive replication; no training or threshold retuning.",
        "",
        "## Design",
        "",
        f"- Analysis unit: 3 model pairs × 5 folds × 3 seeds = {len(primary)} comparisons. All five frozen checkpoints were run on every TUSZ patient, so fold outputs were not concatenated into a pseudo-OOF table.",
        f"- Primary metric: patient-median one-to-one false-positive onset-match F1 within ±30 s.",
        f"- Null: within-recording circular shift of the right-hand model by ≥{int(MIN_NULL_SHIFT_S)} s; recordings shorter than {int(MIN_BLOCK_S)} s excluded from the primary null/observed analysis.",
        f"- Permutations per comparison: {audit['permutations']}; patient bootstrap replicates: {audit['bootstrap_replicates']}.",
        "- Multiple testing: BH-FDR across the 45 primary fold×seed×pair comparisons; robustness rules are reported separately.",
        "",
        "## Primary result",
        "",
        f"- {significant}/{len(primary)} primary comparisons were BH-FDR significant (q<0.05). See `h1_external_tusz_pairwise.csv` for every comparison and CI/bootstrap files for uncertainty.",
        "- A significant result means that the timing of false-positive events overlaps more than expected under the specified within-recording shift null. It does not identify a common artifact or physiological mechanism.",
        "",
        "## Per-pair descriptive summary (averaged over five folds and three seeds)",
        "",
        "| pair | mean observed F1 | mean null F1 | mean observed-minus-null | significant fold×seed units |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_pair.itertuples(index=False):
        lines.append(f"| {row.left_model}–{row.right_model} | {row.observed_patient_median_f1_mean:.3f} | {row.null_patient_median_f1_mean:.3f} | {row.observed_minus_null_f1_mean:.3f} | {int(row.primary_significant_count)}/{int(row.primary_units)} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The analysis tests temporal sharing of detector false-positive events on TUSZ under frozen CHB thresholds. It is an external descriptive replication of H1, not a new detector-training result and not causal evidence about artifact mechanisms. The cohort and outputs had already been inspected before this analysis, so it should not be labeled a virgin confirmatory test.",
        "",
        "## Audit",
        "",
        f"- Source event rows: {audit['source_event_rows']}; false-positive rows: {audit['source_fp_rows']}; cohort: {audit['source_audit']['cohort_patient_count']} patients / {audit['source_audit']['cohort_record_count']} EDFs.",
        f"- Source event SHA256: `{audit['source_event_sha256']}`.",
        f"- Cohort manifest SHA256: `{audit['cohort_manifest_sha256']}`.",
        f"- Scoring manifest SHA256: `{audit['scoring_manifest_sha256']}`.",
        "- `threshold_source=each frozen CHB validation-only threshold`; `model_training=false`; `threshold_retuning=false`; external labels were not used for selection.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    permutations = 5 if args.smoke else int(args.permutations)
    bootstrap_replicates = 25 if args.smoke else int(args.bootstrap)
    output_dir = REPORT_DIR / "_smoke" if args.smoke else REPORT_DIR
    if not args.smoke and (REPORT_DIR / "manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite existing formal output: {REPORT_DIR}")
    for path in (SOURCE_EVENTS, SOURCE_RUN_METRICS, SOURCE_SCORING_MANIFEST, COHORT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    events = pd.read_parquet(SOURCE_EVENTS)
    scoring = json.loads(SOURCE_SCORING_MANIFEST.read_text())
    cohort = json.loads(COHORT_MANIFEST.read_text())
    source_audit = validate_sources(events, cohort, scoring)
    durations = {(str(row["patient_id"]), str(row["relative_edf"])): float(row["duration_s"]) for row in cohort["records"]}
    patients = sorted({key[0] for key in durations})
    eligible_keys = sorted(key for key, duration in durations.items() if duration >= MIN_BLOCK_S)
    patient_records = {patient: [key for key in eligible_keys if key[0] == patient] for patient in patients}
    events["model"] = events.model.astype(str).str.lower()
    events["fold"] = events.fold.astype(int)
    events["seed"] = events.seed.astype(int)
    events["patient_id"] = events.patient_id.astype(str)
    events["recording"] = events.recording.astype(str)
    rng = np.random.default_rng(RNG_SEED)
    pair_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    units = 0
    for fold in FOLDS:
        for seed in SEEDS:
            for left_model, right_model in PAIRS:
                left = event_index(events, left_model, fold, seed)
                right = event_index(events, right_model, fold, seed)
                # Restrict both observed and null calculations to recordings with
                # enough duration for the predeclared 300-s circular-shift margin.
                left = {key: value for key, value in left.items() if key in patient_records.get(key[0], [])}
                right = {key: value for key, value in right.items() if key in patient_records.get(key[0], [])}
                observed_by_rule = {}
                patient_vectors = {}
                for rule in RULES:
                    observed_by_rule[rule] = patient_metric(left, right, patient_records, patients, rule)
                    patient_vectors[rule] = observed_by_rule[rule][-1]
                null_values = {rule: [] for rule in RULES}
                candidate_keys = sorted(set(right) & set(eligible_keys))
                for permutation in range(permutations):
                    offsets = {key: float(rng.uniform(MIN_NULL_SHIFT_S, durations[key] - MIN_NULL_SHIFT_S)) for key in candidate_keys}
                    shifted_right = shift_index(right, offsets, durations)
                    for rule in RULES:
                        value = patient_metric(left, shifted_right, patient_records, patients, rule)[0]
                        null_values[rule].append(value)
                        null_rows.append({"fold": fold, "seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "permutation": permutation, "null_patient_median_f1": value})
                for rule in RULES:
                    median, mean, matched, n_left, n_right, vector = observed_by_rule[rule]
                    null = np.asarray(null_values[rule], dtype=float)
                    p_raw = (1.0 + float(np.sum(null >= median))) / (len(null) + 1.0)
                    pair_rows.append({
                        "fold": fold,
                        "seed": seed,
                        "left_model": left_model,
                        "right_model": right_model,
                        "rule": rule,
                        "patients": len(patients),
                        "patients_with_eligible_records": int(sum(bool(patient_records[p]) for p in patients)),
                        "eligible_record_count": len(eligible_keys),
                        "left_fp_events": n_left,
                        "right_fp_events": n_right,
                        "matched_fp_events": matched,
                        "observed_patient_median_f1": median,
                        "observed_patient_mean_f1": mean,
                        "null_patient_median_f1": float(np.median(null)),
                        "null_f1_q025": float(np.quantile(null, 0.025)),
                        "null_f1_q975": float(np.quantile(null, 0.975)),
                        "observed_minus_null_f1": median - float(np.median(null)),
                        "p_raw": p_raw,
                        "permutations": permutations,
                        "min_null_shift_s": MIN_NULL_SHIFT_S,
                        "min_block_s": MIN_BLOCK_S,
                    })
                    for replicate in range(bootstrap_replicates):
                        indices = rng.integers(0, len(vector), size=len(vector))
                        sample = vector[indices]
                        bootstrap_rows.append({"fold": fold, "seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "replicate": replicate, "patient_median_f1": float(np.median(sample)), "patient_mean_f1": float(np.mean(sample))})
                units += 1
                print(f"H1_EXTERNAL_TUSZ_DONE fold={fold} seed={seed} pair={left_model}-{right_model} unit={units}/45", flush=True)
    pairwise = pd.DataFrame(pair_rows)
    primary_mask = pairwise.rule == PRIMARY_RULE
    primary_q = bh_fdr(pairwise.loc[primary_mask, "p_raw"].tolist())
    pairwise.loc[primary_mask, "q_fdr_primary_45"] = primary_q
    pairwise["q_fdr_all_rules_225"] = np.nan
    all_q = bh_fdr(pairwise["p_raw"].tolist())
    pairwise["q_fdr_all_rules_225"] = all_q
    pairwise["q_fdr_within_rule"] = pairwise.groupby("rule", sort=False)["p_raw"].transform(lambda values: bh_fdr(values.tolist()))
    pairwise["q_fdr_primary_45"] = pairwise["q_fdr_primary_45"].fillna(np.nan)
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary = pairwise[pairwise.rule == PRIMARY_RULE].copy()
    summary_pair_seed = primary.groupby(["left_model", "right_model", "seed"], as_index=False).agg(
        observed_patient_median_f1_mean=("observed_patient_median_f1", "mean"),
        null_patient_median_f1_mean=("null_patient_median_f1", "mean"),
        observed_minus_null_f1_mean=("observed_minus_null_f1", "mean"),
        primary_significant_count=("q_fdr_primary_45", lambda values: int((values < 0.05).sum())),
        primary_units=("q_fdr_primary_45", "size"),
    )
    summary_pair = primary.groupby(["left_model", "right_model"], as_index=False).agg(
        observed_patient_median_f1_mean=("observed_patient_median_f1", "mean"),
        null_patient_median_f1_mean=("null_patient_median_f1", "mean"),
        observed_minus_null_f1_mean=("observed_minus_null_f1", "mean"),
        primary_significant_count=("q_fdr_primary_45", lambda values: int((values < 0.05).sum())),
        primary_units=("q_fdr_primary_45", "size"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_dir / "h1_external_tusz_pairwise.csv", index=False)
    pd.DataFrame(null_rows).to_parquet(output_dir / "h1_external_tusz_circular_shift_nulls.parquet", index=False)
    bootstrap.to_csv(output_dir / "h1_external_tusz_patient_bootstrap.csv", index=False)
    summary_pair_seed.to_csv(output_dir / "h1_external_tusz_summary_by_pair_seed.csv", index=False)
    summary_pair.to_csv(output_dir / "h1_external_tusz_summary_by_pair.csv", index=False)
    make_plot(pairwise, output_dir / "h1_external_tusz_observed_vs_null.png")
    audit = {
        "release_id": OUT_ID,
        "status": "smoke" if args.smoke else "complete_external_descriptive_replication",
        "source_run": SOURCE_RUN,
        "source_event_rows": int(len(events)),
        "source_fp_rows": int((~events.is_true_positive).sum()),
        "source_event_sha256": sha256(SOURCE_EVENTS),
        "cohort_manifest_sha256": sha256(COHORT_MANIFEST),
        "scoring_manifest_sha256": sha256(SOURCE_SCORING_MANIFEST),
        "source_audit": source_audit,
        "models": list(MODELS),
        "pairs": [list(pair) for pair in PAIRS],
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "analysis_units": 45,
        "analysis_unit_definition": "model_pair x external fold x seed; folds preserved because all five checkpoints evaluated every external patient",
        "primary_rule": PRIMARY_RULE,
        "robustness_rules": list(RULES),
        "permutations": permutations,
        "bootstrap_replicates": bootstrap_replicates,
        "rng_seed": RNG_SEED,
        "min_null_shift_s": MIN_NULL_SHIFT_S,
        "min_block_s": MIN_BLOCK_S,
        "threshold_source": "each frozen CHB validation-only threshold; no external retuning",
        "model_training": False,
        "external_labels_used_for_selection": False,
        "outputs": {},
        "created_at_unix": time.time(),
    }
    if not args.smoke:
        audit["outputs"] = {
            "pairwise_sha256": sha256(output_dir / "h1_external_tusz_pairwise.csv"),
            "null_sha256": sha256(output_dir / "h1_external_tusz_circular_shift_nulls.parquet"),
            "bootstrap_sha256": sha256(output_dir / "h1_external_tusz_patient_bootstrap.csv"),
            "summary_pair_sha256": sha256(output_dir / "h1_external_tusz_summary_by_pair.csv"),
            "summary_pair_seed_sha256": sha256(output_dir / "h1_external_tusz_summary_by_pair_seed.csv"),
            "figure_sha256": sha256(output_dir / "h1_external_tusz_observed_vs_null.png"),
        }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "report.md", audit, pairwise, summary_pair_seed, summary_pair)
    if not args.smoke:
        (output_dir / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
