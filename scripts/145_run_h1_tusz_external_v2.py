"""Corrected H1 external descriptive replication for the short-record TUSZ cohort.

The first diagnostic version used the CHB >=720 s null eligibility rule and
left most TUSZ recordings out.  This version is a new, non-overwriting
namespace.  It keeps all recordings and uses a predeclared feasible circular
shift: the offset is sampled from [min(300 s, 0.45*duration), duration-
min(300 s, 0.45*duration)].  Patient-level F1 is computed over patients with
at least one FP event in the compared pair; that informative set is invariant
under within-recording shifts.
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
OUT_ID = "h1-external-tusz-v2"
REPORT_DIR = ROOT / "outputs" / "reports" / OUT_ID
EVENT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_events.parquet"
COHORT_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_manifest.json"
SCORING_PATH = ROOT / "outputs" / "reports" / SOURCE_RUN / "tusz_groupkfold_external_v1_scoring_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)
RULES = ("onset_pm_5s", "onset_pm_15s", "onset_pm_30s", "onset_pm_60s", "interval_overlap_strict")
PRIMARY_RULE = "onset_pm_30s"
MAX_SHIFT_S = 300.0
EXCLUSION_FRACTION = 0.45
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
    return onset_matches(left, right, float(rule.removeprefix("onset_pm_").removesuffix("s")))


def event_index(events: pd.DataFrame, model: str, fold: int, seed: int) -> dict[tuple[str, str], list[Event]]:
    subset = events[(events.model == model) & (events.fold == fold) & (events.seed == seed) & (~events.is_true_positive)]
    result: dict[tuple[str, str], list[Event]] = {}
    for row in subset.itertuples(index=False):
        key = (str(row.patient_id), str(row.recording))
        result.setdefault(key, []).append(Event(row.start_s, row.end_s, row.peak_probability))
    for values in result.values():
        values.sort(key=lambda event: event.start_s)
    return result


def metric(
    left: dict[tuple[str, str], list[Event]],
    right: dict[tuple[str, str], list[Event]],
    patient_records: dict[str, list[tuple[str, str]]],
    patients: list[str],
    rule: str,
) -> tuple[float, float, int, int, int, np.ndarray]:
    values: list[float] = []
    matched_total = left_total = right_total = 0
    for patient in patients:
        n_left = n_right = matched = 0
        for key in patient_records[patient]:
            first = left.get(key, [])
            second = right.get(key, [])
            n_left += len(first)
            n_right += len(second)
            matched += match_count(first, second, rule)
        left_total += n_left
        right_total += n_right
        matched_total += matched
        values.append(2.0 * matched / (n_left + n_right) if n_left + n_right else 0.0)
    vector = np.asarray(values, dtype=float)
    return float(np.median(vector)), float(np.mean(vector)), matched_total, left_total, right_total, vector


def adaptive_min_shift(duration_s: float) -> float:
    return min(MAX_SHIFT_S, EXCLUSION_FRACTION * duration_s)


def shift_index(index: dict[tuple[str, str], list[Event]], durations: dict[tuple[str, str], float], rng: np.random.Generator) -> dict[tuple[str, str], list[Event]]:
    shifted: dict[tuple[str, str], list[Event]] = {}
    for key, values in index.items():
        duration = durations[key]
        minimum = adaptive_min_shift(duration)
        low, high = minimum, duration - minimum
        offset = duration / 2.0 if high <= low else float(rng.uniform(low, high))
        output: list[Event] = []
        for event in values:
            start = (event.start_s + offset) % duration
            end = min(duration, start + min(event.end_s - event.start_s, duration))
            output.append(Event(start, end, event.peak_probability))
        shifted[key] = output
    return shifted


def validate_sources(events: pd.DataFrame, cohort: dict, scoring: dict) -> dict[str, object]:
    required = {"model", "fold", "seed", "patient_id", "recording", "event_index", "start_s", "end_s", "is_true_positive"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise RuntimeError(f"events missing columns: {missing}")
    if int(scoring.get("inference_runs_scored", -1)) != 45:
        raise RuntimeError("source inference is not 45/45")
    if bool(scoring.get("model_training", True)) or bool(scoring.get("threshold_retuning", True)):
        raise RuntimeError("source indicates training or threshold retuning")
    if "validation-only" not in str(scoring.get("threshold_source", "")):
        raise RuntimeError("source threshold is not validation-only")
    records = cohort["records"]
    durations = {(str(row["patient_id"]), str(row["relative_edf"])): float(row["duration_s"]) for row in records}
    event_keys = set(zip(events.patient_id.astype(str), events.recording.astype(str)))
    unknown = event_keys - set(durations)
    if unknown:
        raise RuntimeError(f"{len(unknown)} event recording keys are absent from cohort")
    duplicate_rows = int(events.duplicated(["model", "fold", "seed", "patient_id", "recording", "event_index"]).sum())
    if duplicate_rows:
        raise RuntimeError(f"{duplicate_rows} duplicate event keys")
    return {
        "event_rows": int(len(events)),
        "fp_rows": int((~events.is_true_positive).sum()),
        "event_patients": int(events.patient_id.astype(str).nunique()),
        "cohort_patients": int(len({str(row["patient_id"]) for row in records})),
        "cohort_records": int(len(records)),
        "duration_min_s": float(min(durations.values())),
        "duration_median_s": float(np.median(list(durations.values()))),
        "duration_max_s": float(max(durations.values())),
        "duplicate_event_keys": duplicate_rows,
        "unknown_record_keys": 0,
        "all_recordings_used": True,
    }


def plot_primary(pairwise: pd.DataFrame, output: Path) -> None:
    frame = pairwise[pairwise.rule == PRIMARY_RULE]
    colors = {"singlem|cbramod": "#2455A4", "singlem|tcn_gat": "#008C95", "cbramod|tcn_gat": "#B56B1F"}
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    for row in frame.itertuples(index=False):
        key = f"{row.left_model}|{row.right_model}"
        ax.scatter(row.null_patient_median_f1, row.observed_patient_median_f1, color=colors[key], alpha=0.95 if row.q_fdr_primary_45 < 0.05 else 0.45, s=55, edgecolor="white", linewidth=0.5)
    lo = min(frame.null_patient_median_f1.min(), frame.observed_patient_median_f1.min())
    hi = max(frame.null_patient_median_f1.max(), frame.observed_patient_median_f1.max())
    pad = 0.03 * max(hi - lo, 0.1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#777777", linewidth=1.2, label="Observed = null")
    ax.set_xlabel("Adaptive circular-shift null patient-median F1")
    ax.set_ylabel("Observed patient-median F1 (±30 s)")
    ax.set_title("TUSZ H1 external descriptive replication (all recordings)")
    ax.grid(True, color="#E8EDF3", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, audit: dict[str, object], pairwise: pd.DataFrame, summary: pd.DataFrame) -> None:
    primary = pairwise[pairwise.rule == PRIMARY_RULE]
    sig = int((primary.q_fdr_primary_45 < 0.05).sum())
    lines = [
        "# H1 External TUSZ Replication v2",
        "",
        "Status: complete external descriptive replication; no model training or threshold retuning.",
        "",
        "## Why v2 was required",
        "",
        "TUSZ contains many short EDFs. The diagnostic long-record-only analysis left most records out and produced zero-inflated patient medians, so it is retained only as an input-adaptation failure audit. This v2 analysis predeclares an adaptive feasible circular-shift null and uses all 880 recordings.",
        "",
        "## Design",
        "",
        f"- Unit: 3 model pairs × 5 external folds × 3 seeds = {len(primary)} comparisons; folds remain separate because every external patient was evaluated by every frozen checkpoint.",
        "- Primary metric: patient-median one-to-one false-positive onset-match F1 within ±30 s among informative patients (at least one FP in either model; this set is unchanged by within-recording shifts).",
        f"- Null: within-recording circular shift. For duration D, offset is sampled from [min(300, 0.45D), D−min(300, 0.45D)].",
        f"- Permutations per comparison: {audit['permutations']}; patient bootstrap replicates: {audit['bootstrap_replicates']}.",
        "- BH-FDR is applied across the 45 primary fold×seed×pair tests; the four robustness rules are reported separately.",
        "",
        "## Primary result",
        "",
        f"- {sig}/{len(primary)} primary comparisons were BH-FDR significant (q<0.05).",
        "- Significance means that FP timing overlaps more than the adaptive within-recording shift null. It does not identify a shared artifact or prove a physiological mechanism.",
        "",
        "## Pair summary",
        "",
        "| pair | observed F1 | null F1 | observed−null | significant units |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.left_model}–{row.right_model} | {row.observed_f1_mean:.3f} | {row.null_f1_mean:.3f} | {row.delta_mean:.3f} | {int(row.sig_units)}/{int(row.units)} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is an external descriptive replication of H1 under frozen CHB validation-only thresholds. It is not a virgin confirmatory test because the TUSZ inference outputs were inspected before this analysis. It tests temporal sharing only; it cannot establish a common artifact mechanism or causality.",
        "",
        "## Audit",
        "",
        f"- Source events: {audit['source_event_rows']} rows ({audit['source_fp_rows']} FP rows); cohort: {audit['source_audit']['cohort_patients']} patients / {audit['source_audit']['cohort_records']} recordings.",
        f"- Source event SHA256: `{audit['source_event_sha256']}`.",
        f"- Cohort SHA256: `{audit['cohort_manifest_sha256']}`; scoring manifest SHA256: `{audit['scoring_manifest_sha256']}`.",
        "- `threshold_source=validation-only`, `model_training=false`, `threshold_retuning=false`, `external_labels_used_for_selection=false`.",
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
    out = REPORT_DIR / "_smoke" if args.smoke else REPORT_DIR
    if not args.smoke and (REPORT_DIR / "manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite {REPORT_DIR}")
    for path in (EVENT_PATH, COHORT_PATH, SCORING_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    events = pd.read_parquet(EVENT_PATH)
    cohort = json.loads(COHORT_PATH.read_text())
    scoring = json.loads(SCORING_PATH.read_text())
    source_audit = validate_sources(events, cohort, scoring)
    durations = {(str(row["patient_id"]), str(row["relative_edf"])): float(row["duration_s"]) for row in cohort["records"]}
    patients = sorted({key[0] for key in durations})
    patient_records = {patient: [key for key in durations if key[0] == patient] for patient in patients}
    events["model"] = events.model.astype(str).str.lower()
    events["fold"] = events.fold.astype(int)
    events["seed"] = events.seed.astype(int)
    events["patient_id"] = events.patient_id.astype(str)
    events["recording"] = events.recording.astype(str)
    rng = np.random.default_rng(RNG_SEED)
    pair_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    unit = 0
    for fold in FOLDS:
        for seed in SEEDS:
            for left_model, right_model in PAIRS:
                left = event_index(events, left_model, fold, seed)
                right = event_index(events, right_model, fold, seed)
                informative = [patient for patient in patients if any(left.get(key) or right.get(key) for key in patient_records[patient])]
                if not informative:
                    raise RuntimeError(f"no informative patients for fold={fold} seed={seed} pair={left_model}-{right_model}")
                records_for_informative = {patient: patient_records[patient] for patient in informative}
                observed = {rule: metric(left, right, records_for_informative, informative, rule) for rule in RULES}
                null_values = {rule: [] for rule in RULES}
                for permutation in range(permutations):
                    shifted = shift_index(right, durations, rng)
                    for rule in RULES:
                        value = metric(left, shifted, records_for_informative, informative, rule)[0]
                        null_values[rule].append(value)
                        null_rows.append({"fold": fold, "seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "permutation": permutation, "null_patient_median_f1": value})
                for rule in RULES:
                    median, mean, matched, n_left, n_right, vector = observed[rule]
                    null = np.asarray(null_values[rule], dtype=float)
                    pair_rows.append({
                        "fold": fold, "seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule,
                        "cohort_patients": len(patients), "informative_patients": len(informative), "recordings": len(durations),
                        "left_fp_events": n_left, "right_fp_events": n_right, "matched_fp_events": matched,
                        "observed_patient_median_f1": median, "observed_patient_mean_f1": mean,
                        "null_patient_median_f1": float(np.median(null)), "null_f1_q025": float(np.quantile(null, 0.025)), "null_f1_q975": float(np.quantile(null, 0.975)),
                        "observed_minus_null_f1": median - float(np.median(null)), "p_raw": (1.0 + float(np.sum(null >= median))) / (len(null) + 1.0),
                        "permutations": permutations, "bootstrap_replicates": bootstrap_replicates,
                    })
                    for replicate in range(bootstrap_replicates):
                        indices = rng.integers(0, len(vector), size=len(vector))
                        sample = vector[indices]
                        bootstrap_rows.append({"fold": fold, "seed": seed, "left_model": left_model, "right_model": right_model, "rule": rule, "replicate": replicate, "patient_median_f1": float(np.median(sample)), "patient_mean_f1": float(np.mean(sample))})
                unit += 1
                print(f"H1_EXTERNAL_TUSZ_V2_DONE fold={fold} seed={seed} pair={left_model}-{right_model} unit={unit}/45", flush=True)
    pairwise = pd.DataFrame(pair_rows)
    pairwise["q_fdr_all_225"] = bh_fdr(pairwise.p_raw.tolist())
    primary_mask = pairwise.rule == PRIMARY_RULE
    primary_q = bh_fdr(pairwise.loc[primary_mask, "p_raw"].tolist())
    pairwise["q_fdr_primary_45"] = np.nan
    pairwise.loc[primary_mask, "q_fdr_primary_45"] = primary_q
    pairwise["q_fdr_within_rule"] = pairwise.groupby("rule", sort=False).p_raw.transform(lambda values: bh_fdr(values.tolist()))
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary = pairwise[pairwise.rule == PRIMARY_RULE]
    summary = primary.groupby(["left_model", "right_model"], as_index=False).agg(observed_f1_mean=("observed_patient_median_f1", "mean"), null_f1_mean=("null_patient_median_f1", "mean"), delta_mean=("observed_minus_null_f1", "mean"), sig_units=("q_fdr_primary_45", lambda values: int((values < 0.05).sum())), units=("q_fdr_primary_45", "size"))
    out.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(out / "h1_external_tusz_v2_pairwise.csv", index=False)
    pd.DataFrame(null_rows).to_parquet(out / "h1_external_tusz_v2_circular_shift_nulls.parquet", index=False)
    bootstrap.to_csv(out / "h1_external_tusz_v2_patient_bootstrap.csv", index=False)
    summary.to_csv(out / "h1_external_tusz_v2_summary_by_pair.csv", index=False)
    plot_primary(pairwise, out / "h1_external_tusz_v2_observed_vs_null.png")
    audit = {
        "release_id": OUT_ID, "status": "smoke" if args.smoke else "complete_external_descriptive_replication",
        "source_run": SOURCE_RUN, "source_event_rows": int(len(events)), "source_fp_rows": int((~events.is_true_positive).sum()),
        "source_event_sha256": sha256(EVENT_PATH), "cohort_manifest_sha256": sha256(COHORT_PATH), "scoring_manifest_sha256": sha256(SCORING_PATH),
        "source_audit": source_audit, "models": list(MODELS), "pairs": [list(pair) for pair in PAIRS], "folds": list(FOLDS), "seeds": list(SEEDS),
        "analysis_units": 45, "analysis_unit_definition": "model_pair x external fold x seed; folds preserved because every external patient was evaluated by every frozen checkpoint",
        "primary_rule": PRIMARY_RULE, "robustness_rules": list(RULES), "permutations": permutations, "bootstrap_replicates": bootstrap_replicates, "rng_seed": RNG_SEED,
        "null": {"type": "within_recording_circular_shift", "max_shift_s": MAX_SHIFT_S, "short_record_rule": "minimum exclusion=min(300 s, 0.45*duration); all recordings retained"},
        "threshold_source": "each frozen CHB validation-only threshold; no external retuning", "model_training": False, "threshold_retuning": False, "external_labels_used_for_selection": False,
        "outputs": {}, "created_at_unix": time.time(),
    }
    if not args.smoke:
        audit["outputs"] = {name: sha256(out / filename) for name, filename in {"pairwise_sha256": "h1_external_tusz_v2_pairwise.csv", "null_sha256": "h1_external_tusz_v2_circular_shift_nulls.parquet", "bootstrap_sha256": "h1_external_tusz_v2_patient_bootstrap.csv", "summary_sha256": "h1_external_tusz_v2_summary_by_pair.csv", "figure_sha256": "h1_external_tusz_v2_observed_vs_null.png"}.items()}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out / "report.md", audit, pairwise, summary)
    if not args.smoke:
        (out / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
