"""Recompute matched pre/post event-processing metrics from frozen OOF rows.

This is a read-only downstream analysis of the frozen v3 GroupKFold OOF
probabilities.  It does not fit a model or select a threshold.  For each
fold-seed/model run it evaluates the same validation-locked threshold under:

* ``raw_frame_runs``: thresholded frame outputs, with no EMA, duration,
  merging, or refractory filtering (adjacent positive frames are represented
  as separate raw prediction intervals);
* ``raw_contiguous_runs``: thresholded frame outputs grouped into contiguous
  runs, with no smoothing, minimum-duration, merge-gap, or refractory rules;
* ``processed_events``: the frozen eventizer used by the primary release.

The primary before/after table compares ``raw_contiguous_runs`` and
``processed_events`` because both are event-level and use the same matcher.
The frame-level confusion counts are also written separately so they remain
clearly distinct from event metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


SOURCE_ROOT = Path(r"\\wsl$\Ubuntu-24.04\root\b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
SOURCE = SOURCE_ROOT / "outputs" / "reports" / RUN_ID
OOF_PATH = SOURCE / "oof_probabilities_seed.parquet"
SEIZURES_PATH = SOURCE_ROOT / "manifests" / "seizures.parquet"
RECORDINGS_PATH = SOURCE_ROOT / "manifests" / "recordings.parquet"
OUT = Path(__file__).resolve().parent.parent / "reports" / "pre-post-eventization-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)
FOLDS = tuple(range(5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def eventize(
    times: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    ema_alpha: float,
    min_duration_s: float,
    merge_gap_s: float,
    refractory_s: float,
) -> list[tuple[float, float, float]]:
    """Match bfa.eventize exactly, with explicit parameters."""
    times = np.asarray(times, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(times) < 2:
        return []
    step_s = float(np.median(np.diff(times)))
    smoothed = causal_ema(probabilities, ema_alpha)
    above = smoothed >= float(threshold)
    candidates: list[tuple[float, float, float]] = []
    index = 0
    while index < len(above):
        if not above[index]:
            index += 1
            continue
        end_index = index
        while end_index + 1 < len(above) and above[end_index + 1]:
            end_index += 1
        start_s = float(times[index])
        end_s = float(times[end_index] + step_s)
        if end_s - start_s >= float(min_duration_s):
            candidates.append((start_s, end_s, float(np.max(smoothed[index : end_index + 1]))))
        index = end_index + 1
    merged: list[tuple[float, float, float]] = []
    for candidate in candidates:
        if merged and candidate[0] - merged[-1][1] <= float(merge_gap_s):
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], candidate[1]), max(prev[2], candidate[2]))
        else:
            merged.append(candidate)
    accepted: list[tuple[float, float, float]] = []
    for candidate in merged:
        if accepted and candidate[0] < accepted[-1][1] + float(refractory_s):
            continue
        accepted.append(candidate)
    return accepted


def match(predictions: list[tuple[float, float, float]], truths: list[tuple[float, float]]) -> tuple[int, int, int, list[float]]:
    if not predictions or not truths:
        return 0, len(predictions), len(truths), []
    overlaps = np.zeros((len(predictions), len(truths)), dtype=float)
    for i, prediction in enumerate(predictions):
        for j, truth in enumerate(truths):
            overlaps[i, j] = max(0.0, min(prediction[1], truth[1]) - max(prediction[0], truth[0]))
    rows, cols = linear_sum_assignment(-overlaps)
    pairs = [(int(i), int(j), float(overlaps[i, j])) for i, j in zip(rows, cols, strict=True) if overlaps[i, j] >= 1.0]
    matched_predictions = {i for i, _, _ in pairs}
    matched_truths = {j for _, j, _ in pairs}
    delays = [max(0.0, predictions[i][0] - truths[j][0]) for i, j, _ in pairs]
    return len(pairs), len(predictions) - len(matched_predictions), len(truths) - len(matched_truths), delays


def union_length(intervals: list[tuple[float, float]], clip_start: float, clip_end: float) -> float:
    clipped = [(max(clip_start, float(a)), min(clip_end, float(b))) for a, b in intervals if float(b) > clip_start and float(a) < clip_end]
    if not clipped:
        return 0.0
    clipped.sort()
    total = 0.0
    start, end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += max(0.0, end - start)
            start, end = next_start, next_end
    return total + max(0.0, end - start)


def truths_for(seizures: pd.DataFrame, recording: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str) == str(recording)]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def score_run(
    table: pd.DataFrame,
    *,
    threshold: float,
    seizures: pd.DataFrame,
    durations: dict[str, float],
    condition: str,
) -> dict[str, object]:
    tp = fp = truth_count = 0
    nonseizure_s = 0.0
    monitoring_s = 0.0
    alarm_false_s = 0.0
    delays: list[float] = []
    prediction_count = 0
    recording_count = 0
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if len(ordered) < 2:
            continue
        recording_count += 1
        recording_id = str(recording)
        evaluation_start = float(ordered.end.iloc[0])
        truth = truths_for(seizures, recording_id, evaluation_start)
        if condition == "raw_frame_runs":
            times = ordered.end.to_numpy(dtype=float)
            probs = ordered.probability.to_numpy(dtype=float)
            step = float(np.median(np.diff(times)))
            predictions = [
                (float(row.start), float(row.end), float(row.probability))
                for row in ordered.itertuples(index=False)
                if float(row.probability) >= threshold
            ]
            # The frame is the observed [start,end] window.  This condition
            # is descriptive and is not used in the primary event comparison.
            del times, probs, step
        elif condition == "raw_contiguous_runs":
            predictions = eventize(
                ordered.end.to_numpy(dtype=float),
                ordered.probability.to_numpy(dtype=float),
                threshold=threshold,
                ema_alpha=1.0,
                min_duration_s=0.0,
                merge_gap_s=0.0,
                refractory_s=0.0,
            )
        elif condition == "processed_events":
            predictions = eventize(
                ordered.end.to_numpy(dtype=float),
                ordered.probability.to_numpy(dtype=float),
                threshold=threshold,
                ema_alpha=1.0 / 3.0,
                min_duration_s=4.0,
                merge_gap_s=10.0,
                refractory_s=30.0,
            )
        else:
            raise ValueError(condition)
        prediction_count += len(predictions)
        matched_tp, matched_fp, _, run_delays = match(predictions, truth)
        tp += matched_tp
        fp += matched_fp
        truth_count += len(truth)
        delays.extend(run_delays)
        seizure_seconds = sum(max(0.0, end - start) for start, end in truth)
        duration = float(durations[recording_id])
        usable_nonseizure = max(0.0, duration - evaluation_start - seizure_seconds)
        nonseizure_s += usable_nonseizure
        monitoring_s += max(0.0, duration - evaluation_start)
        alarm_union = union_length([(p[0], p[1]) for p in predictions], evaluation_start, duration)
        seizure_union = union_length(truth, evaluation_start, duration)
        alarm_false_s += max(0.0, alarm_union - seizure_union)
    return {
        "condition": condition,
        "recordings": recording_count,
        "prediction_events_or_frames": prediction_count,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(max(0, truth_count - tp)),
        "truth": int(truth_count),
        "sensitivity": float(tp / truth_count) if truth_count else np.nan,
        "nonseizure_s": float(nonseizure_s),
        "fa_per_24h": float(fp * 86400.0 / nonseizure_s) if nonseizure_s else np.nan,
        "monitoring_s": float(monitoring_s),
        "false_alarm_seconds": float(alarm_false_s),
        "false_alarm_minutes_per_24h": float(alarm_false_s * 1440.0 / monitoring_s) if monitoring_s else np.nan,
        "false_alarm_minutes_per_24h_nonseizure": float(alarm_false_s * 1440.0 / nonseizure_s) if nonseizure_s else np.nan,
        "mean_delay_s": float(np.mean(delays)) if delays else np.nan,
        "matched_events": int(len(delays)),
    }


def processed_window_labels(ordered: pd.DataFrame, threshold: float) -> np.ndarray:
    """Project accepted eventized alarms back onto the original window ends.

    The frozen eventizer represents an alarm interval as ``[window_end,
    window_end + step)``.  A window is therefore positive after processing if
    its end marker lies inside one of the accepted event intervals.  This
    preserves the original window label unit while applying the fixed EMA,
    duration, merge, and refractory rules.
    """
    times = ordered.end.to_numpy(dtype=float)
    probabilities = ordered.probability.to_numpy(dtype=float)
    events = eventize(
        times,
        probabilities,
        threshold=threshold,
        ema_alpha=1.0 / 3.0,
        min_duration_s=4.0,
        merge_gap_s=10.0,
        refractory_s=30.0,
    )
    labels = np.zeros(len(ordered), dtype=bool)
    for start_s, end_s, _ in events:
        labels |= (times >= start_s) & (times < end_s)
    return labels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [OOF_PATH, SEIZURES_PATH, RECORDINGS_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    columns = ["patient", "recording", "start", "end", "label", "warmup", "probability", "model", "fold", "seed", "validation_locked_threshold"]
    oof = pd.read_parquet(OOF_PATH, columns=columns)
    oof = oof[~oof.warmup.astype(bool)].copy()
    oof["model"] = oof.model.astype(str)
    seizures = pd.read_parquet(SEIZURES_PATH)
    recordings = pd.read_parquet(RECORDINGS_PATH)
    durations = recordings.set_index(recordings.recording_id.astype(str)).duration_s.astype(float).to_dict()

    # Window-level confusion counts reproduce the supplied pre-eventization
    # figure exactly: labels are evaluated against each locked threshold.
    window_rows: list[dict[str, object]] = []
    for (model, fold, seed), group in oof.groupby(["model", "fold", "seed"], sort=True):
        threshold_values = group.validation_locked_threshold.dropna().unique()
        if len(threshold_values) != 1:
            raise RuntimeError(f"non-unique threshold for {model=} {fold=} {seed=}: {threshold_values}")
        threshold = float(threshold_values[0])
        labeled = group[group.label.notna()].copy()
        actual = labeled.label.astype(int).to_numpy()
        pred = (labeled.probability.to_numpy(dtype=float) >= threshold).astype(int)
        window_rows.append({
            "model": model, "fold": int(fold), "seed": int(seed), "threshold": threshold,
            "tp": int(((actual == 1) & (pred == 1)).sum()),
            "fn": int(((actual == 1) & (pred == 0)).sum()),
            "fp": int(((actual == 0) & (pred == 1)).sum()),
            "tn": int(((actual == 0) & (pred == 0)).sum()),
            "seizure_windows": int((actual == 1).sum()),
            "background_windows": int((actual == 0).sum()),
        })
    window_frame = pd.DataFrame(window_rows)
    window_frame.to_csv(OUT / "window_confusion_per_run.csv", index=False)
    window_summary = (
        window_frame.groupby("model", as_index=False)[["tp", "fn", "fp", "tn", "seizure_windows", "background_windows"]]
        .sum()
    )
    window_summary["sensitivity"] = window_summary.tp / window_summary.seizure_windows
    window_summary["specificity"] = window_summary.tn / window_summary.background_windows
    window_summary["balanced_accuracy"] = (window_summary.sensitivity + window_summary.specificity) / 2.0
    window_summary.to_csv(OUT / "window_confusion_summary.csv", index=False)

    # A second confusion matrix at the same window unit.  This is obtained by
    # projecting the accepted post-processed event intervals back to window
    # ends, so it is directly comparable with the pre-eventization matrix.
    window_after_rows: list[dict[str, object]] = []
    for (model, fold, seed), group in oof.groupby(["model", "fold", "seed"], sort=True):
        threshold_values = group.validation_locked_threshold.dropna().unique()
        if len(threshold_values) != 1:
            raise RuntimeError(f"non-unique threshold for {model=} {fold=} {seed=}: {threshold_values}")
        threshold = float(threshold_values[0])
        ordered = group.sort_values(["recording", "end"], kind="stable")
        post_series = pd.Series(False, index=group.index, dtype=bool)
        for _, rec in ordered.groupby("recording", sort=False):
            labels = processed_window_labels(rec, threshold)
            post_series.loc[rec.index] = labels
        labeled = group[group.label.notna()].copy()
        actual = labeled.label.astype(int).to_numpy()
        post = post_series.loc[labeled.index].to_numpy(dtype=int)
        window_after_rows.append({
            "model": model, "fold": int(fold), "seed": int(seed), "threshold": threshold,
            "tp": int(((actual == 1) & (post == 1)).sum()),
            "fn": int(((actual == 1) & (post == 0)).sum()),
            "fp": int(((actual == 0) & (post == 1)).sum()),
            "tn": int(((actual == 0) & (post == 0)).sum()),
            "seizure_windows": int((actual == 1).sum()),
            "background_windows": int((actual == 0).sum()),
        })
    window_after = pd.DataFrame(window_after_rows)
    window_after.to_csv(OUT / "window_confusion_after_eventization_per_run.csv", index=False)
    window_after_summary = window_after.groupby("model", as_index=False)[
        ["tp", "fn", "fp", "tn", "seizure_windows", "background_windows"]
    ].sum()
    window_after_summary["sensitivity"] = window_after_summary.tp / window_after_summary.seizure_windows
    window_after_summary["specificity"] = window_after_summary.tn / window_after_summary.background_windows
    window_after_summary["balanced_accuracy"] = (
        window_after_summary.sensitivity + window_after_summary.specificity
    ) / 2.0
    window_after_summary.to_csv(OUT / "window_confusion_after_eventization_summary.csv", index=False)
    before_after_window = pd.concat(
        [window_summary.assign(stage="before_event_postprocessing"), window_after_summary.assign(stage="after_event_postprocessing")],
        ignore_index=True,
    )
    before_after_window.to_csv(OUT / "window_confusion_before_after_summary.csv", index=False)

    rows: list[dict[str, object]] = []
    for (model, fold, seed), group in oof.groupby(["model", "fold", "seed"], sort=True):
        threshold_values = group.validation_locked_threshold.dropna().unique()
        if len(threshold_values) != 1:
            raise RuntimeError(f"non-unique threshold for {model=} {fold=} {seed=}: {threshold_values}")
        threshold = float(threshold_values[0])
        for condition in ("raw_frame_runs", "raw_contiguous_runs", "processed_events"):
            score = score_run(group, threshold=threshold, seizures=seizures, durations=durations, condition=condition)
            rows.append({"model": model, "fold": int(fold), "seed": int(seed), "threshold": threshold, **score})
        print(f"DONE model={model} fold={fold} seed={seed}", flush=True)
    run_metrics = pd.DataFrame(rows)
    run_metrics.to_csv(OUT / "raw_vs_processed_per_run.csv", index=False)

    # Means across the same 15 fold-seed evaluations used by the event plot.
    summary = (
        run_metrics.groupby(["model", "condition"], as_index=False)
        .agg(
            evaluations=("sensitivity", "size"),
            mean_sensitivity=("sensitivity", "mean"),
            mean_fa_per_24h=("fa_per_24h", "mean"),
            mean_false_alarm_minutes_per_24h=("false_alarm_minutes_per_24h", "mean"),
            mean_delay_s=("mean_delay_s", "mean"),
            pooled_tp=("tp", "sum"), pooled_fp=("fp", "sum"), pooled_fn=("fn", "sum"), pooled_truth=("truth", "sum"),
            pooled_nonseizure_s=("nonseizure_s", "sum"),
            pooled_monitoring_s=("monitoring_s", "sum"),
            pooled_false_alarm_seconds=("false_alarm_seconds", "sum"),
        )
    )
    summary["pooled_sensitivity"] = summary.pooled_tp / summary.pooled_truth
    summary["pooled_fa_per_24h"] = summary.pooled_fp * 86400.0 / summary.pooled_nonseizure_s
    # Alarm-time burden follows the continuous-alarm audit: normalize by all
    # evaluable monitoring time, including seizure time.
    summary["pooled_false_alarm_minutes_per_24h"] = (
        summary["pooled_false_alarm_seconds"] * 1440.0 / summary["pooled_monitoring_s"]
    )
    summary.to_csv(OUT / "raw_vs_processed_summary.csv", index=False)
    primary = summary[summary.condition.isin(["raw_contiguous_runs", "processed_events"])].copy()
    primary.to_csv(OUT / "before_after_event_metrics_table.csv", index=False)

    processed = run_metrics[run_metrics.condition == "processed_events"].copy()
    stored = pd.read_csv(SOURCE / "oof_run_metrics.csv")
    parity = processed.merge(stored, on=["model", "fold", "seed"], suffixes=("_recomputed", "_stored"))
    # The frozen OOF table calls sensitivity ``event_sensitivity``; retain a
    # direct, explicit mapping so parity checks cannot silently compare the
    # wrong columns.
    metric_pairs = {
        "tp": ("tp_recomputed", "tp_stored"),
        "fp": ("fp_recomputed", "fp_stored"),
        "truth": ("truth_recomputed", "truth_stored"),
        "sensitivity": ("sensitivity", "event_sensitivity"),
        "fa_per_24h": ("fa_per_24h_recomputed", "fa_per_24h_stored"),
    }
    for metric, (left, right) in metric_pairs.items():
        parity[f"abs_error_{metric}"] = (parity[left] - parity[right]).abs()
    parity.to_csv(OUT / "processed_parity_with_frozen_oof.csv", index=False)

    manifest = {
        "analysis_id": "pre-post-eventization-v1",
        "status": "complete",
        "source_protocol": RUN_ID,
        "source_oof_sha256": sha256(OOF_PATH),
        "source_oof_rows": int(len(oof)),
        "evaluations": 15,
        "models": list(MODELS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "threshold_source": "validation_locked_threshold column in frozen OOF; no threshold refit",
        "processed_eventizer": {"ema_alpha": 1 / 3, "min_duration_s": 4, "merge_gap_s": 10, "refractory_s": 30},
        "raw_contiguous_definition": "thresholded frames grouped only into contiguous runs; EMA=1, min duration=0, merge gap=0, refractory=0",
        "raw_frame_definition": "each threshold-positive OOF window evaluated as its observed [start,end] interval; descriptive only",
        "matcher": "Hungarian maximum overlap with minimum_overlap_s=1.0",
        "evaluation_start": "first ordered OOF end per recording, matching frozen OOF scorer",
        "test_used_for_selection": False,
        "processed_parity_max_abs_error": {
            metric: float(parity[f"abs_error_{metric}"].max()) for metric in ("tp", "fp", "truth", "sensitivity", "fa_per_24h")
        },
        "outputs": sorted(path.name for path in OUT.iterdir() if path.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nSUMMARY\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
