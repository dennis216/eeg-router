from __future__ import annotations

"""Independent read-only audit and Oracle recalculation for the BIOT pilot.

This script never retrains an expert or a Router and never changes a frozen
expert artifact.  It reads completed expert probability tables, recomputes
the event metrics, and writes a new audit namespace.  The two Oracle variants
are deliberately reported separately:

* ``window_confidence_oracle`` selects an expert on the 10-s grid using the
  frozen per-expert threshold and normalized threshold-relative confidence.
  Its event score uses the historical mixed-margin eventization definition.
* ``decision_state_oracle`` first computes each expert's own causal EMA
  decision state, then selects among those states and eventizes the selected
  binary state.  This is an event-rule-aligned diagnostic that avoids creating
  a new EMA trajectory by mixing probabilities from different experts.

Both are post-hoc upper bounds (they use the known label to choose an expert)
and are not deployable Routers.  No threshold is reselected here.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
RUNS = ROOT / "runs" / NAMESPACE / "formal"
REPORT = ROOT / "outputs" / "reports" / NAMESPACE
AUDIT_ROOT = REPORT / "oracle_recalculation_audit_v1"
MODELS = ["singlem", "cbramod", "tcn_gat"]
KEYS = ["patient", "recording", "start", "end", "relative_path"]


def load_v3() -> Any:
    path = ROOT / "scripts" / "28_run_v3_unified.py"
    spec = importlib.util.spec_from_file_location("v3_helpers_for_oracle_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3()
from bfa.evaluation.eventize import causal_ema, eventize  # noqa: E402
from bfa.evaluation.match import match_events  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def labels_equal(left: pd.Series, right: pd.Series) -> bool:
    a = left.to_numpy(dtype=float)
    b = right.to_numpy(dtype=float)
    if a.shape != b.shape:
        return False
    both_known = np.isfinite(a) & np.isfinite(b)
    if np.any(a[both_known] != b[both_known]):
        return False
    return bool(np.all(np.isfinite(a) == np.isfinite(b)))


def truth_intervals(
    seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float
) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def score_state_table(
    table: pd.DataFrame,
    state: np.ndarray,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> dict[str, float | int]:
    """Score a precomputed binary state with the frozen event rules."""
    if len(table) != len(state):
        raise ValueError("state/table length mismatch")
    tp = fa = truth_count = 0
    nonseizure_seconds = 0.0
    for recording_id, group in table.assign(_state=state).groupby(
        "recording", sort=False
    ):
        ordered = group.sort_values("end", kind="stable")
        times = ordered.end.to_numpy(dtype=float)
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError(f"invalid time grid for recording {recording_id}")
        predictions = eventize(
            times,
            ordered["_state"].to_numpy(dtype=float),
            threshold=0.5,
        )
        evaluation_start_s = float(times[0])
        duration_s = float(
            recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0]
        )
        truths = truth_intervals(seizures, str(recording_id), evaluation_start_s)
        matched = match_events(predictions, truths)
        tp += len(matched.pairs)
        fa += len(matched.unmatched_predictions)
        truth_count += len(truths)
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        nonseizure_seconds += max(
            0.0, duration_s - evaluation_start_s - seizure_seconds
        )
    hours = nonseizure_seconds / 3600.0
    return {
        "true_positive_events": tp,
        "false_alarm_events": fa,
        "truth_events": truth_count,
        "event_sensitivity": tp / truth_count if truth_count else 1.0,
        "nonseizure_hours": hours,
        "fa_per_24h": fa * 24.0 / hours if hours > 0 else float("nan"),
    }


def score_selected_mixed(
    table: pd.DataFrame,
    selections: np.ndarray,
    thresholds: np.ndarray,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> dict[str, float | int]:
    """Replicate the pilot's selected-probability/mixed-margin scorer."""
    probabilities = table[[f"prob_{model}" for model in MODELS]].to_numpy(float)
    selected = probabilities[np.arange(len(table)), selections]
    margin = selected - thresholds[selections]
    working = table[KEYS + ["label"]].copy()
    working["margin"] = margin
    tp = fa = truth_count = 0
    nonseizure_seconds = 0.0
    for recording_id, group in working.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        times = ordered.end.to_numpy(dtype=float)
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError(f"invalid time grid for recording {recording_id}")
        predictions = eventize(
            times,
            ordered.margin.to_numpy(dtype=float),
            threshold=0.0,
        )
        evaluation_start_s = float(times[0])
        duration_s = float(
            recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0]
        )
        truths = truth_intervals(seizures, str(recording_id), evaluation_start_s)
        matched = match_events(predictions, truths)
        tp += len(matched.pairs)
        fa += len(matched.unmatched_predictions)
        truth_count += len(truths)
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        nonseizure_seconds += max(
            0.0, duration_s - evaluation_start_s - seizure_seconds
        )
    hours = nonseizure_seconds / 3600.0
    return {
        "true_positive_events": tp,
        "false_alarm_events": fa,
        "truth_events": truth_count,
        "event_sensitivity": tp / truth_count if truth_count else 1.0,
        "nonseizure_hours": hours,
        "fa_per_24h": fa * 24.0 / hours if hours > 0 else float("nan"),
    }


def threshold_relative_confidence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    t = np.asarray(thresholds, dtype=float).reshape(-1)
    if p.ndim != 2 or p.shape != (len(y), len(t)):
        raise ValueError("incompatible probability, label, and threshold shapes")
    positive = y[:, None] == 1.0
    signed = np.where(positive, p - t[None, :], t[None, :] - p)
    correct_denominator = np.where(positive, 1.0 - t[None, :], t[None, :])
    wrong_denominator = np.where(positive, t[None, :], 1.0 - t[None, :])
    denominator = np.where(signed >= 0.0, correct_denominator, wrong_denominator)
    denominator = np.maximum(denominator, np.finfo(float).eps)
    confidence = signed / denominator
    confidence[~np.isfinite(y), :] = np.nan
    return confidence


def select_oracle(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    fallback_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = threshold_relative_confidence(probabilities, labels, thresholds)
    known = np.isfinite(labels)
    predicted_positive = probabilities >= thresholds[None, :]
    correct = np.zeros_like(predicted_positive, dtype=bool)
    correct[known] = predicted_positive[known] == (labels[known, None] == 1.0)
    selected = np.argmax(np.where(correct, confidence, -np.inf), axis=1)
    selected[~known] = int(fallback_index)
    no_correct = known & ~correct.any(axis=1)
    selected[no_correct] = np.argmax(confidence[no_correct], axis=1)
    return selected, confidence, correct


def summarize_confidence(
    confidence: np.ndarray,
    correct: np.ndarray,
    selected: np.ndarray,
    labels: np.ndarray,
    best_fixed_index: int,
) -> dict[str, Any]:
    rows = np.arange(len(selected))
    selected_conf = confidence[rows, selected]
    fixed_conf = confidence[:, best_fixed_index]
    selected_correct = correct[rows, selected]
    fixed_correct = correct[:, best_fixed_index]
    labels = np.asarray(labels, dtype=float)
    known = np.isfinite(labels)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"n_windows": 0}
        gain = selected_conf[mask] - fixed_conf[mask]
        return {
            "n_windows": int(mask.sum()),
            "oracle_mean_threshold_relative_confidence": float(np.nanmean(selected_conf[mask])),
            "best_fixed_mean_threshold_relative_confidence": float(np.nanmean(fixed_conf[mask])),
            "oracle_mean_confidence_gain_vs_best_fixed": float(np.nanmean(gain)),
            "oracle_median_confidence_gain_vs_best_fixed": float(np.nanmedian(gain)),
            "oracle_correct_window_fraction": float(np.mean(selected_correct[mask])),
            "best_fixed_correct_window_fraction": float(np.mean(fixed_correct[mask])),
            "oracle_correct_window_gain_pp_vs_best_fixed": float(
                100.0
                * (np.mean(selected_correct[mask]) - np.mean(fixed_correct[mask]))
            ),
        }

    return {
        "window_correctness_definition": "raw_probability_threshold_decision_before_eventization",
        "known_window_fraction": float(np.mean(known)),
        "unknown_window_count": int((~known).sum()),
        "overall_known": subset(known),
        "by_label": {
            "seizure_detection_windows": subset(known & (labels == 1.0)),
            "nonseizure_false_alarm_windows": subset(known & (labels == 0.0)),
        },
    }


def expert_window_and_event_stats(
    table: pd.DataFrame,
    threshold: float,
    recordings: pd.DataFrame,
) -> dict[str, Any]:
    raw_positive = 0
    ema_positive = 0
    total = 0
    known = correct = 0
    by_label = {
        "seizure": {"n": 0, "correct": 0},
        "nonseizure": {"n": 0, "correct": 0},
    }
    event_count = 0
    covered_seconds = 0.0
    evaluation_seconds = 0.0
    recording_count = 0
    for recording_id, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        times = ordered.end.to_numpy(float)
        probabilities = ordered.probability.to_numpy(float)
        labels = ordered.label.to_numpy(float)
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError(f"invalid time grid for recording {recording_id}")
        smoothed = causal_ema(probabilities, alpha=1 / 3)
        raw_decision = probabilities >= threshold
        ema_decision = smoothed >= threshold
        events = eventize(times, probabilities, threshold=threshold)
        duration_s = float(
            recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0]
        )
        evaluation_start = float(times[0])
        evaluation_end = max(evaluation_start, duration_s)
        evaluation_seconds += max(0.0, evaluation_end - evaluation_start)
        for event in events:
            # Clip only for the time-occupancy diagnostic; event counts remain
            # exactly those used by the frozen scorer.
            covered_seconds += max(
                0.0,
                min(float(event.end_s), evaluation_end)
                - max(float(event.start_s), evaluation_start),
            )
        event_count += len(events)
        recording_count += 1
        raw_positive += int(raw_decision.sum())
        ema_positive += int(ema_decision.sum())
        total += len(ordered)
        finite = np.isfinite(labels)
        known += int(finite.sum())
        correct += int(np.sum(raw_decision[finite] == (labels[finite] == 1.0)))
        for name, value in (("seizure", 1.0), ("nonseizure", 0.0)):
            mask = finite & (labels == value)
            by_label[name]["n"] += int(mask.sum())
            by_label[name]["correct"] += int(
                np.sum(raw_decision[mask] == (labels[mask] == 1.0))
            )
    by_label_out = {
        name: {
            "n_windows": int(values["n"]),
            "correct_windows": int(values["correct"]),
            "correct_window_fraction": (
                float(values["correct"] / values["n"]) if values["n"] else float("nan")
            ),
        }
        for name, values in by_label.items()
    }
    return {
        "recordings": recording_count,
        "total_windows": total,
        "raw_threshold_positive_fraction": raw_positive / total if total else float("nan"),
        "causal_ema_positive_fraction": ema_positive / total if total else float("nan"),
        "eventized_alarm_time_fraction": (
            covered_seconds / evaluation_seconds if evaluation_seconds else float("nan")
        ),
        "eventized_alarm_covered_seconds": covered_seconds,
        "eventized_event_count": event_count,
        "known_windows": known,
        "unknown_windows": total - known,
        "known_raw_threshold_correct_fraction": correct / known if known else float("nan"),
        "by_label": by_label_out,
    }


def load_expert(round_index: int, model: str, split_kind: str) -> pd.DataFrame:
    model_dir = RUNS / f"round_{round_index}" / model
    path = model_dir / (
        "router_dev_probabilities.parquet"
        if split_kind == "router_dev"
        else "final_test_probabilities.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def audit_expert(
    round_index: int,
    model: str,
    split_kind: str,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> dict[str, Any]:
    model_dir = RUNS / f"round_{round_index}" / model
    probability_path = model_dir / (
        "router_dev_probabilities.parquet"
        if split_kind == "router_dev"
        else "final_test_probabilities.parquet"
    )
    index_path = model_dir / (
        "router_dev_index.parquet"
        if split_kind == "router_dev"
        else "final_test_index.parquet"
    )
    metric_path = model_dir / (
        "router_dev_metrics.json"
        if split_kind == "router_dev"
        else "final_test_metrics.json"
    )
    lock_path = model_dir / "threshold_lock.json"
    run_manifest_path = model_dir / "run_manifest.json"
    table = pd.read_parquet(probability_path)
    index = pd.read_parquet(index_path)
    lock = json.loads(lock_path.read_text())
    saved = json.loads(metric_path.read_text())
    threshold = float(lock["threshold"])
    if lock.get("source") != "expert_train_only":
        raise RuntimeError(f"threshold source is not expert_train_only: {lock_path}")
    if len(table) != len(index) or not table[KEYS].reset_index(drop=True).equals(
        index[KEYS].reset_index(drop=True)
    ):
        raise RuntimeError(f"probability/index alignment failed: {probability_path}")
    if table[KEYS].duplicated().any() or index[KEYS].duplicated().any():
        raise RuntimeError(f"duplicate window key: {probability_path}")
    if not labels_equal(table["label"], index["label"]):
        raise RuntimeError(f"label/index mismatch: {probability_path}")
    probabilities = table["probability"].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"non-finite probability: {probability_path}")
    out_of_range = int(np.sum((probabilities < 0.0) | (probabilities > 1.0)))
    if out_of_range:
        raise RuntimeError(f"probability outside [0,1]: {probability_path}")
    recomputed = v3.score_probabilities(table, seizures, recordings, threshold)
    compare_keys = [
        "true_positive_events",
        "false_alarm_events",
        "truth_events",
        "event_sensitivity",
        "nonseizure_hours",
        "fa_per_24h",
    ]
    diffs = {
        key: float(recomputed[key]) - float(saved[key]) for key in compare_keys
    }
    manifest = json.loads(run_manifest_path.read_text())
    return {
        "round": round_index,
        "model": model,
        "split_kind": split_kind,
        "probability_path": str(probability_path.relative_to(ROOT)),
        "probability_sha256": sha256_file(probability_path),
        "index_sha256": sha256_file(index_path),
        "threshold_lock_sha256": sha256_file(lock_path),
        "threshold": threshold,
        "threshold_source": lock.get("source"),
        "test_accessed": bool(manifest.get("test_accessed", False)),
        "rows": int(len(table)),
        "duplicate_keys": int(table[KEYS].duplicated().sum()),
        "probability_out_of_range": out_of_range,
        "saved_metrics": saved,
        "recomputed_metrics": recomputed,
        "recompute_minus_saved": diffs,
        "window_and_event_stats": expert_window_and_event_stats(
            table, threshold, recordings
        ),
    }


def merge_experts(
    round_index: int, split_kind: str
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    merged: pd.DataFrame | None = None
    thresholds: dict[str, float] = {}
    source_hashes: dict[str, str] = {}
    for model in MODELS:
        model_dir = RUNS / f"round_{round_index}" / model
        probability_path = model_dir / (
            "router_dev_probabilities.parquet"
            if split_kind == "router_dev"
            else "final_test_probabilities.parquet"
        )
        lock_path = model_dir / "threshold_lock.json"
        table = pd.read_parquet(probability_path)
        lock = json.loads(lock_path.read_text())
        thresholds[model] = float(lock["threshold"])
        source_hashes[f"{model}_probabilities"] = sha256_file(probability_path)
        source_hashes[f"{model}_threshold_lock"] = sha256_file(lock_path)
        if table[KEYS].duplicated().any():
            raise RuntimeError(f"duplicate key before expert merge: {probability_path}")
        keep = KEYS + ["label", "probability"]
        renamed = table[keep].rename(columns={"probability": f"prob_{model}"})
        if merged is None:
            merged = renamed
        else:
            # Align labels by the actual immutable window key, rather than by
            # row position.  This catches missing/reordered windows without
            # treating valid NaN boundary labels as mismatches.
            label_check = merged[KEYS + ["label"]].merge(
                renamed[KEYS + ["label"]],
                on=KEYS,
                how="outer",
                validate="one_to_one",
                suffixes=("_left", "_right"),
                indicator=True,
                sort=False,
            )
            if not (label_check["_merge"] == "both").all() or not labels_equal(
                label_check["label_left"], label_check["label_right"]
            ):
                raise RuntimeError(f"label/key mismatch during merge for round {round_index}")
            merged = merged.merge(
                renamed.drop(columns=["label"]),
                on=KEYS,
                how="inner",
                validate="one_to_one",
                sort=False,
            )
    if merged is None or merged.empty:
        raise RuntimeError(f"empty expert merge round={round_index} split={split_kind}")
    return merged, thresholds, source_hashes


def build_oracle_report(
    round_index: int,
    split_kind: str,
    merged: pd.DataFrame,
    thresholds: dict[str, float],
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    best_fixed: str,
    fixed_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    probabilities = merged[[f"prob_{model}" for model in MODELS]].to_numpy(float)
    labels = merged.label.to_numpy(float)
    threshold_array = np.array([thresholds[model] for model in MODELS], dtype=float)
    fallback = MODELS.index(best_fixed)
    selected, confidence, correct = select_oracle(
        probabilities, labels, threshold_array, fallback
    )
    mixed_score = score_selected_mixed(
        merged, selected, threshold_array, seizures, recordings
    )

    # Per-expert causal EMA states are computed independently on each EDF.
    smoothed = np.empty_like(probabilities)
    for _, group in merged.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        indices = ordered.index.to_numpy()
        for j, model in enumerate(MODELS):
            smoothed[indices, j] = causal_ema(
                ordered[f"prob_{model}"].to_numpy(float), alpha=1 / 3
            )
    state_selected, state_confidence, state_correct = select_oracle(
        smoothed, labels, threshold_array, fallback
    )
    selected_state = (
        smoothed[np.arange(len(merged)), state_selected]
        >= threshold_array[state_selected]
    ).astype(float)
    state_score = score_state_table(merged, selected_state, seizures, recordings)

    rows = np.arange(len(selected))
    known = np.isfinite(labels)
    summary = summarize_confidence(
        confidence, correct, selected, labels, fallback
    )
    state_summary = summarize_confidence(
        state_confidence, state_correct, state_selected, labels, fallback
    )
    return {
        "split_kind": split_kind,
        "rows": int(len(merged)),
        "known_label_rows": int(known.sum()),
        "unknown_label_rows": int((~known).sum()),
        "best_fixed_expert_selected_on_router_dev": best_fixed,
        "thresholds": thresholds,
        "window_confidence_oracle": {
            "selection_rule": "known-label correctness first, then bounded threshold-relative confidence; unknown labels fallback to validation Best Fixed",
            "eventization": "selected raw probability minus selected expert threshold, then frozen causal EMA/event rules",
            "post_hoc_upper_bound": True,
            "selected_expert_proportions_all_windows": {
                model: float(np.mean(selected == j)) for j, model in enumerate(MODELS)
            },
            "selected_expert_proportions_known_labels": {
                model: float(np.mean(selected[known] == j))
                for j, model in enumerate(MODELS)
            },
            "no_correct_known_label_fraction": float(
                np.mean(~correct[known].any(axis=1)) if np.any(known) else float("nan")
            ),
            "confidence_summary": summary,
            "event_score": mixed_score,
        },
        "decision_state_oracle": {
            "selection_rule": "independently causal-EMA each expert, then known-label correctness first and bounded confidence",
            "eventization": "select binary expert decision state, then frozen min-duration/merge/refractory event rules",
            "post_hoc_upper_bound": True,
            "selected_expert_proportions_all_windows": {
                model: float(np.mean(state_selected == j))
                for j, model in enumerate(MODELS)
            },
            "confidence_summary_on_ema_state": state_summary,
            "event_score": state_score,
        },
        "fixed_expert_event_scores": fixed_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=range(5))
    args = parser.parse_args()
    if AUDIT_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite audit namespace: {AUDIT_ROOT}")
    AUDIT_ROOT.mkdir(parents=True, exist_ok=False)
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    rounds = [args.round] if args.round is not None else list(range(5))
    expert_rows: list[dict[str, Any]] = []
    round_reports: list[dict[str, Any]] = []
    completed_rounds = 0
    for round_index in rounds:
        round_dir = RUNS / f"round_{round_index}"
        if not all((round_dir / model / "completed.json").is_file() for model in MODELS):
            continue
        dev_tables: dict[str, pd.DataFrame] = {}
        test_tables: dict[str, pd.DataFrame] = {}
        report: dict[str, Any] = {"round": round_index, "experts": [], "oracle": {}}
        for model in MODELS:
            for split_kind in ("router_dev", "final_test"):
                item = audit_expert(
                    round_index, model, split_kind, seizures, recordings
                )
                expert_rows.append(
                    {
                        "round": round_index,
                        "model": model,
                        "split_kind": split_kind,
                        "rows": item["rows"],
                        "threshold": item["threshold"],
                        "saved_event_sensitivity": item["saved_metrics"]["event_sensitivity"],
                        "recomputed_event_sensitivity": item["recomputed_metrics"]["event_sensitivity"],
                        "saved_fa_per_24h": item["saved_metrics"]["fa_per_24h"],
                        "recomputed_fa_per_24h": item["recomputed_metrics"]["fa_per_24h"],
                        "max_abs_recompute_diff": max(
                            abs(float(value)) for value in item["recompute_minus_saved"].values()
                        ),
                        "raw_threshold_positive_fraction": item["window_and_event_stats"]["raw_threshold_positive_fraction"],
                        "causal_ema_positive_fraction": item["window_and_event_stats"]["causal_ema_positive_fraction"],
                        "eventized_alarm_time_fraction": item["window_and_event_stats"]["eventized_alarm_time_fraction"],
                        "eventized_event_count": item["window_and_event_stats"]["eventized_event_count"],
                        "known_windows": item["window_and_event_stats"]["known_windows"],
                        "unknown_windows": item["window_and_event_stats"]["unknown_windows"],
                    }
                )
                report["experts"].append(item)
                if split_kind == "router_dev":
                    dev_tables[model] = load_expert(round_index, model, split_kind)
                else:
                    test_tables[model] = load_expert(round_index, model, split_kind)
        dev_merged, thresholds, dev_hashes = merge_experts(round_index, "router_dev")
        test_merged, test_thresholds, test_hashes = merge_experts(round_index, "final_test")
        if thresholds != test_thresholds:
            raise RuntimeError(f"threshold changed between dev/test for round {round_index}")
        fixed_scores = {
            model: v3.score_probabilities(
                dev_merged[KEYS + ["label", f"prob_{model}"]].rename(
                    columns={f"prob_{model}": "probability"}
                ),
                seizures,
                recordings,
                thresholds[model],
            )
            for model in MODELS
        }
        feasible = [
            model for model in MODELS if fixed_scores[model]["event_sensitivity"] >= 0.80
        ]
        if feasible:
            best_fixed = min(
                feasible,
                key=lambda model: (
                    float(fixed_scores[model]["fa_per_24h"]),
                    -float(fixed_scores[model]["event_sensitivity"]),
                ),
            )
        else:
            best_fixed = max(
                MODELS,
                key=lambda model: (
                    float(fixed_scores[model]["event_sensitivity"]),
                    -float(fixed_scores[model]["fa_per_24h"]),
                ),
            )
        fixed_test_scores = {
            model: v3.score_probabilities(
                test_merged[KEYS + ["label", f"prob_{model}"]].rename(
                    columns={f"prob_{model}": "probability"}
                ),
                seizures,
                recordings,
                thresholds[model],
            )
            for model in MODELS
        }
        report["oracle"]["router_dev"] = build_oracle_report(
            round_index,
            "router_dev",
            dev_merged,
            thresholds,
            seizures,
            recordings,
            best_fixed,
            fixed_scores,
        )
        report["oracle"]["final_test"] = build_oracle_report(
            round_index,
            "final_test",
            test_merged,
            thresholds,
            seizures,
            recordings,
            best_fixed,
            fixed_test_scores,
        )
        report["best_fixed"] = {
            "selected_on_router_dev": best_fixed,
            "router_dev_scores": fixed_scores,
            "final_test_scores_using_dev_selection": fixed_test_scores,
        }
        report["source_hashes"] = {
            "router_dev": dev_hashes,
            "final_test": test_hashes,
        }
        write_json(AUDIT_ROOT / f"round_{round_index}_oracle_recalculated.json", report)
        completed_rounds += 1
        round_reports.append(
            {
                "round": round_index,
                "best_fixed": best_fixed,
                "router_dev_oracle_fa_per_24h": report["oracle"]["router_dev"]["window_confidence_oracle"]["event_score"]["fa_per_24h"],
                "router_dev_oracle_sensitivity": report["oracle"]["router_dev"]["window_confidence_oracle"]["event_score"]["event_sensitivity"],
                "final_test_oracle_fa_per_24h": report["oracle"]["final_test"]["window_confidence_oracle"]["event_score"]["fa_per_24h"],
                "final_test_oracle_sensitivity": report["oracle"]["final_test"]["window_confidence_oracle"]["event_score"]["event_sensitivity"],
                "final_test_state_oracle_fa_per_24h": report["oracle"]["final_test"]["decision_state_oracle"]["event_score"]["fa_per_24h"],
                "final_test_state_oracle_sensitivity": report["oracle"]["final_test"]["decision_state_oracle"]["event_score"]["event_sensitivity"],
            }
        )
    if expert_rows:
        pd.DataFrame(expert_rows).to_csv(AUDIT_ROOT / "expert_metric_audit.csv", index=False)
    if round_reports:
        pd.DataFrame(round_reports).to_csv(AUDIT_ROOT / "oracle_summary.csv", index=False)
    manifest = {
        "protocol_id": NAMESPACE,
        "audit_id": "oracle-recalculation-audit-v1",
        "status": "audit_only_not_frozen",
        "rounds_requested": rounds,
        "completed_rounds": completed_rounds,
        "expert_models": MODELS,
        "source_metric_rule": "recompute from saved probability tables with frozen threshold locks and frozen eventization",
        "threshold_source_required": "expert_train_only",
        "oracle_definitions": {
            "window_confidence_oracle": "post-hoc correctness-first selection on raw probability threshold decisions; event score retains pilot mixed-margin eventization",
            "decision_state_oracle": "post-hoc correctness-first selection on independently causal-EMA expert decision states; eventize selected binary state",
        },
        "unknown_label_policy": "unknown NaN windows excluded from correctness/confidence summaries and retained in temporal grid; fallback is validation-selected Best Fixed",
        "test_policy": "read-only one-time audit of already saved test probabilities; no threshold or model selection uses final test",
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(AUDIT_ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
