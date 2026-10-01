"""Freeze non-degenerate Train-only workpoints for the H6 competence probe.

The first workpoint sweep was anchored to CBraMod's frozen threshold.  That
threshold alarms on essentially every window and can therefore create a
misleading event-level match through event merging.  This replacement keeps
that failed audit intact and writes a new namespace.  It anchors matching to
the only non-degenerate frozen expert (SingleM) and rejects candidate
thresholds whose smoothed window alarm fraction is outside [1%, 50%].

No Dev or Eval outcomes are read or used for threshold selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event, causal_ema
from bfa.evaluation.match import match_events


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE / "matched_workpoints_nondegenerate_v2"
MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/train_manifest.json"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1/train"
MODELS = ("singlem", "cbramod", "tcn_gat")
REFERENCE_MODEL = "singlem"
MIN_ALARM_FRACTION = 0.01
MAX_ALARM_FRACTION = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truths(record: dict[str, object]) -> list[tuple[float, float]]:
    return [
        (max(70.0, float(event["start_s"])), float(event["end_s"]))
        for event in record.get("seizure_events", [])
        if float(event["end_s"]) > 70.0
    ]


def load_model(model: str, records: list[dict[str, object]]):
    loaded = []
    smoothed_values = []
    frozen_threshold = None
    for index, record in enumerate(records, start=1):
        recording = str(record["relative_edf"])
        base = EXPERT_ROOT / model / "fold0_seed17"
        probability_path = base / "probabilities" / Path(recording).with_suffix(".parquet")
        metadata_path = base / "metadata" / Path(recording).with_suffix(".json")
        table = pd.read_parquet(probability_path, columns=["window_end_s", "probability"]).sort_values("window_end_s")
        metadata = json.loads(metadata_path.read_text())
        threshold = float(metadata["threshold"])
        if frozen_threshold is None:
            frozen_threshold = threshold
        elif not np.isclose(frozen_threshold, threshold):
            raise RuntimeError(f"non-constant frozen threshold for {model}")
        times = table.window_end_s.to_numpy(dtype=float)
        values = table.probability.to_numpy(dtype=float)
        smooth = causal_ema(values) if len(values) else values
        loaded.append((times, values, smooth, truths(record), float(record["duration_s"])))
        smoothed_values.append(smooth)
        if index % 1000 == 0:
            print(f"{model}: loaded {index}/{len(records)}", flush=True)
    return loaded, np.concatenate(smoothed_values), float(frozen_threshold)


def score(loaded, threshold: float) -> dict[str, float | int]:
    tp = fp = truth_count = 0
    nonseizure_s = 0.0
    delay_sum = 0.0
    delay_count = 0
    alarm_windows = total_windows = 0
    for times, probabilities, smooth, truth, duration in loaded:
        predictions = eventize_smoothed(times, smooth, threshold=threshold) if len(times) >= 2 else []
        matched = match_events(predictions, truth)
        truth_count += len(truth)
        tp += len(matched.pairs)
        fp += len(matched.unmatched_predictions)
        seizure_s = sum(max(0.0, end - start) for start, end in truth)
        nonseizure_s += max(0.0, duration - 70.0 - seizure_s)
        alarm_windows += int(np.count_nonzero(smooth >= threshold))
        total_windows += int(len(smooth))
        for pair in matched.pairs:
            delay_sum += max(0.0, predictions[pair.prediction_index].start_s - truth[pair.truth_index][0])
            delay_count += 1
    return {
        "threshold": float(threshold),
        "truth_events": int(truth_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(truth_count - tp),
        "sensitivity": float(tp / truth_count) if truth_count else float("nan"),
        "fa_per_24h": float(fp / (nonseizure_s / 86400.0)) if nonseizure_s else float("nan"),
        "mean_delay_s": float(delay_sum / delay_count) if delay_count else float("nan"),
        "alarm_fraction": float(alarm_windows / total_windows) if total_windows else float("nan"),
    }


def eventize_smoothed(
    times: np.ndarray,
    smooth: np.ndarray,
    *,
    threshold: float,
    min_duration_s: float = 4,
    merge_gap_s: float = 10,
    refractory_s: float = 30,
) -> list[Event]:
    """Apply the frozen event rules to already-computed causal EMA values."""
    step_s = float(np.median(np.diff(times)))
    above = smooth >= threshold
    candidates: list[Event] = []
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
        if end_s - start_s >= min_duration_s:
            candidates.append(Event(start_s, end_s, float(np.max(smooth[index : end_index + 1]))))
        index = end_index + 1
    merged: list[Event] = []
    for candidate in candidates:
        if merged and candidate.start_s - merged[-1].end_s <= merge_gap_s:
            previous = merged[-1]
            merged[-1] = Event(
                previous.start_s,
                max(previous.end_s, candidate.end_s),
                max(previous.peak_probability, candidate.peak_probability),
            )
        else:
            merged.append(candidate)
    accepted: list[Event] = []
    for candidate in merged:
        if accepted and candidate.start_s < accepted[-1].end_s + refractory_s:
            continue
        accepted.append(candidate)
    return accepted


def threshold_grid(smoothed_values: np.ndarray, frozen: float) -> np.ndarray:
    # EMA creates nearly continuous values, so enumerating every distinct value
    # would be intractable.  Use a dense, deterministic quantile grid with extra
    # resolution in the upper tail, where all three frozen experts operate.
    quantile_levels = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.99, 100),
                np.linspace(0.9901, 0.9999, 99),
                np.asarray([0.99995, 0.99999, 1.0]),
            ]
        )
    )
    quantiles = np.quantile(smoothed_values, quantile_levels)
    extra = np.asarray([0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.87, 0.9, 0.95, 0.98, 0.99, frozen, 1.000001])
    return np.unique(np.clip(np.concatenate([quantiles, extra]), 0.0, 1.000001))


def row_payload(row: pd.Series) -> dict[str, float | bool]:
    return {
        key: float(row[key])
        for key in ("threshold", "sensitivity", "fa_per_24h", "mean_delay_s", "alarm_fraction")
    } | {"admissible": bool(row["admissible"])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output_manifest = OUT / "manifest.json"
    if output_manifest.exists():
        raise RuntimeError(f"refusing to overwrite frozen workpoints: {output_manifest}")
    cohort = json.loads(MANIFEST.read_text())
    records = list(cohort["records"])
    curves = []
    frozen_rows = {}
    for model in MODELS:
        loaded, smoothed_values, frozen = load_model(model, records)
        grid = threshold_grid(smoothed_values, frozen)
        model_rows = []
        for index, threshold in enumerate(grid, start=1):
            row = {
                "model": model,
                **score(loaded, float(threshold)),
                "is_frozen_threshold": bool(np.isclose(threshold, frozen)),
            }
            row["admissible"] = bool(MIN_ALARM_FRACTION <= row["alarm_fraction"] <= MAX_ALARM_FRACTION)
            model_rows.append(row)
            if index % 100 == 0:
                print(f"{model}: scored {index}/{len(grid)} thresholds", flush=True)
        curve = pd.DataFrame(model_rows)
        if not curve.is_frozen_threshold.any():
            raise RuntimeError(f"frozen threshold missing for {model}")
        frozen_rows[model] = curve[curve.is_frozen_threshold].iloc[0]
        curves.append(curve)

    curve_frame = pd.concat(curves, ignore_index=True)
    reference = frozen_rows[REFERENCE_MODEL]
    if not bool(reference["admissible"]):
        raise RuntimeError("SingleM frozen reference unexpectedly violates the non-degenerate alarm-fraction constraint")
    target_sensitivity = float(reference["sensitivity"])
    target_fa = float(reference["fa_per_24h"])
    selections: dict[str, dict[str, dict[str, float | bool]]] = {
        "frozen": {}, "matched_sensitivity": {}, "matched_fa": {}
    }
    residuals: dict[str, dict[str, float]] = {"matched_sensitivity": {}, "matched_fa": {}}
    for model in MODELS:
        model_curve = curve_frame[curve_frame.model.eq(model)].copy()
        admissible = model_curve[model_curve.admissible].copy()
        if admissible.empty:
            raise RuntimeError(f"no non-degenerate threshold exists for {model}")
        frozen_row = model_curve[model_curve.is_frozen_threshold].iloc[0]
        sensitivity_row = admissible.assign(distance=(admissible.sensitivity - target_sensitivity).abs()).sort_values(
            ["distance", "fa_per_24h", "threshold"], ascending=[True, True, False]
        ).iloc[0]
        fa_row = admissible.assign(distance=(admissible.fa_per_24h - target_fa).abs()).sort_values(
            ["distance", "sensitivity", "threshold"], ascending=[True, False, False]
        ).iloc[0]
        for condition, row in (("frozen", frozen_row), ("matched_sensitivity", sensitivity_row), ("matched_fa", fa_row)):
            selections[condition][model] = row_payload(row)
        residuals["matched_sensitivity"][model] = float(abs(sensitivity_row.sensitivity - target_sensitivity))
        residuals["matched_fa"][model] = float(abs(fa_row.fa_per_24h - target_fa))

    curve_path = OUT / "train_threshold_curves.csv"
    workpoint_path = OUT / "workpoints.json"
    curve_frame.to_csv(curve_path, index=False)
    workpoints = {
        "release_id": NAMESPACE,
        "revision": "nondegenerate-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_partition": "TUSZ Train only",
        "dev_read": False,
        "eval_read": False,
        "reference_model": REFERENCE_MODEL,
        "reference_reason": "only frozen expert with an admissible smoothed-window alarm fraction",
        "admissible_alarm_fraction": [MIN_ALARM_FRACTION, MAX_ALARM_FRACTION],
        "targets": {"sensitivity": target_sensitivity, "fa_per_24h": target_fa},
        "selection_rules": {
            "matched_sensitivity": "among admissible thresholds, minimum absolute sensitivity distance; tie lower FA/24h then higher threshold",
            "matched_fa": "among admissible thresholds, minimum absolute FA/24h distance; tie higher sensitivity then higher threshold",
            "grid": "dense deterministic EMA-probability quantiles with upper-tail refinement plus frozen thresholds",
            "residuals": "residual mismatch is reported, not hidden",
        },
        "conditions": selections,
        "residual_mismatch": residuals,
    }
    workpoint_path.write_text(json.dumps(workpoints, indent=2, sort_keys=True) + "\n")
    manifest = {
        "release_id": NAMESPACE,
        "stage": "train_only_nondegenerate_workpoint_selection_v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dev_used_for_selection": False,
        "eval_read": False,
        "failed_v1_preserved": str(ROOT / "outputs/reports" / NAMESPACE / "matched_workpoints"),
        "source_manifest_sha256": sha256(MANIFEST),
        "threshold_curve_sha256": sha256(curve_path),
        "workpoints_sha256": sha256(workpoint_path),
        "workpoints": workpoints,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(f"{sha256(output_manifest)}  manifest.json\n")
    print(json.dumps(workpoints, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
