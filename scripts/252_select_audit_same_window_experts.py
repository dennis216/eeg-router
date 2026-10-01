from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.same_window_moe import EXPERTS, aggregate_components, eventize_common, score_events


ROOT = Path("/root/b_false_alarm_atlas")
DEFAULT_PROBABILITY_ROOT = ROOT / "outputs/reports/same-window-router-first-moe-v1/expert_probabilities/tusz"
DEFAULT_OUTPUT = ROOT / "outputs/reports/same-window-router-first-moe-v1/expert_audit"
MANIFEST_ROOT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests"


def load_manifest(partition: str) -> dict[str, object]:
    source = {"train": "train", "val": "dev", "test": "eval"}[partition]
    return json.loads((MANIFEST_ROOT / f"{source}_manifest.json").read_text())


def probability_path(root: Path, partition: str, model: str) -> Path:
    return root / partition / f"{model}.parquet"


def read_probabilities(root: Path, partition: str, model: str) -> pd.DataFrame:
    path = probability_path(root, partition, model)
    if path.is_file():
        return pd.read_parquet(path)
    directory = root / partition / model
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(path)
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def validate_probability_table(frame: pd.DataFrame) -> None:
    forbidden = [column for column in frame if "future" in column.lower() or "horizon" in column.lower()]
    if forbidden:
        raise RuntimeError(f"future-policy columns are forbidden: {forbidden}")
    required = {"patient_id", "recording", "window_start_s", "window_end_s", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing same-window probability columns: {sorted(missing)}")
    length = frame.window_end_s.to_numpy(float) - frame.window_start_s.to_numpy(float)
    if not np.allclose(length, 10.0):
        raise RuntimeError("every expert probability must describe an exact 10 s window")
    for _, group in frame.groupby("recording", sort=False):
        delta = np.diff(np.sort(group.window_start_s.to_numpy(float)))
        if len(delta) and not np.allclose(delta, 2.0):
            raise RuntimeError("expert probability stream is not dense 2 s sliding inference")


def truths(record: dict[str, object]) -> list[tuple[float, float]]:
    output = []
    for item in record.get("seizure_events", []):
        if isinstance(item, dict):
            output.append((float(item.get("start_s", item.get("start", 0))), float(item.get("end_s", item.get("end", 0)))))
        else:
            output.append((float(item[0]), float(item[1])))
    return output


def score_table(frame: pd.DataFrame, manifest: dict[str, object], threshold: float) -> dict[str, float | int]:
    records = {str(row["relative_edf"]): row for row in manifest["records"]}
    parts = []
    for recording, group in frame.groupby("recording", sort=False):
        ordered = group.sort_values("window_end_s")
        events = eventize_common(
            ordered.window_end_s.to_numpy(float),
            ordered.probability.to_numpy(float),
            threshold=threshold,
        )
        record = records[str(recording)]
        parts.append(score_events(events, truths(record), float(ordered.window_end_s.min()), float(record["duration_s"])))
    result = aggregate_components(parts)
    result["positive_window_fraction"] = float((frame.probability >= threshold).mean())
    result["probability_mean"] = float(frame.probability.mean())
    result["probability_std"] = float(frame.probability.std())
    return result


def select_threshold(frame: pd.DataFrame, manifest: dict[str, object]) -> tuple[float, pd.DataFrame, bool]:
    candidates = sorted(set(np.round(np.arange(0.01, 1.00, 0.01), 2).tolist()))
    rows = []
    for threshold in candidates:
        rows.append({"threshold": threshold, **score_table(frame, manifest, threshold)})
    curve = pd.DataFrame(rows)
    feasible = curve[curve.sensitivity >= 0.80]
    used_quantiles = False
    if feasible.empty:
        quantiles = np.unique(np.quantile(frame.probability.to_numpy(float), np.linspace(0.001, 0.999, 199)))
        extra = [float(x) for x in quantiles if 0 < x < 1 and not np.any(np.isclose(curve.threshold, x))]
        rows.extend({"threshold": threshold, **score_table(frame, manifest, threshold)} for threshold in extra)
        curve = pd.DataFrame(rows).drop_duplicates("threshold").sort_values("threshold")
        feasible = curve[curve.sensitivity >= 0.80]
        used_quantiles = True
    if len(feasible):
        pool = feasible.copy()
        pool["fa_rank"] = pool.fa_per_24h.rank(method="average", pct=True)
        pool["false_alarm_time_rank"] = pool.false_alarm_time_min_per_24h.rank(method="average", pct=True)
        pool["joint_false_alarm_score"] = pool.fa_rank + pool.false_alarm_time_rank
        selected = pool.sort_values(
            ["joint_false_alarm_score", "false_alarm_time_min_per_24h", "fa_per_24h", "threshold"],
            ascending=[True, True, True, False],
        ).iloc[0]
    else:
        pool = curve[curve.sensitivity == curve.sensitivity.max()].copy()
        pool["fa_rank"] = pool.fa_per_24h.rank(method="average", pct=True)
        pool["false_alarm_time_rank"] = pool.false_alarm_time_min_per_24h.rank(method="average", pct=True)
        pool["joint_false_alarm_score"] = pool.fa_rank + pool.false_alarm_time_rank
        selected = pool.sort_values(
            ["joint_false_alarm_score", "false_alarm_time_min_per_24h", "fa_per_24h", "threshold"],
            ascending=[True, True, True, False],
        ).iloc[0]
    return float(selected.threshold), curve.reset_index(drop=True), used_quantiles


def collapse_reasons(train: dict[str, object], val: dict[str, object], threshold: float) -> list[str]:
    reasons = []
    if float(val["sensitivity"]) < 0.05:
        reasons.append("validation_sensitivity_near_zero")
    if float(train["sensitivity"]) - float(val["sensitivity"]) > 0.20:
        reasons.append("train_to_val_sensitivity_drop_gt_20pp")
    if float(train["positive_window_fraction"]) < 0.001 or float(train["positive_window_fraction"]) > 0.99:
        reasons.append("almost_no_or_almost_all_windows_alarm")
    if float(val["false_alarm_time_min_per_24h"]) > 720.0:
        reasons.append("false_alarm_time_gt_12h_per_24h")
    if float(train["probability_std"]) < 1e-6:
        reasons.append("near_constant_probability")
    if threshold <= 0.01 or threshold >= 0.99:
        reasons.append("threshold_on_search_boundary")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-root", type=Path, default=DEFAULT_PROBABILITY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = load_manifest("train"), load_manifest("val")
    thresholds: dict[str, float] = {}
    summaries = []
    adjustment_report = []
    for model in EXPERTS:
        train = read_probabilities(args.probability_root, "train", model)
        val = read_probabilities(args.probability_root, "val", model)
        validate_probability_table(train)
        validate_probability_table(val)
        threshold, curve, expanded = select_threshold(train, train_manifest)
        train_metric = score_table(train, train_manifest, threshold)
        val_metric = score_table(val, val_manifest, threshold)
        reasons = collapse_reasons(train_metric, val_metric, threshold)
        thresholds[model] = threshold
        curve.to_csv(args.output / f"{model}_train_threshold_curve.csv", index=False)
        for partition, metric in (("train", train_metric), ("val", val_metric)):
            summaries.append({"split": partition, "method": model, "threshold": threshold, **metric, "status": "ADJUSTMENT_REQUIRED" if reasons else "PASS"})
        adjustment_report.append({"model": model, "quantile_search_added": expanded, "collapse_reasons": reasons, "next_adjustment": "preprocessing parity, then probability calibration" if reasons else "none"})
    pd.DataFrame(summaries).to_csv(args.output / "expert_train_val_audit.csv", index=False)
    (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    (args.output / "adjustments.json").write_text(json.dumps(adjustment_report, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(summaries)[["split", "method", "sensitivity", "fa_per_24h", "false_alarm_time_min_per_24h", "alarm_time_pct", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
