"""Audit alarm-duration occupancy for all H6 development diagnostics.

Alarm time is computed from the union of predicted intervals, clipped to the
valid scored part of each recording (after the 70-s warm-up).  False-alarm
time is the part of that union outside the union of seizure intervals.  This
is a read-only Dev audit: it does not retrain, retune, or read TUSZ Eval.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
OUT = ROOT / "outputs/reports/h6-moe-complementarity-alarm-time-audit-v1"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1/dev"
DEV_MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/dev_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")
PREDICTION_SOURCES = {
    "original_h6": ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/scoring_dev_formal_train_only/dev_router_predictions.parquet",
    "target_balanced": ROOT / "outputs/reports/h6-moe-complementarity-gate-balanced-v2/dev_predictions.parquet",
    "mask_balanced": ROOT / "outputs/reports/h6-moe-complementarity-gate-mask-balanced-v1/dev_predictions.parquet",
}
SELECTION_SOURCES = {
    "original_h6": ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/scoring_dev_formal_train_only/threshold_selection.json",
    "target_balanced": ROOT / "outputs/reports/h6-moe-complementarity-gate-balanced-v2/dev_event_scoring/threshold_selection.json",
    "mask_balanced": ROOT / "outputs/reports/h6-moe-complementarity-gate-mask-balanced-v1/dev_event_scoring/threshold_selection.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def clipped_union(events, duration: float) -> list[tuple[float, float]]:
    return merge_intervals([(max(70.0, float(e.start_s)), min(duration, float(e.end_s))) for e in events if float(e.end_s) > 70.0 and float(e.start_s) < duration])


def outside_length(intervals: list[tuple[float, float]], truth: list[tuple[float, float]]) -> float:
    truth_union = merge_intervals(truth)
    total = 0.0
    for start, end in intervals:
        covered = 0.0
        for t0, t1 in truth_union:
            if t1 <= start:
                continue
            if t0 >= end:
                break
            covered += max(0.0, min(end, t1) - max(start, t0))
        total += max(0.0, end - start - covered)
    return total


def action_events(labels: pd.DataFrame, recording: str, group: pd.DataFrame, events, method: str, thresholds: dict[str, float] | None, scorer):
    if method.startswith("fixed_"):
        return events[recording][method.removeprefix("fixed_")]
    if method == "best_fixed":
        return events[recording]["cbramod"]
    if method not in {"router", "router_no_reject"}:
        raise ValueError(method)
    selected = []
    for row in group.itertuples(index=False):
        competence = {m: float(getattr(row, f"router_competence_{m}")) for m in MODELS}
        chosen = max(MODELS, key=competence.get)
        if method == "router" and thresholds is not None:
            allow_reject = float(row.router_reject_probability) >= float(thresholds["reject_probability"]) and max(competence.values()) < float(thresholds["expert_competence"]) and float(row.router_seizure_risk) < float(thresholds["seizure_risk"])
            if allow_reject:
                continue
        start, end = float(row.future_start_s), float(row.future_end_s)
        selected.extend(event for event in events[recording][chosen] if event.end_s > start and event.start_s < end)
    return scorer._dedupe(selected)


def score_source(labels: pd.DataFrame, source_name: str, selection: dict[str, object], events, manifest: dict[str, object], scorer) -> pd.DataFrame:
    thresholds = {k: float(selection[k]) for k in ("reject_probability", "expert_competence", "seizure_risk")}
    record_by_path = {str(r["relative_edf"]): r for r in manifest["records"]}
    methods = ["fixed_singlem", "fixed_cbramod", "fixed_tcn_gat", "best_fixed", "router_no_reject", "router"]
    rows: list[dict[str, object]] = []
    for recording, group in labels.groupby("recording", sort=False):
        record = record_by_path[str(recording)]
        duration = float(record["duration_s"])
        valid = max(0.0, duration - 70.0)
        truth = scorer._truths(record)
        for method in methods:
            preds = action_events(labels, str(recording), group, events, method, thresholds if method == "router" else None, scorer)
            alarm = clipped_union(preds, duration)
            alarm_seconds = float(sum(end - start for start, end in alarm))
            false_seconds = float(outside_length(alarm, truth))
            rows.append({"source": source_name, "method": method, "patient_id": str(record["patient_id"]), "recording": str(recording), "valid_monitoring_s": valid, "alarm_s": alarm_seconds, "false_alarm_s": false_seconds, "alarm_intervals": len(alarm), "truth_intervals": len(truth)})
    return pd.DataFrame(rows)


def summary(record_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = record_metrics.groupby(["source", "method"], sort=False)
    out = grouped.agg(patients=("patient_id", "nunique"), records=("recording", "nunique"), valid_monitoring_s=("valid_monitoring_s", "sum"), alarm_s=("alarm_s", "sum"), false_alarm_s=("false_alarm_s", "sum"), alarm_intervals=("alarm_intervals", "sum")).reset_index()
    out["alarm_fraction_pct"] = 100.0 * out.alarm_s / out.valid_monitoring_s
    out["false_alarm_fraction_pct"] = 100.0 * out.false_alarm_s / out.valid_monitoring_s
    out["alarm_minutes_per_24h"] = out.alarm_s / out.valid_monitoring_s * 1440.0
    out["false_alarm_minutes_per_24h"] = out.false_alarm_s / out.valid_monitoring_s * 1440.0
    return out


def paired_bootstrap(record_metrics: pd.DataFrame, source: str, router_method: str, baseline_method: str) -> pd.DataFrame:
    sub = record_metrics[record_metrics.source.eq(source)]
    patients = sorted(sub.patient_id.unique().tolist())
    pivot = sub.groupby(["patient_id", "method"], sort=False)[["valid_monitoring_s", "alarm_s", "false_alarm_s"]].sum().reset_index()
    base = pivot[pivot.method.eq(baseline_method)].set_index("patient_id")
    route = pivot[pivot.method.eq(router_method)].set_index("patient_id")
    common = sorted(set(base.index) & set(route.index))
    rng = np.random.default_rng(20260902)
    rows = []
    for rep in range(2000):
        chosen = rng.choice(common, len(common), replace=True)
        b = base.loc[chosen].sum(numeric_only=True)
        r = route.loc[chosen].sum(numeric_only=True)
        b_alarm = b.alarm_s / b.valid_monitoring_s
        r_alarm = r.alarm_s / r.valid_monitoring_s
        b_false = b.false_alarm_s / b.valid_monitoring_s
        r_false = r.false_alarm_s / r.valid_monitoring_s
        rows.append({"source": source, "router_method": router_method, "baseline_method": baseline_method, "replicate": rep, "delta_alarm_fraction": r_alarm - b_alarm, "delta_false_alarm_fraction": r_false - b_false, "delta_alarm_minutes_per_24h": (r_alarm - b_alarm) * 1440.0, "delta_false_alarm_minutes_per_24h": (r_false - b_false) * 1440.0})
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        existing = {p.name for p in OUT.iterdir()}
        if existing != {"no_duplicate_overlap_preflight.json"}:
            raise RuntimeError(f"refusing to overwrite {OUT}")
    for path in [DEV_MANIFEST, *PREDICTION_SOURCES.values(), *SELECTION_SOURCES.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    scorer = load_module(ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py", "alarm_time_scorer")
    manifest = json.loads(DEV_MANIFEST.read_text())
    events, expert_thresholds, _ = scorer._read_event_cache(manifest, EXPERT_ROOT)
    all_records = []
    selections = {}
    for source, pred_path in PREDICTION_SOURCES.items():
        labels = pd.read_parquet(pred_path)
        selection = json.loads(SELECTION_SOURCES[source].read_text())
        selections[source] = selection
        all_records.append(score_source(labels, source, selection, events, manifest, scorer))
    record_metrics = pd.concat(all_records, ignore_index=True)
    summary_frame = summary(record_metrics)
    bootstrap = pd.concat([paired_bootstrap(record_metrics, source, "router", "best_fixed") for source in PREDICTION_SOURCES], ignore_index=True)
    # The preflight creates the namespace directory and is retained.
    OUT.mkdir(parents=True, exist_ok=True)
    record_metrics.to_csv(OUT / "record_alarm_time_metrics.csv", index=False)
    summary_frame.to_csv(OUT / "alarm_time_summary.csv", index=False)
    bootstrap.to_csv(OUT / "paired_alarm_time_bootstrap.csv", index=False)
    bootstrap_summary = []
    for (source, router_method, baseline_method), group in bootstrap.groupby(["source", "router_method", "baseline_method"], sort=False):
        bootstrap_summary.append({"source": source, "router_method": router_method, "baseline_method": baseline_method, "delta_alarm_fraction_ci95_low": float(group.delta_alarm_fraction.quantile(0.025)), "delta_alarm_fraction_ci95_high": float(group.delta_alarm_fraction.quantile(0.975)), "delta_false_alarm_fraction_ci95_low": float(group.delta_false_alarm_fraction.quantile(0.025)), "delta_false_alarm_fraction_ci95_high": float(group.delta_false_alarm_fraction.quantile(0.975)), "delta_alarm_minutes_per_24h_ci95_low": float(group.delta_alarm_minutes_per_24h.quantile(0.025)), "delta_alarm_minutes_per_24h_ci95_high": float(group.delta_alarm_minutes_per_24h.quantile(0.975)), "delta_false_alarm_minutes_per_24h_ci95_low": float(group.delta_false_alarm_minutes_per_24h.quantile(0.025)), "delta_false_alarm_minutes_per_24h_ci95_high": float(group.delta_false_alarm_minutes_per_24h.quantile(0.975)), "bootstrap_replicates": 2000})
    (OUT / "paired_alarm_time_bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2, sort_keys=True) + "\n")
    manifest_out = {"release_id": "h6-moe-complementarity-alarm-time-audit-v1", "stage": "development_only_alarm_duration_audit", "status": "complete_train_dev_only", "created_utc": datetime.now(timezone.utc).isoformat(), "eval_read": False, "test_read": False, "valid_scored_start_s": 70.0, "alarm_definition": "union of predicted intervals clipped to valid scored recording", "false_alarm_definition": "alarm union outside seizure interval union", "threshold_source": "TUSZ Dev-only inherited selections; no retuning", "bootstrap_replicates": 2000, "bootstrap_unit": "patient", "expert_thresholds": expert_thresholds, "source_manifests": {name: {"prediction": str(PREDICTION_SOURCES[name].relative_to(ROOT)), "prediction_sha256": sha256(PREDICTION_SOURCES[name]), "selection": str(SELECTION_SOURCES[name].relative_to(ROOT)), "selection_sha256": sha256(SELECTION_SOURCES[name])} for name in PREDICTION_SOURCES}, "dev_manifest": str(DEV_MANIFEST.relative_to(ROOT)), "dev_manifest_sha256": sha256(DEV_MANIFEST), "summary": str((OUT / "alarm_time_summary.csv").relative_to(ROOT)), "summary_sha256": sha256(OUT / "alarm_time_summary.csv"), "paired_bootstrap_summary": str((OUT / "paired_alarm_time_bootstrap_summary.json").relative_to(ROOT)), "paired_bootstrap_summary_sha256": sha256(OUT / "paired_alarm_time_bootstrap_summary.json"), "note": "No detector retraining, no Eval/Test read, no threshold modification."}
    (OUT / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "  manifest.json\n")
    print(summary_frame.to_string(index=False))
    print(json.dumps(bootstrap_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
