"""Compute the H6 v2 causal-policy oracle gate on Train/Dev only.

This is a non-deployable upper-bound analysis. It uses frozen expert event
streams and v2 offline labels, never router predictions and never Eval.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
LABEL_NS = "h6-causal-policy-router-v2"
SOURCE_NS = "h6-tusz-raw-pre-router-v1"
SOURCE_DIR = "h6_tusz_raw_pre_router_v1"
NS = "h6-causal-policy-router-v2"
OUT = ROOT / "outputs/reports" / NS / "oracle"
MODELS = ("singlem", "cbramod", "tcn_gat")
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260830

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_scorer():
    path = ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py"
    spec = importlib.util.spec_from_file_location("h6_v1_scorer_for_v2_oracle", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def summary_ci(rows: pd.DataFrame, scorer, method: str):
    summary = dict(scorer._summary(rows))
    # The generic scorer bootstrap concatenates every patient's full row table
    # for every replicate and is needlessly slow on Train.  Bootstrap only the
    # patient-level sufficient statistics; this is algebraically identical for
    # sensitivity, FA/24h, and delay and keeps the gate reproducible.
    patient = rows.groupby("patient_id", sort=True).agg(
        truth_events=("truth_events", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        delay_count=("delay_count", "sum"), delay_sum_s=("delay_sum_s", "sum"),
        nonseizure_s=("nonseizure_s", "sum"),
    )
    values = {metric: np.empty(BOOTSTRAP, dtype=float) for metric in ("sensitivity", "fa_per_24h", "mean_delay_s")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(patient)
    arrays = {name: patient[name].to_numpy(dtype=float) for name in patient.columns}
    for i in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        truth = arrays["truth_events"][idx].sum()
        fp = arrays["fp"][idx].sum()
        nonseizure = arrays["nonseizure_s"][idx].sum()
        delays = arrays["delay_count"][idx].sum()
        values["sensitivity"][i] = arrays["tp"][idx].sum() / truth if truth else np.nan
        values["fa_per_24h"][i] = fp / (nonseizure / 86400.0) if nonseizure > 0 else np.nan
        values["mean_delay_s"][i] = arrays["delay_sum_s"][idx].sum() / delays if delays else np.nan
    ci = {}
    for metric in ("sensitivity", "fa_per_24h", "mean_delay_s"):
        valid = values[metric][np.isfinite(values[metric])]
        ci[f"{metric}_ci2_5"] = float(np.quantile(valid, .025)) if len(valid) else float("nan")
        ci[f"{metric}_ci50"] = float(np.quantile(valid, .5)) if len(valid) else float("nan")
        ci[f"{metric}_ci97_5"] = float(np.quantile(valid, .975)) if len(valid) else float("nan")
    return summary, ci

def forced_oracle(labels: pd.DataFrame, manifest: dict, event_cache: dict, scorer, allow_reject: bool):
    records = {str(r["relative_edf"]): r for r in manifest["records"]}
    output = []
    for recording, group in labels.groupby("recording", sort=False):
        record = records[str(recording)]
        truths = scorer._truths(record)
        selected = []
        for row in group.sort_values(["future_start_s", "decision_index_v1"]).itertuples(index=False):
            correct = {m: int(getattr(row, f"correct_{m}")) for m in MODELS}
            candidates = [m for m in MODELS if correct[m] == 1]
            if candidates:
                chosen = max(candidates, key=lambda m: (float(getattr(row, f"policy_target_{m}")), m))
            elif not bool(int(row.future_has_seizure)) and not allow_reject:
                chosen = min(MODELS, key=lambda m: (int(getattr(row, f"event_count_{m}")), -float(getattr(row, f"confidence_{m}")), m))
            else:
                continue
            start, end = float(row.future_start_s), float(row.future_end_s)
            selected.extend(e for e in event_cache[str(recording)][chosen] if e.end_s > start and e.start_s < end)
        selected = scorer._dedupe(selected)
        matching = scorer.match_events(selected, truths)
        delays = [max(0.0, selected[p.prediction_index].start_s - truths[p.truth_index][0]) for p in matching.pairs]
        seizure_seconds = sum(max(0.0, b - a) for a, b in truths)
        output.append({"method": "interval_oracle_reject" if allow_reject else "interval_oracle_forced","patient_id": str(record["patient_id"]),"recording": str(recording),"truth_events": len(truths),"tp": len(matching.pairs),"fp": len(matching.unmatched_predictions),"fn": len(matching.unmatched_truths),"delay_count": len(delays),"delay_sum_s": float(sum(delays)),"median_delay_s": float(np.median(delays)) if delays else np.nan,"nonseizure_s": max(0.0, float(record["duration_s"]) - 70.0 - seizure_seconds)})
    return pd.DataFrame(output)

def run_partition(partition: str, scorer):
    label_path = ROOT / "outputs/reports" / LABEL_NS / "labels" / f"{partition}_causal_policy.parquet"
    manifest_path = ROOT / "outputs/reports" / SOURCE_NS / "manifests" / f"{partition}_manifest.json"
    # The inference directory predates the report namespace and intentionally
    # uses underscores; keep this explicit so provenance is not guessed.
    expert_root = ROOT / "outputs/tuh_external" / SOURCE_DIR / partition
    labels = pd.read_parquet(label_path)
    manifest = json.loads(manifest_path.read_text())
    events, expert_thresholds, _ = scorer._read_event_cache(manifest, expert_root)
    oracle_labels = labels.copy()
    for m in MODELS:
        oracle_labels[f"target_weight_{m}"] = oracle_labels[f"policy_target_{m}"]
    rows_by_method = {}
    for m in MODELS:
        rows_by_method[f"fixed_{m}"] = scorer._score_actions(oracle_labels, manifest, events, f"fixed_{m}", None)
    rows_by_method["interval_oracle_reject"] = forced_oracle(oracle_labels, manifest, events, scorer, True)
    rows_by_method["interval_oracle_forced"] = forced_oracle(oracle_labels, manifest, events, scorer, False)
    summaries = []
    for method, rows in rows_by_method.items():
        summary, ci = summary_ci(rows, scorer, method)
        summaries.append({"partition": partition, "method": method, **summary, **ci})
    frame = pd.DataFrame(summaries)
    fixed = frame[frame.method.str.startswith("fixed_")].copy()
    best_sens = float(fixed.sensitivity.max())
    safe = fixed[fixed.sensitivity >= best_sens - .03]
    best = safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0]
    oracle = frame[frame.method == "interval_oracle_reject"].iloc[0]
    forced = frame[frame.method == "interval_oracle_forced"].iloc[0]
    base_fa = float(best.fa_per_24h)
    reduction = lambda value: float(1.0 - value / base_fa) if base_fa > 0 else float("nan")
    gate = {"partition": partition,"best_fixed_method": str(best.method),"best_fixed_sensitivity": float(best.sensitivity),"best_fixed_fa_per_24h": base_fa,"oracle_reject_sensitivity": float(oracle.sensitivity),"oracle_reject_fa_per_24h": float(oracle.fa_per_24h),"oracle_reject_sensitivity_delta_vs_best_fixed": float(oracle.sensitivity - best.sensitivity),"oracle_reject_fa_reduction_vs_best_fixed": reduction(float(oracle.fa_per_24h)),"oracle_forced_sensitivity": float(forced.sensitivity),"oracle_forced_fa_per_24h": float(forced.fa_per_24h),"oracle_forced_fa_reduction_vs_best_fixed": reduction(float(forced.fa_per_24h)),"sensitivity_safe_3pp": bool(float(oracle.sensitivity) >= float(best.sensitivity) - .03),"fa_reduction_at_least_25pct": bool(reduction(float(oracle.fa_per_24h)) >= .25),"oracle_has_any_fa_reduction": bool(float(oracle.fa_per_24h) < base_fa),"expert_thresholds": expert_thresholds,"status":"oracle_gate_candidate"}
    return frame, gate, sha256(manifest_path), sha256(label_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", default="train,dev")
    args = ap.parse_args()
    parts = tuple(x.strip() for x in args.partitions.split(",") if x.strip())
    if any(x not in ("train", "dev") for x in parts):
        raise ValueError("Eval is intentionally excluded")
    manifest_out = OUT / "oracle_manifest.json"
    if manifest_out.exists():
        raise RuntimeError(f"refusing overwrite {manifest_out}")
    OUT.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer()
    results = {}
    for partition in parts:
        frame, gate, manifest_hash, label_hash = run_partition(partition, scorer)
        frame.to_csv(OUT / f"{partition}_oracle_summary.csv", index=False)
        (OUT / f"{partition}_oracle_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + chr(10))
        results[partition] = {"summary": frame.to_dict(orient="records"), "gate": gate, "manifest_sha256": manifest_hash, "label_sha256": label_hash}
    audits = {}
    for partition in parts:
        path = ROOT / "outputs/reports" / LABEL_NS / "labels" / f"{partition}_causal_policy_audit.json"
        audits[partition] = json.loads(path.read_text())
    dev = results.get("dev", {}).get("gate")
    da = audits.get("dev", {})
    counts = da.get("policy_target_argmax_counts", {})
    eligible = int(da.get("policy_eligible_rows", 0))
    max_share = float(max(counts.values(), default=0) / eligible) if eligible else 1.0
    ready = bool(dev and dev["sensitivity_safe_3pp"] and dev["fa_reduction_at_least_25pct"] and max_share <= .95)
    overall = {"release_id": NS,"status":"pre_training_gate","eval_read":False,"training_started":False,"gate_rules":{"dev_oracle_sensitivity_within_3pp":True,"dev_oracle_fa_reduction_at_least_25pct":True,"dev_policy_argmax_max_share_at_most_95pct":True,"no_eval_threshold_or_model_selection":True},"results":results,"label_audit":audits,"dev_policy_argmax_max_share":max_share,"dev_policy_collapse": bool(max_share > .95),"ready_for_router_training": ready,"recommended_action":"start_25_update_smoke_then_formal_training" if ready else "do_not_train; revise_target_or_sampling_and_recompute_oracle","created_utc":datetime.now(timezone.utc).isoformat()}
    manifest_out.write_text(json.dumps(overall, indent=2, sort_keys=True, default=float) + chr(10))
    manifest_out.with_suffix(".sha256").write_text(sha256(manifest_out) + chr(10))
    print(json.dumps(overall, indent=2, sort_keys=True, default=float))

if __name__ == "__main__":
    main()
