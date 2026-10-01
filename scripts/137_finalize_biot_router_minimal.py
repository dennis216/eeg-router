from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events


ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
RUNS = ROOT / "runs" / NAMESPACE / "formal"
REPORT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ["singlem", "cbramod", "tcn_gat"]
METHODS = MODELS + ["best_fixed_expert", "oracle", "biot_router"]
KEYS = ["patient", "recording", "start", "end", "relative_path"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truth_intervals(seizures: pd.DataFrame, recording: str, start: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording]
    return [(max(float(r.start_s), start), float(r.end_s)) for r in rows.itertuples(index=False) if float(r.end_s) > start]


def patient_metrics(table: pd.DataFrame, margins: np.ndarray, seizures: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    working = table[KEYS + ["label"]].copy()
    working["margin"] = margins
    rows: list[dict[str, object]] = []
    for patient, pgroup in working.groupby("patient", sort=True):
        tp = fa = truth_count = 0
        nonseizure_seconds = 0.0
        for recording_id, group in pgroup.groupby("recording", sort=False):
            group = group.sort_values("end", kind="stable")
            times = group.end.to_numpy(dtype=float)
            if len(times) < 2:
                continue
            predictions = eventize(times, group.margin.to_numpy(dtype=float), threshold=0.0)
            evaluation_start = float(times[0])
            duration = float(recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0])
            truths = truth_intervals(seizures, str(recording_id), evaluation_start)
            matched = match_events(predictions, truths)
            tp += len(matched.pairs)
            fa += len(matched.unmatched_predictions)
            truth_count += len(truths)
            nonseizure_seconds += max(0.0, duration - evaluation_start - sum(max(0.0, e - s) for s, e in truths))
        hours = nonseizure_seconds / 3600.0
        rows.append({"patient": patient, "true_positive_events": tp, "false_alarm_events": fa, "truth_events": truth_count, "nonseizure_hours": hours, "event_sensitivity": tp / truth_count if truth_count else 1.0, "fa_per_24h": fa * 24.0 / hours if hours > 0 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    final_path = REPORT / "final_summary.json"
    if final_path.exists():
        print(final_path.read_text())
        return
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    per_patient_rows: list[dict[str, object]] = []
    round_summaries: list[dict] = []
    source_hashes: dict[str, str] = {}
    for round_id in range(5):
        router_dir = RUNS / f"round_{round_id}/router"
        summary_path = router_dir / "router_test_metrics.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        round_summaries.append(summary)
        source_hashes[f"round_{round_id}_router_summary"] = sha256_file(summary_path)
        dev_tables = []
        test_tables = []
        for model in MODELS:
            test_tables.append(pd.read_parquet(RUNS / f"round_{round_id}/{model}/final_test_probabilities.parquet").rename(columns={"probability": f"prob_{model}"}))
        test = test_tables[0][KEYS + ["label", "prob_singlem"]]
        for table, model in zip(test_tables[1:], MODELS[1:], strict=True):
            test = test.merge(table[KEYS + [f"prob_{model}"]], on=KEYS, how="inner", validate="one_to_one", sort=False)
        selection = pd.read_parquet(router_dir / "router_test_selections.parquet")
        test = test.merge(selection[KEYS + ["router_selected_expert", "oracle_selected_expert"]], on=KEYS, how="inner", validate="one_to_one", sort=False)
        thresholds = {m: float(json.loads((RUNS / f"round_{round_id}/{m}/threshold_lock.json").read_text())["threshold"]) for m in MODELS}
        probs = test[[f"prob_{m}" for m in MODELS]].to_numpy(dtype=float)
        signed = np.where(test.label.to_numpy()[:, None] == 1, probs - np.array([thresholds[m] for m in MODELS]), np.array([thresholds[m] for m in MODELS]) - probs)
        best_fixed = str(summary["best_fixed_expert"])
        fixed_indices = {m: i for i, m in enumerate(MODELS)}
        for method in MODELS:
            per = patient_metrics(test, probs[:, fixed_indices[method]] - thresholds[method], seizures, recordings)
            per["method"] = method
            per["round"] = round_id
            per_patient_rows.extend(per.to_dict("records"))
        per = patient_metrics(test, probs[:, fixed_indices[best_fixed]] - thresholds[best_fixed], seizures, recordings)
        per["method"] = "best_fixed_expert"
        per["round"] = round_id
        per_patient_rows.extend(per.to_dict("records"))
        oracle_idx = test.oracle_selected_expert.map(fixed_indices).to_numpy(dtype=int)
        router_idx = test.router_selected_expert.map(fixed_indices).to_numpy(dtype=int)
        per = patient_metrics(test, signed[np.arange(len(test)), oracle_idx], seizures, recordings)
        per["method"] = "oracle"
        per["round"] = round_id
        per_patient_rows.extend(per.to_dict("records"))
        per = patient_metrics(test, probs[np.arange(len(test)), router_idx] - np.array([thresholds[m] for m in MODELS])[router_idx], seizures, recordings)
        per["method"] = "biot_router"
        per["round"] = round_id
        per_patient_rows.extend(per.to_dict("records"))
        source_hashes[f"round_{round_id}_selection"] = sha256_file(router_dir / "router_test_selections.parquet")
    per_patient = pd.DataFrame(per_patient_rows)
    per_patient.to_csv(REPORT / "patient_metrics.csv", index=False)
    rng = np.random.default_rng(3407)
    boot_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for method in METHODS:
        subset = per_patient[per_patient.method == method].copy()
        if subset.patient.nunique() != 22:
            raise RuntimeError(f"expected 22 unique patients for {method}, got {subset.patient.nunique()}")
        patient_ids = np.array(sorted(subset.patient.unique().tolist()))
        by_patient = subset.set_index("patient").loc[patient_ids]
        indices = rng.integers(0, len(patient_ids), size=(args.bootstrap, len(patient_ids)))
        sens = by_patient.event_sensitivity.to_numpy()[indices].mean(axis=1)
        fa = np.nanmedian(by_patient.fa_per_24h.to_numpy()[indices], axis=1)
        for metric, values in (("macro_event_sensitivity", sens), ("median_patient_fa_per_24h", fa)):
            boot_rows.append({"method": method, "metric": metric, "bootstrap_replicates": args.bootstrap, "estimate": float(np.median(values)), "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975))})
        aggregate_rows.append({"method": method, "patients": 22, "event_sensitivity_macro": float(subset.event_sensitivity.mean()), "median_patient_fa_per_24h": float(subset.fa_per_24h.median())})
    pd.DataFrame(boot_rows).to_csv(REPORT / "patient_bootstrap_2000.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(REPORT / "method_patient_summary.csv", index=False)
    payload = {
        "protocol_id": NAMESPACE,
        "status": "PASS",
        "rounds": 5,
        "expert_runs": 15,
        "router_runs": 5,
        "methods": METHODS,
        "round_summaries": round_summaries,
        "bootstrap": {"unit": "patient", "replicates": args.bootstrap, "seed": 3407},
        "threshold_source": "expert_train_only",
        "test_evaluation_count_per_round": 1,
        "test_used_for_router_selection": False,
        "checkpoint_reuse_for_formal_runs": False,
        "source_hashes": source_hashes,
        "completed_at_unix": time.time(),
    }
    final_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT / "release_manifest.json").write_text(json.dumps({"protocol_id": NAMESPACE, "status": "frozen", "final_summary_sha256": sha256_file(final_path), "patient_metrics_sha256": sha256_file(REPORT / "patient_metrics.csv"), "bootstrap_sha256": sha256_file(REPORT / "patient_bootstrap_2000.csv"), "test_evaluation_count_per_round": 1, "threshold_source": "expert_train_only", "model_training": "experts_retrained_and_router_head_only", "checkpoint_reuse": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
