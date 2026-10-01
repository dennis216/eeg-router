"""Dev-only gate and one-time holdout scoring for H7 fine-tuning runs.

This file is intentionally separate from training.  ``score-dev`` never reads
holdout rows; ``gate`` freezes one threshold/checkpoint per detector and
method; ``score-holdout`` refuses to run unless that gate is present and
passing.  The holdout is exploratory because the legacy H7 cohort was already
viewed by earlier work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
NS = os.environ.get("H7_FT_NAMESPACE", "h7-patient-finetune-comparison-v1")
OUT = ROOT / "outputs/reports" / NS
LABEL_DIR = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/labels"
SOURCE = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
MODELS = ("singlem", "cbramod", "tcn_gat")
METHODS = ("head_only", "lora", "full")


def source_checkpoint(model: str) -> Path:
    return ROOT / "runs/v3-groupkfold-confirmatory-v1" / model / "split0_seed17_main/checkpoints/step_05000.pt"


def source_threshold(model: str) -> float:
    """Read the original detector threshold from its frozen CHB manifest."""
    manifest = source_checkpoint(model).parents[1] / "run_manifest.json"
    payload = json.loads(manifest.read_text())
    return float(payload["validation_threshold"])


def load_training_module():
    path = ROOT / "scripts/199_h7_patient_finetune_comparison.py"
    spec = importlib.util.spec_from_file_location("h7_finetune", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def subset_units_from_env() -> set[tuple[str, str, str]] | None:
    """Return an optional patient-unit subset for the reduced H7 pilot."""
    value = os.environ.get("H7_UNIT_SUBSET_MANIFEST")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text())
    return {
        (str(item["model"]), str(item["partition"]), str(item["patient_id"]))
        for item in payload.get("units", [])
    }


def source_records() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for partition in ("train", "dev", "eval"):
        payload = json.loads((SOURCE / "manifests" / f"{partition}_manifest.json").read_text())
        for record in payload["records"]:
            result[str(record["relative_edf"])] = record
    return result


def phase_bounds(records: dict[str, dict[str, Any]], partition: str, patient: str) -> dict[str, tuple[float, float]]:
    rows = [r for r in records.values() if str(r["relative_edf"]).split("/", maxsplit=2)[1] == partition and str(r["patient_id"]) == patient]
    total = sum(float(r["duration_s"]) for r in rows)
    return {"fit": (0.0, 0.60 * total), "dev": (0.60 * total, 0.80 * total), "holdout": (0.80 * total, total)}


def recording_global_offset(records: dict[str, dict[str, Any]], partition: str, patient: str) -> dict[str, float]:
    rows = [r for r in records.values() if str(r["relative_edf"]).split("/", maxsplit=2)[1] == partition and str(r["patient_id"]) == patient]
    offset = 0.0
    result: dict[str, float] = {}
    for row in sorted(rows, key=lambda item: str(item["relative_edf"])):
        result[str(row["relative_edf"])] = offset
        offset += float(row["duration_s"])
    return result


def truths_for(record: dict[str, Any], phase_name: str, offset: float, bounds: tuple[float, float]) -> tuple[list[tuple[float, float]], float]:
    # Clip only at the phase boundary.  A truth crossing a boundary is counted
    # in the phase containing its center, which matches the window labels.
    lo, hi = bounds
    intervals: list[tuple[float, float]] = []
    for event in record.get("seizure_events", []):
        start, end = float(event["start_s"]), float(event["end_s"])
        center = offset + (start + end) / 2.0
        if lo <= center < hi:
            intervals.append((max(start, lo - offset), min(end, hi - offset)))
    nonseizure_s = max(0.0, hi - lo - sum(end - start for start, end in intervals))
    return intervals, nonseizure_s / 3600.0


def score_table(table: pd.DataFrame, records: dict[str, dict[str, Any]], partition: str, patient: str, phase_name: str, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    rec_offset = recording_global_offset(records, partition, patient)
    bounds = phase_bounds(records, partition, patient)[phase_name]
    detail: list[dict[str, Any]] = []
    tp = fp = truth_n = 0
    nonseizure_h = 0.0
    for recording, group in table.groupby("recording", sort=True):
        ordered = group.sort_values("context_end_s", kind="stable")
        if len(ordered) < 2:
            continue
        record = records[str(recording)]
        truths, hours = truths_for(record, phase_name, rec_offset[str(recording)], bounds)
        predictions = eventize(ordered.context_end_s.to_numpy(float), ordered.probability.to_numpy(float), threshold=threshold)
        matched = match_events(predictions, truths, minimum_overlap_s=1.0)
        tp += len(matched.pairs); fp += len(matched.unmatched_predictions); truth_n += len(truths); nonseizure_h += hours
        detail.append({"recording": recording, "patient_id": patient, "phase": phase_name, "truth_events": len(truths), "true_positive_events": len(matched.pairs), "false_alarm_events": len(matched.unmatched_predictions), "nonseizure_hours": hours})
    sensitivity = tp / truth_n if truth_n else 1.0
    return {"threshold": float(threshold), "true_positive_events": tp, "false_alarm_events": fp, "truth_events": truth_n, "event_sensitivity": sensitivity, "nonseizure_hours": nonseizure_h, "fa_per_24h": fp * 24.0 / nonseizure_h if nonseizure_h > 0 else float("nan")}, pd.DataFrame(detail)


def predict_unit(
    model: str,
    method: str,
    partition: str,
    patient: str,
    phase_name: str,
    device: torch.device,
    checkpoint_path: Path,
    batch_size: int = 8,
    output_method: str | None = None,
    seed: int = 17,
) -> pd.DataFrame:
    """Predict one patient phase from a frozen or adapted checkpoint.

    ``method='head_only'`` with the source checkpoint is the untouched frozen
    detector.  Adapted checkpoints retain their method so LoRA
    parametrizations are reconstructed before loading the state dict.
    """
    module = load_training_module()
    labels = module.load_label_table()
    unit = module.load_unit_table(labels, model, patient, partition)
    unit = unit[unit.phase == phase_name].reset_index(drop=True)
    if unit.empty:
        return pd.DataFrame(columns=["patient_id", "recording", "context_end_s", "probability", "label"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    source = source_checkpoint(model)
    encoder, head, meta = module.model_and_head(model, method, source, device)
    if checkpoint_path != source:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder"], strict=True)
        head.load_state_dict(state["head"], strict=True)
    encoder.eval(); head.eval()
    cache = module.RecordingViewCache(model, max_recordings=module.MAX_RECORDINGS)
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(unit), batch_size):
            rows = unit.iloc[start : start + batch_size]
            windows, quality = module.batch_contexts(rows, cache)
            logits = module.forward_context(model, encoder, head, windows, quality, device)
            probabilities.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
    output = unit[["patient_id", "recording", "context_end_s", "label"]].copy()
    output["probability"] = np.asarray(probabilities, dtype=np.float32)
    output["model"] = model; output["method"] = output_method or method; output["partition"] = partition; output["seed"] = seed
    return output


def predict_run(run_dir: Path, phase_name: str, device: torch.device, batch_size: int = 8) -> pd.DataFrame:
    parts = run_dir.name.split("__")
    if len(parts) != 5 or not parts[-1].startswith("seed"):
        raise ValueError(f"unexpected run directory {run_dir.name}")
    model, method, partition, patient, seed_text = parts
    seed = int(seed_text.removeprefix("seed"))
    return predict_unit(model, method, partition, patient, phase_name, device, run_dir / "checkpoint.pt", batch_size=batch_size, seed=seed)


def eligible_units(labels: pd.DataFrame) -> list[tuple[str, str, str]]:
    units = (
        labels[labels.phase == "fit"]
        .groupby(["model", "partition", "patient_id"], sort=True)
        .size()
        .reset_index(name="n")
    )
    result = [(str(row.model), str(row.partition), str(row.patient_id)) for row in units.itertuples(index=False)]
    subset = subset_units_from_env()
    return [unit for unit in result if subset is None or unit in subset]


def pooled_phase_metric(
    table: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    phase_name: str,
    threshold: float,
) -> dict[str, Any]:
    """Aggregate event counts across patients before deriving FA/24 h."""
    metrics: list[dict[str, Any]] = []
    for (partition, patient), group in table.groupby(["partition", "patient_id"], sort=True):
        metric, _ = score_table(group, records, str(partition), str(patient), phase_name, threshold)
        metrics.append(metric)
    # score_table's true_positive_events is the matched count; its truth_events
    # is the denominator.  Keep both names out of the serialized output.
    tp = sum(int(m["true_positive_events"]) for m in metrics)
    fp = sum(int(m["false_alarm_events"]) for m in metrics)
    truth_events = sum(int(m["truth_events"]) for m in metrics)
    hours = sum(float(m["nonseizure_hours"]) for m in metrics)
    return {
        "threshold": float(threshold),
        "true_positive_events": tp,
        "false_alarm_events": fp,
        "truth_events": truth_events,
        "event_sensitivity": tp / truth_events if truth_events else 1.0,
        "nonseizure_hours": hours,
        "fa_per_24h": fp * 24.0 / hours if hours > 0 else float("nan"),
        "patient_metrics": metrics,
    }


def score_dev(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    records = source_records(); prediction_root = OUT / "dev_predictions"
    prediction_root.mkdir(parents=True, exist_ok=False)
    labels = load_training_module().load_label_table()
    summary: list[dict[str, Any]] = []
    # Build the untouched frozen reference on exactly the same eligible
    # patient-detector units.  This is required for a fair recalibration and
    # adaptation comparison; the old H7 event table contains a slightly
    # different 70-unit cohort.
    for model, partition, patient in eligible_units(labels):
        table = predict_unit(model, "head_only", partition, patient, "dev", device, source_checkpoint(model), batch_size=args.batch_size, output_method="frozen_original", seed=17)
        path = prediction_root / f"frozen__{model}__{partition}__{patient}.parquet"
        table.to_parquet(path, index=False)
        summary.append({"run": path.stem, "kind": "frozen_original", "model": model, "method": "frozen_original", "partition": partition, "patient_id": patient, "rows": len(table), "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        print(json.dumps({"run": path.stem, "kind": "frozen_original", "rows": len(table)}), flush=True)
    runs = sorted((OUT / "runs").glob("*") if (OUT / "runs").exists() else [])
    for run in runs:
        if not (run / "checkpoint.pt").exists():
            continue
        parts = run.name.split("__"); model, method, partition, patient = parts[:4]
        table = predict_run(run, "dev", device, batch_size=args.batch_size)
        out = prediction_root / f"{run.name}.parquet"; table.to_parquet(out, index=False)
        summary.append({"run": run.name, "kind": "adapted", "model": model, "method": method, "partition": partition, "patient_id": patient, "rows": len(table), "checkpoint": str((run / "checkpoint.pt").relative_to(ROOT)), "path": str(out.relative_to(ROOT)), "sha256": sha256(out)})
        print(json.dumps({"run": run.name, "rows": len(table)}), flush=True)
    pd.DataFrame(summary).to_csv(prediction_root / "index.csv", index=False)
    atomic_json(prediction_root / "manifest.json", {"release_id": NS, "stage": "dev_scoring", "holdout_read": False, "eligible_units": len(eligible_units(labels)), "prediction_tables": len(summary), "frozen_thresholds": {model: source_threshold(model) for model in MODELS}, "created_utc": pd.Timestamp.utcnow().isoformat(), "index_sha256": sha256(prediction_root / "index.csv")})


def baseline_dev_metrics(records: dict[str, dict[str, Any]], model: str) -> dict[str, float]:
    events = pd.read_parquet(ROOT / "outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet")
    events = events[(events.model == model) & (events.phase == "dev")]
    tp = int(events.matched_seizure.sum()); fp = int(events.is_fp.sum())
    truth = 0; hours = 0.0
    for (partition, patient), group in events.groupby(["partition", "patient_id"]):
        offsets = recording_global_offset(records, str(partition), str(patient)); bounds = phase_bounds(records, str(partition), str(patient))["dev"]
        for recording in group.recording.unique():
            t, h = truths_for(records[str(recording)], "dev", offsets[str(recording)], bounds); truth += len(t); hours += h
    return {"event_sensitivity": tp / truth if truth else 1.0, "fa_per_24h": fp * 24.0 / hours if hours else float("nan"), "true_positive_events": tp, "false_alarm_events": fp, "truth_events": truth, "nonseizure_hours": hours}


def _sweep_thresholds(combined: pd.DataFrame, records: dict[str, dict[str, Any]], baseline_sensitivity: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pooled = pooled_phase_metric(combined, records, "dev", float(threshold))
        candidate = {k: v for k, v in pooled.items() if k != "patient_metrics"}
        candidate["sensitivity_gate"] = candidate["event_sensitivity"] >= baseline_sensitivity - 0.03
        candidates.append(candidate)
    feasible = [candidate for candidate in candidates if candidate["sensitivity_gate"]]
    selected = min(feasible, key=lambda c: (c["fa_per_24h"], -c["threshold"])) if feasible else None
    return candidates, selected


def gate(args: argparse.Namespace) -> None:
    del args
    prediction_root = OUT / "dev_predictions"; index_path = prediction_root / "index.csv"
    if not index_path.exists(): raise RuntimeError("run score-dev first")
    records = source_records(); index = pd.read_csv(index_path)
    if "kind" not in index.columns:
        raise RuntimeError("dev prediction index predates frozen-reference support; use a new namespace")
    rows: list[dict[str, Any]] = []
    frozen_tables: dict[str, pd.DataFrame] = {}
    baseline_rows: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        group = index[(index.kind == "frozen_original") & (index.model == model)]
        if group.empty:
            raise RuntimeError(f"missing frozen dev reference for {model}")
        combined = pd.concat([pd.read_parquet(ROOT / path) for path in group.path], ignore_index=True)
        frozen_tables[model] = combined
        original = pooled_phase_metric(combined, records, "dev", source_threshold(model))
        recal_sweep, recal_selected = _sweep_thresholds(combined, records, original["event_sensitivity"])
        baseline_rows[model] = {
            "original_threshold": source_threshold(model),
            "original": {k: v for k, v in original.items() if k != "patient_metrics"},
            "recalibrated": recal_selected,
            "recalibration_sweep": recal_sweep,
        }
        rows.append({"model": model, "method": "frozen_original", "status": "REFERENCE", "baseline_sensitivity": original["event_sensitivity"], "baseline_fa_per_24h": original["fa_per_24h"], "selected_threshold": source_threshold(model), "selected_sensitivity": original["event_sensitivity"], "selected_fa_per_24h": original["fa_per_24h"], "sensitivity_loss": 0.0, "checkpoint": str(source_checkpoint(model).relative_to(ROOT))})
        rows.append({"model": model, "method": "frozen_dev_recalibrated", "status": "REFERENCE" if recal_selected else "dev_negative", "baseline_sensitivity": original["event_sensitivity"], "baseline_fa_per_24h": original["fa_per_24h"], "selected_threshold": recal_selected["threshold"] if recal_selected else None, "selected_sensitivity": recal_selected["event_sensitivity"] if recal_selected else None, "selected_fa_per_24h": recal_selected["fa_per_24h"] if recal_selected else None, "sensitivity_loss": (recal_selected["event_sensitivity"] - original["event_sensitivity"]) if recal_selected else None, "checkpoint": str(source_checkpoint(model).relative_to(ROOT))})
    for (model, method), group in index[index.kind == "adapted"].groupby(["model", "method"], sort=True):
        combined = pd.concat([pd.read_parquet(ROOT / path) for path in group.path], ignore_index=True)
        baseline = baseline_rows[str(model)]["original"]
        sweep, selected = _sweep_thresholds(combined, records, float(baseline["event_sensitivity"]))
        checkpoint_paths = sorted(set(str(value) for value in group.checkpoint.dropna()))
        rows.append({"model": model, "method": method, "status": "PASS" if selected else "dev_negative", "baseline_sensitivity": baseline["event_sensitivity"], "baseline_fa_per_24h": baseline["fa_per_24h"], "selected_threshold": selected["threshold"] if selected else None, "selected_sensitivity": selected["event_sensitivity"] if selected else None, "selected_fa_per_24h": selected["fa_per_24h"] if selected else None, "sensitivity_loss": (selected["event_sensitivity"] - baseline["event_sensitivity"]) if selected else None, "checkpoint": checkpoint_paths[0] if checkpoint_paths else None, "checkpoint_candidates": checkpoint_paths, "sweep": sweep})
    out = pd.DataFrame([{k: v for k, v in row.items() if k not in {"sweep", "checkpoint_candidates"}} for row in rows]); out.to_csv(OUT / "dev_gate.csv", index=False)
    atomic_json(OUT / "dev_gate.json", {"release_id": NS, "stage": "dev_gate_frozen", "status": "complete", "holdout_read": False, "threshold_rule": "pooled dev sensitivity >= frozen original - 3 percentage points; then minimum FA/24h", "checkpoint_rule": "final checkpoint currently scored; intermediate checkpoints are retained and listed for audit", "methods": rows, "frozen_baselines": baseline_rows, "template_reference": "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1/holdout_metrics.csv", "created_utc": pd.Timestamp.utcnow().isoformat(), "dev_gate_sha256": sha256(OUT / "dev_gate.csv")})
    print(out.to_string(index=False))


def bootstrap_metrics(patient_metrics: pd.DataFrame, seed: int = 17, n: int = 2000) -> dict[str, float]:
    if patient_metrics.empty: return {"sensitivity_ci_low": float("nan"), "sensitivity_ci_high": float("nan"), "fa_ci_low": float("nan"), "fa_ci_high": float("nan")}
    rng = np.random.default_rng(seed); values_s = []; values_f = []; patients = patient_metrics.patient_id.unique()
    for _ in range(n):
        sampled = patient_metrics.iloc[rng.integers(0, len(patient_metrics), len(patients))]
        values_s.append(float(sampled.true_positive_events.sum() / max(1, sampled.truth_events.sum())))
        values_f.append(float(sampled.false_alarm_events.sum() * 24.0 / max(1e-9, sampled.nonseizure_hours.sum())))
    return {"sensitivity_ci_low": float(np.quantile(values_s, .025)), "sensitivity_ci_high": float(np.quantile(values_s, .975)), "fa_ci_low": float(np.quantile(values_f, .025)), "fa_ci_high": float(np.quantile(values_f, .975))}


def score_holdout(args: argparse.Namespace) -> None:
    gate_path = OUT / "dev_gate.json"
    if not gate_path.exists(): raise RuntimeError("dev gate is not frozen")
    gate_payload = json.loads(gate_path.read_text())
    passed = {(str(row["model"]), str(row["method"])): row for row in gate_payload["methods"] if row["status"] == "PASS"}
    if not passed:
        raise RuntimeError("no method passed the dev safety gate; holdout must remain unread")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); records = source_records(); root = OUT / "holdout_predictions"; root.mkdir(parents=True, exist_ok=False); metrics_rows: list[dict[str, Any]] = []
    labels = load_training_module().load_label_table()

    # Score the untouched frozen detector on the exact same eligible units
    # before any adapted method.  This gives a paired holdout reference and
    # makes the dev-recalibrated comparison auditable without reusing the
    # legacy 70-patient H7 event table as the primary baseline.
    frozen_rows = {(str(row["model"]), str(row["method"])): row for row in gate_payload["methods"] if row["method"] in {"frozen_original", "frozen_dev_recalibrated"}}
    frozen_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for model, partition, patient in eligible_units(labels):
        table = predict_unit(model, "head_only", partition, patient, "holdout", device, source_checkpoint(model), batch_size=args.batch_size, output_method="frozen_original", seed=17)
        path = root / f"frozen__{model}__{partition}__{patient}.parquet"; table.to_parquet(path, index=False)
        frozen_tables[(model, partition, patient)] = table
        original_gate = frozen_rows.get((model, "frozen_original"))
        if original_gate is None:
            raise RuntimeError(f"missing frozen gate row for {model}")
        original_threshold = float(original_gate["selected_threshold"])
        metric, detail = score_table(table, records, partition, patient, "holdout", original_threshold)
        detail["model"] = model; detail["method"] = "frozen_original"; detail["partition"] = partition; detail["patient_id"] = patient
        metrics_rows.append({"run": path.stem, "model": model, "method": "frozen_original", "partition": partition, "patient_id": patient, **metric, "threshold_source": "frozen_source_validation_only", "prediction_sha256": sha256(path)})

        recal_gate = frozen_rows.get((model, "frozen_dev_recalibrated"))
        if recal_gate is not None and pd.notna(recal_gate.get("selected_threshold")):
            recal_threshold = float(recal_gate["selected_threshold"])
            recal_metric, recal_detail = score_table(table, records, partition, patient, "holdout", recal_threshold)
            recal_detail["model"] = model; recal_detail["method"] = "frozen_dev_recalibrated"; recal_detail["partition"] = partition; recal_detail["patient_id"] = patient
            metrics_rows.append({"run": path.stem, "model": model, "method": "frozen_dev_recalibrated", "partition": partition, "patient_id": patient, **recal_metric, "threshold_source": "dev-only", "prediction_sha256": sha256(path)})
    for run in sorted((OUT / "runs").glob("*")):
        parts = run.name.split("__");
        if len(parts) != 5: continue
        model, method, partition, patient = parts[:4]
        if (model, method) not in passed: continue
        table = predict_run(run, "holdout", device, batch_size=args.batch_size); path = root / f"{run.name}.parquet"; table.to_parquet(path, index=False)
        threshold = float(passed[(model, method)]["selected_threshold"]); metric, detail = score_table(table, records, partition, patient, "holdout", threshold); detail["model"] = model; detail["method"] = method; detail["partition"] = partition; detail["patient_id"] = patient; metrics_rows.append({"run": run.name, "model": model, "method": method, "partition": partition, "patient_id": patient, **metric, "threshold_source": "dev-only", "prediction_sha256": sha256(path)})
    metrics = pd.DataFrame(metrics_rows); metrics.to_csv(root / "holdout_patient_metrics.csv", index=False); summary: list[dict[str, Any]] = []
    for (model, method), group in metrics.groupby(["model", "method"], sort=True):
        row = {"model": model, "method": method, "runs": len(group), "event_sensitivity": group.true_positive_events.sum() / max(1, group.truth_events.sum()), "fa_per_24h": group.false_alarm_events.sum() * 24.0 / max(1e-9, group.nonseizure_hours.sum()), **bootstrap_metrics(group, seed=17, n=2000), "reference_only": False}; summary.append(row)

    # The earlier H7 template suppressor was frozen on a 70-patient
    # exploratory cohort.  Preserve it as an explicitly labelled reference,
    # never as if it were rerun on this 72-unit fine-tuning cohort.
    template_path = ROOT / "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1/holdout_metrics.csv"
    template_reference: list[dict[str, Any]] = []
    if template_path.exists():
        template = pd.read_csv(template_path)
        template.to_csv(root / "template_holdout_reference.csv", index=False)
        for row in template.itertuples(index=False):
            if str(row.method) == "baseline":
                continue
            template_reference.append({"model": "all_detectors", "method": f"template_{row.method}", "runs": int(row.patients), "event_sensitivity": float(row.pooled_sensitivity), "fa_per_24h": float(row.pooled_FA_24h), "sensitivity_ci_low": float("nan"), "sensitivity_ci_high": float("nan"), "fa_ci_low": float("nan"), "fa_ci_high": float("nan"), "reference_only": True})
        summary.extend(template_reference)
    pd.DataFrame(summary).to_csv(root / "holdout_summary.csv", index=False)
    atomic_json(root / "manifest.json", {"release_id": NS, "stage": "exploratory_holdout_scoring", "status": "complete", "exploratory": True, "holdout_evaluation_count_per_method": 1, "threshold_source": "dev-only (adapted and recalibrated); frozen source validation-only for original reference", "patient_bootstrap": 2000, "methods": summary, "template_reference": str((root / "template_holdout_reference.csv").relative_to(ROOT)) if template_reference else None, "created_utc": pd.Timestamp.utcnow().isoformat(), "metrics_sha256": sha256(root / "holdout_patient_metrics.csv")})
    print(pd.DataFrame(summary).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score-dev", "score-holdout"):
        p = sub.add_parser(name); p.add_argument("--device"); p.add_argument("--batch-size", type=int, default=8)
    sub.add_parser("gate")
    args = parser.parse_args()
    if args.command == "score-dev": score_dev(args)
    elif args.command == "gate": gate(args)
    else: score_holdout(args)


if __name__ == "__main__": main()
