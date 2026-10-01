"""Audit whether MoE has useful complementarity and whether its gate collapsed.

This is an audit-only diagnostic.  It uses frozen TUSZ expert event outputs and
the already-frozen H6 train/dev policy labels.  It deliberately reads only
official train/dev records and patient-local fit/dev portions; no Eval or
patient-local holdout rows are loaded.  No model is trained and no threshold
is changed.

The audit separates three questions that are often conflated:

1. Are false-positive intervals complementary (mixed masks such as 110/101)?
2. Does an oracle have useful headroom under the existing fixed experts?
3. Is the observed SingLEM-heavy routing prior caused by the labels/data or by
   a learned gate that ignores minority winners?
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NS = "h6-moe-complementarity-gate-audit-v2"
OUT = ROOT / "outputs" / "reports" / NS
EVENTS = ROOT / "outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet"
H6_LABEL_DIR = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/labels"
H6_PRED = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/scoring_dev_formal_train_only/dev_router_predictions.parquet"
H6_ORACLE = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/oracle/dev_oracle_gate.json"
H6_ORACLE_SUMMARY = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/oracle/oracle_manifest.json"
H7_DIR = ROOT / "outputs/reports/h7-personalized-consensus-fp-suppressor-v1"
PARTS = ("train", "dev")
PHASES = ("fit", "dev")
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
MASKS = tuple(f"{i:03b}" for i in range(8))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    out: list[tuple[float, float]] = []
    for start, end in clean:
        if out and start <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def interval_intersection_seconds(left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    for a, b in merge_intervals(left):
        for c, d in merge_intervals(right):
            total += max(0.0, min(b, d) - max(a, c))
    return total


def event_match_count(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> int:
    """Greedy one-to-one overlap count for event-level complementarity."""
    used: set[int] = set()
    matched = 0
    for a, b in sorted(left):
        choices = [
            (j, max(0.0, min(b, d) - max(a, c)))
            for j, (c, d) in enumerate(sorted(right))
            if j not in used and min(b, d) > max(a, c)
        ]
        if choices:
            j, _ = max(choices, key=lambda x: x[1])
            used.add(j)
            matched += 1
    return matched


def phase_at(global_time: float, total: float) -> str:
    if global_time < 0.60 * total:
        return "fit"
    if global_time < 0.80 * total:
        return "dev"
    return "holdout"


def load_record_metadata() -> dict[tuple[str, str], dict[str, object]]:
    """Build patient-local chronology from official train/dev manifests only."""
    records: dict[tuple[str, str], dict[str, object]] = {}
    for part in PARTS:
        path = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests" / f"{part}_manifest.json"
        payload = json.loads(path.read_text())
        by_patient: dict[str, list[dict[str, object]]] = {}
        for row in payload["records"]:
            by_patient.setdefault(str(row["patient_id"]), []).append(row)
        for patient, rows in by_patient.items():
            ordered = sorted(rows, key=lambda r: str(r["relative_edf"]))
            total = sum(float(r["duration_s"]) for r in ordered)
            offset = 0.0
            for row in ordered:
                duration = float(row["duration_s"])
                # Keep seizure intervals that can intersect fit/dev; intervals
                # wholly inside the patient-local holdout are never processed.
                truth = []
                for event in row.get("seizure_events", []):
                    start, end = float(event["start_s"]), float(event["end_s"])
                    if end > 0.0 and start < min(duration, 0.80 * total - offset):
                        truth.append((max(0.0, start), min(duration, end)))
                key = (part, str(row["relative_edf"]))
                records[key] = {
                    "partition": part,
                    "patient_id": patient,
                    "recording": str(row["relative_edf"]),
                    "duration_s": duration,
                    "offset_s": offset,
                    "patient_total_s": total,
                    "truth": truth,
                    "manifest_sha256": sha256(path),
                }
                offset += duration
    return records


def load_fp_events() -> pd.DataFrame:
    """Read only train/dev + fit/dev + FP rows using parquet predicate filters."""
    df = pd.read_parquet(
        EVENTS,
        filters=[
            ("partition", "in", list(PARTS)),
            ("phase", "in", list(PHASES)),
            ("is_fp", "=", 1),
        ],
    )
    required = {"partition", "patient_id", "recording", "model", "start_s", "end_s", "phase", "is_fp"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"event source missing columns: {sorted(missing)}")
    df = df[df.partition.isin(PARTS) & df.phase.isin(PHASES) & (df.is_fp == 1)].copy()
    if set(df.partition.unique()) - set(PARTS) or set(df.phase.unique()) - set(PHASES):
        raise RuntimeError("holdout rows passed the source filter")
    return df


def build_background_segments(records: dict[tuple[str, str], dict[str, object]], events: pd.DataFrame) -> pd.DataFrame:
    """Create atomic background intervals, including the all-safe 000 mask."""
    rows: list[dict[str, object]] = []
    by_record = {
        key: group.sort_values(["start_s", "end_s"])
        for key, group in events.groupby(["partition", "recording"], sort=False)
    }
    for key, info in records.items():
        part, recording = key
        group = by_record.get(key, events.iloc[0:0])
        duration = float(info["duration_s"])
        total = float(info["patient_total_s"])
        offset = float(info["offset_s"])
        boundaries = {0.0, duration}
        for cutoff in (0.60, 0.80):
            local = cutoff * total - offset
            if 0.0 < local < duration:
                boundaries.add(float(local))
        truth = [(float(a), float(b)) for a, b in info["truth"]]
        for start, end in truth:
            boundaries.update((max(0.0, start), min(duration, end)))
        model_intervals: dict[str, list[tuple[float, float]]] = {m: [] for m in MODELS}
        for row in group.itertuples(index=False):
            model = str(row.model)
            if model not in model_intervals:
                continue
            start, end = max(0.0, float(row.start_s)), min(duration, float(row.end_s))
            if end <= start:
                continue
            model_intervals[model].append((start, end))
            boundaries.update((start, end))
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            mid = (start + end) / 2.0
            phase = phase_at(offset + mid, total)
            if phase not in PHASES:
                continue
            if any(min(end, truth_end) > max(start, truth_start) for truth_start, truth_end in truth):
                continue
            mask = "".join(
                "1" if any(min(end, b) > max(start, a) for a, b in model_intervals[model]) else "0"
                for model in MODELS
            )
            rows.append(
                {
                    "partition": part,
                    "phase": phase,
                    "patient_id": str(info["patient_id"]),
                    "recording": recording,
                    "start_s": start,
                    "end_s": end,
                    "mask": mask,
                    "mask_code": f"m{mask}",
                    "duration_s": end - start,
                }
            )
    return pd.DataFrame(rows)


def build_fp_episodes(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nonzero = segments[segments["mask"] != "000"]
    for (part, phase, patient, recording, mask), group in nonzero.groupby(
        ["partition", "phase", "patient_id", "recording", "mask"], sort=False
    ):
        current_start = current_end = None
        for row in group.sort_values("start_s").itertuples(index=False):
            if current_start is None:
                current_start, current_end = float(row.start_s), float(row.end_s)
            elif float(row.start_s) <= float(current_end) + 1e-9:
                current_end = max(float(current_end), float(row.end_s))
            else:
                rows.append(
                    {
                        "partition": part,
                        "phase": phase,
                        "patient_id": patient,
                        "recording": recording,
                        "mask": mask,
                        "mask_code": f"m{mask}",
                        "start_s": current_start,
                        "end_s": current_end,
                        "duration_s": float(current_end - current_start),
                    }
                )
                current_start, current_end = float(row.start_s), float(row.end_s)
        if current_start is not None:
            rows.append(
                {
                    "partition": part,
                    "phase": phase,
                    "patient_id": patient,
                    "recording": recording,
                    "mask": mask,
                    "mask_code": f"m{mask}",
                    "start_s": current_start,
                    "end_s": current_end,
                    "duration_s": float(current_end - current_start),
                }
            )
    return pd.DataFrame(rows)


def summarize_masks(segments: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for part in PARTS:
        for phase in PHASES:
            bg = segments[(segments.partition == part) & (segments.phase == phase)]
            fp = bg[bg["mask"] != "000"]
            bg_total = float(bg.duration_s.sum())
            fp_total = float(fp.duration_s.sum())
            ep = episodes[(episodes.partition == part) & (episodes.phase == phase)]
            for scope, frame, denominator in (
                ("all_background", bg, bg_total),
                ("fp_union_only", fp, fp_total),
            ):
                for mask in MASKS:
                    sub = frame[frame["mask"] == mask]
                    ep_sub = ep[ep["mask"] == mask]
                    rows.append(
                        {
                            "partition": part,
                            "phase": phase,
                            "scope": scope,
                            "mask": mask,
                            "mask_bits": mask,
                            "mask_code": f"m{mask}",
                            "duration_s": float(sub.duration_s.sum()),
                            "fraction_of_scope": float(sub.duration_s.sum() / denominator) if denominator else 0.0,
                            "atomic_segments": int(len(sub)),
                            "fp_episodes": int(len(ep_sub)),
                            "records": int(sub.recording.nunique()),
                            "patients": int(sub.patient_id.nunique()),
                        }
                    )
    return pd.DataFrame(rows)


def build_pairwise_metrics(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_rows: list[dict[str, object]] = []
    for (part, phase, patient, recording), group in events.groupby(
        ["partition", "phase", "patient_id", "recording"], sort=False
    ):
        by_model = {
            model: [(float(row.start_s), float(row.end_s)) for row in group[group.model == model].itertuples(index=False)]
            for model in MODELS
        }
        for left, right in PAIRS:
            left_events, right_events = by_model[left], by_model[right]
            left_u, right_u = merge_intervals(left_events), merge_intervals(right_events)
            left_seconds = sum(b - a for a, b in left_u)
            right_seconds = sum(b - a for a, b in right_u)
            intersection = interval_intersection_seconds(left_u, right_u)
            union = left_seconds + right_seconds - intersection
            matched = event_match_count(left_events, right_events)
            event_union = len(left_events) + len(right_events) - matched
            record_rows.append(
                {
                    "partition": part,
                    "phase": phase,
                    "patient_id": patient,
                    "recording": recording,
                    "pair": f"{left}|{right}",
                    "left_events": len(left_events),
                    "right_events": len(right_events),
                    "matched_events": matched,
                    "event_jaccard": matched / event_union if event_union else 0.0,
                    "left_fp_seconds": left_seconds,
                    "right_fp_seconds": right_seconds,
                    "overlap_seconds": intersection,
                    "time_jaccard": intersection / union if union else 0.0,
                }
            )
    per_record = pd.DataFrame(record_rows)
    summary_rows: list[dict[str, object]] = []
    for (part, phase, pair), group in per_record.groupby(["partition", "phase", "pair"], sort=False):
        matched = int(group.matched_events.sum())
        union_events = int(group.left_events.sum() + group.right_events.sum() - matched)
        left_seconds = float(group.left_fp_seconds.sum())
        right_seconds = float(group.right_fp_seconds.sum())
        overlap_seconds = float(group.overlap_seconds.sum())
        union_seconds = left_seconds + right_seconds - overlap_seconds
        summary_rows.append(
            {
                "partition": part,
                "phase": phase,
                "pair": pair,
                "records": int(group.recording.nunique()),
                "patients": int(group.patient_id.nunique()),
                "left_events": int(group.left_events.sum()),
                "right_events": int(group.right_events.sum()),
                "matched_events": matched,
                "event_jaccard": matched / union_events if union_events else 0.0,
                "left_fp_seconds": left_seconds,
                "right_fp_seconds": right_seconds,
                "overlap_seconds": overlap_seconds,
                "time_jaccard": overlap_seconds / union_seconds if union_seconds else 0.0,
            }
        )
    return per_record, pd.DataFrame(summary_rows)


def row_oracle(row: pd.Series) -> str:
    has_seizure = bool(int(row.future_has_seizure))
    correct = [int(row[f"correct_{m}"]) for m in MODELS]
    n_correct = sum(correct)
    if n_correct == 0:
        return "unroutable" if has_seizure else "reject"
    if not has_seizure and n_correct == 3:
        return "all_safe"
    candidates = [m for m, ok in zip(MODELS, correct) if ok]
    return max(candidates, key=lambda m: (float(row[f"policy_target_{m}"]), -float(row[f"delay_s_{m}"]), m))


def entropy(values: Iterable[float]) -> float:
    arr = np.asarray([float(x) for x in values if float(x) > 0.0], dtype=float)
    return float(-(arr * np.log(arr)).sum()) if len(arr) else 0.0


def build_gate_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    pred_diag: dict[str, object] = {}
    for part in PARTS:
        path = H6_LABEL_DIR / f"{part}_causal_policy.parquet"
        labels = pd.read_parquet(path)
        if set(labels.partition.unique()) != {part}:
            raise RuntimeError(f"unexpected partition in {path}")
        labels["oracle_label"] = labels.apply(row_oracle, axis=1)
        labels["decision_needed"] = labels.outcome_group.isin(["seizure_selective", "background_selective"])
        labels["target_label"] = labels.policy_target_argmax.astype(str)
        for scope, frame in (
            ("all_rows", labels),
            ("policy_eligible", labels[labels.policy_eligible == 1]),
            ("decision_needed", labels[labels.decision_needed]),
        ):
            frame = frame.copy()
            for target_name, col in (("target_argmax", "target_label"), ("data_oracle", "oracle_label")):
                counts = frame[col].value_counts().to_dict()
                denom = len(frame)
                for model in (*MODELS, "reject", "unroutable", "all_safe"):
                    summary_rows.append(
                        {
                            "partition": part,
                            "scope": scope,
                            "label_source": target_name,
                            "label": model,
                            "rows": int(counts.get(model, 0)),
                            "share": float(counts.get(model, 0) / denom) if denom else 0.0,
                            "patients": int(frame.loc[frame[col] == model, "patient_id"].nunique()) if model in counts else 0,
                        }
                    )
            if scope != "all_rows":
                for model in MODELS:
                    share = float((frame.target_label == model).mean()) if len(frame) else 0.0
                    oracle_share = float((frame.oracle_label == model).mean()) if len(frame) else 0.0
                    summary_rows.append(
                        {
                            "partition": part,
                            "scope": scope,
                            "label_source": "target_vs_oracle",
                            "label": model,
                            "rows": int(len(frame)),
                            "share": share,
                            "oracle_share": oracle_share,
                            "patients": int(frame.patient_id.nunique()),
                        }
                    )
        decision = labels[labels.decision_needed].copy()
        for patient, frame in decision.groupby("patient_id", sort=False):
            counts = frame.target_label.value_counts(normalize=True)
            patient_rows.append(
                {
                    "partition": part,
                    "patient_id": str(patient),
                    "decision_rows": int(len(frame)),
                    "singlem_target_share": float(counts.get("singlem", 0.0)),
                    "cbramod_target_share": float(counts.get("cbramod", 0.0)),
                    "tcn_gat_target_share": float(counts.get("tcn_gat", 0.0)),
                    "target_entropy": entropy(counts.reindex(MODELS, fill_value=0.0).to_numpy()),
                    "max_target_share": float(counts.max()) if len(counts) else 0.0,
                }
            )

        if part == "dev" and H6_PRED.exists():
            pred = pd.read_parquet(H6_PRED)
            merged = labels.merge(
                pred[["sample_id", *[f"router_competence_{m}" for m in MODELS], "router_reject_probability", "router_seizure_risk"]],
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(labels):
                raise RuntimeError("dev label/prediction identity mismatch")
            comp_cols = [f"router_competence_{m}" for m in MODELS]
            merged["gate_argmax"] = merged[comp_cols].idxmax(axis=1).str.removeprefix("router_competence_")
            eval_frame = merged[merged.decision_needed & merged.oracle_label.isin(MODELS)].copy()
            rows = []
            for method, pred_col in [("constant_singlem", None), ("constant_cbramod", None), ("constant_tcn_gat", None), ("current_gate_argmax", "gate_argmax")]:
                if pred_col is None:
                    chosen = method.removeprefix("constant_")
                    pred_values = pd.Series(chosen, index=eval_frame.index)
                else:
                    pred_values = eval_frame[pred_col]
                recalls = {}
                for model in MODELS:
                    truth = eval_frame.oracle_label == model
                    recalls[model] = float((pred_values[truth] == model).mean()) if truth.any() else math.nan
                balanced = float(np.nanmean(list(recalls.values())))
                accuracy = float((pred_values.to_numpy() == eval_frame.oracle_label.to_numpy()).mean()) if len(eval_frame) else math.nan
                row = {
                    "partition": "dev",
                    "scope": "decision_needed",
                    "method": method,
                    "rows": int(len(eval_frame)),
                    "accuracy_vs_data_oracle": accuracy,
                    "balanced_accuracy_vs_data_oracle": balanced,
                    "selected_singlem_share": float((pred_values == "singlem").mean()) if len(pred_values) else math.nan,
                    "selected_cbramod_share": float((pred_values == "cbramod").mean()) if len(pred_values) else math.nan,
                    "selected_tcn_gat_share": float((pred_values == "tcn_gat").mean()) if len(pred_values) else math.nan,
                    "recall_singlem": recalls["singlem"],
                    "recall_cbramod": recalls["cbramod"],
                    "recall_tcn_gat": recalls["tcn_gat"],
                }
                if pred_col is not None:
                    target_cols = [f"policy_target_{m}" for m in MODELS]
                    max_target = eval_frame[target_cols].max(axis=1).to_numpy(float)
                    chosen_target = np.asarray(
                        [float(row[f"policy_target_{str(chosen)}"]) for _, row, chosen in zip(eval_frame.index, eval_frame.to_dict("records"), pred_values)], dtype=float
                    )
                    row["mean_routing_regret"] = float(np.mean(max_target - chosen_target)) if len(chosen_target) else math.nan
                else:
                    row["mean_routing_regret"] = math.nan
                rows.append(row)
            pred_diag["dev_gate_summary"] = rows
            pred_diag["dev_gate_full_rows"] = int(len(merged))
            pred_diag["dev_gate_decision_rows"] = int(len(eval_frame))
    return pd.DataFrame(summary_rows), pd.DataFrame(patient_rows), pred_diag


def load_existing_oracle_context() -> dict[str, object]:
    out: dict[str, object] = {}
    if H6_ORACLE.exists():
        out["h6_dev_oracle_gate"] = json.loads(H6_ORACLE.read_text())
        out["h6_dev_oracle_gate_sha256"] = sha256(H6_ORACLE)
    if H6_ORACLE_SUMMARY.exists():
        payload = json.loads(H6_ORACLE_SUMMARY.read_text())
        out["h6_oracle_status"] = payload.get("status")
        out["h6_oracle_ready_for_training"] = payload.get("ready_for_router_training")
        out["h6_oracle_manifest_sha256"] = sha256(H6_ORACLE_SUMMARY)
    return out


def main() -> None:
    if OUT.exists():
        unexpected = [p.name for p in OUT.iterdir() if p.name != "no_duplicate_overlap_preflight.json"]
        if unexpected:
            raise RuntimeError(f"refusing overwrite of {OUT}; existing files={unexpected[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = OUT / "no_duplicate_overlap_preflight.json"
    if preflight.exists():
        preflight_payload = json.loads(preflight.read_text())
        if preflight_payload.get("status") != "PASS":
            raise RuntimeError("duplicate/process preflight did not pass")
    records = load_record_metadata()
    events = load_fp_events()
    segments = build_background_segments(records, events)
    if segments.empty:
        raise RuntimeError("no train/dev background segments were produced")
    episodes = build_fp_episodes(segments)
    mask_summary = summarize_masks(segments, episodes)
    pair_record, pair_summary = build_pairwise_metrics(events)
    gate_summary, patient_summary, gate_context = build_gate_diagnostics()

    segments.to_parquet(OUT / "background_fp_mask_segments.parquet", index=False)
    episodes.to_csv(OUT / "fp_mask_episodes.csv", index=False)
    mask_summary.to_csv(OUT / "fp_mask_pattern_summary.csv", index=False)
    pair_record.to_csv(OUT / "pairwise_complementarity_by_record.csv", index=False)
    pair_summary.to_csv(OUT / "pairwise_complementarity_summary.csv", index=False)
    gate_summary.to_csv(OUT / "gate_target_vs_oracle_summary.csv", index=False)
    patient_summary.to_csv(OUT / "patient_target_distribution.csv", index=False)
    (OUT / "gate_prediction_diagnostics.json").write_text(json.dumps(gate_context, indent=2, sort_keys=True, default=float) + "\n")

    # Compact, non-deployable diagnostic numbers for the final manifest.
    fp_only = mask_summary[mask_summary.scope == "fp_union_only"]
    mixed_mask = fp_only[fp_only["mask"].str.count("1").between(1, 2)]
    all_fp = fp_only[fp_only["mask"] == "111"]
    mixed_fraction: dict[str, float] = {}
    all_three_fraction: dict[str, float] = {}
    fp_union_seconds: dict[str, float] = {}
    for (part, phase), frame in fp_only.groupby(["partition", "phase"], sort=False):
        denominator = float(frame.duration_s.sum())
        key = f"{part}_{phase}"
        fp_union_seconds[key] = denominator
        mixed_fraction[key] = float(
            frame.loc[frame["mask"].str.count("1").between(1, 2), "duration_s"].sum() / denominator
        ) if denominator else 0.0
        all_three_fraction[key] = float(
            frame.loc[frame["mask"] == "111", "duration_s"].sum() / denominator
        ) if denominator else 0.0
    diagnostic = {
        "train_dev_only": True,
        "holdout_accessed": False,
        "model_training": False,
        "threshold_changes": False,
        "records_metadata": int(len(records)),
        "fp_event_rows": int(len(events)),
        "background_segments": int(len(segments)),
        "fp_episodes": int(len(episodes)),
        "fp_union_seconds": fp_union_seconds,
        "mixed_fp_duration_fraction": mixed_fraction,
        "all_three_fp_duration_fraction": all_three_fraction,
        "event_source_sha256": sha256(EVENTS),
        "h7_manifest_sha256": sha256(H7_DIR / "manifest.json"),
        "existing_h6_oracle_context": load_existing_oracle_context(),
    }
    (OUT / "diagnostic_summary.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True, default=float) + "\n")

    manifest = {
        "release_id": NS,
        "status": "complete_train_dev_audit_no_training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "separate intrinsic expert complementarity/data prior from learned gate collapse",
        "protocol": {
            "partitions": list(PARTS),
            "patient_local_phases": list(PHASES),
            "fp_mask_order": list(MODELS),
            "background_definition": "frozen eventized detector predictions with any overlap with a truth seizure removed",
            "mask_000": "all three experts safe",
            "mask_111": "all three experts false-positive on the same interval",
            "mixed_masks": "exactly one or two experts false-positive",
            "gate_comparison": "constant-prior baselines and existing frozen H6 gate argmax on Dev decision-needed rows",
        },
        "data_discipline": {
            "official_eval_read": False,
            "patient_local_holdout_read": False,
            "new_model_training": False,
            "threshold_retuning": False,
            "raw_eeg_loaded": False,
            "source_event_filter": "partition in {train,dev}, phase in {fit,dev}, is_fp=1",
        },
        "source_hashes": {
            "frozen_expert_events": sha256(EVENTS),
            "h7_manifest": sha256(H7_DIR / "manifest.json"),
            "h6_train_labels": sha256(H6_LABEL_DIR / "train_causal_policy.parquet"),
            "h6_dev_labels": sha256(H6_LABEL_DIR / "dev_causal_policy.parquet"),
            "h6_dev_predictions": sha256(H6_PRED) if H6_PRED.exists() else None,
        },
        "preflight": preflight_payload if preflight.exists() else None,
        "outputs": {
            "background_fp_mask_segments": "background_fp_mask_segments.parquet",
            "fp_mask_episodes": "fp_mask_episodes.csv",
            "fp_mask_pattern_summary": "fp_mask_pattern_summary.csv",
            "pairwise_complementarity_by_record": "pairwise_complementarity_by_record.csv",
            "pairwise_complementarity_summary": "pairwise_complementarity_summary.csv",
            "gate_target_vs_oracle_summary": "gate_target_vs_oracle_summary.csv",
            "patient_target_distribution": "patient_target_distribution.csv",
            "gate_prediction_diagnostics": "gate_prediction_diagnostics.json",
            "diagnostic_summary": "diagnostic_summary.json",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "\n")
    print(json.dumps({"release_id": NS, "status": manifest["status"], "output": str(OUT), "diagnostic": diagnostic}, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
