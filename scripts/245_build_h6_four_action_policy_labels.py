"""Build a single four-action H6 policy label dataset.

The four actions are ``singlem``, ``cbramod``, ``tcn_gat`` and ``reject``.
There is deliberately no independent risk target or risk head in this label
contract.  ``future_has_seizure`` is retained only as provenance/audit
metadata and for later safety-stratified sampling.

The input remains a causal 10-second EEG context.  Labels are derived from
the frozen expert outcomes on the following 60-second horizon.  Existing H6
outputs are never overwritten; this script writes a new release namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_NS = "h6-tusz-raw-pre-router-v1"
NS = "h6-causal-policy-router-v3-four-action"
SOURCE_OUT = ROOT / "outputs/reports" / SOURCE_NS
OUT = ROOT / "outputs/reports" / NS
MODELS = ("singlem", "cbramod", "tcn_gat")
ACTIONS = MODELS + ("reject",)
PARTITIONS = ("train", "dev")
TAU_OUTCOME = 0.25
TAU_CONF = 0.25
CONTEXT_SECONDS = 10.0
FUTURE_SECONDS = 60.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray, tau: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.ones(1, dtype=float)
    scaled = values / tau
    scaled -= float(np.nanmax(scaled))
    exponent = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(exponent.sum())
    return exponent / total if total > 0 else np.full(values.size, 1.0 / values.size)


def classify_group(row: pd.Series, correct: np.ndarray) -> str:
    has_seizure = bool(int(row.future_has_seizure))
    correct_count = int(correct.sum())
    if has_seizure:
        if correct_count == 0:
            return "unroutable_seizure"
        if correct_count == len(MODELS):
            return "seizure_all_correct"
        return "seizure_selective"

    mask = str(row.correct_mask)
    if mask == "000":
        return "background_all_fp"
    if mask == "111":
        return "background_all_safe"
    return "background_selective"


def expert_soft_target(row: pd.Series, correct: np.ndarray, has_seizure: bool) -> np.ndarray:
    """Return a target over the three experts for a routable row."""
    if not int(correct.sum()):
        return np.zeros(len(MODELS), dtype=float)
    indices = np.flatnonzero(correct == 1)
    if has_seizure:
        utility = np.asarray(
            [
                1.0
                - min(max(float(row[f"delay_s_{model}"]), 0.0), FUTURE_SECONDS)
                / FUTURE_SECONDS
                for model in MODELS
            ],
            dtype=float,
        )[indices]
    else:
        utility = np.ones(len(indices), dtype=float)
    confidence = np.asarray([float(row[f"confidence_{model}"]) for model in MODELS], dtype=float)[indices]
    target = np.zeros(len(MODELS), dtype=float)
    target[indices] = 0.90 * softmax(utility, TAU_OUTCOME) + 0.10 * softmax(confidence, TAU_CONF)
    target /= float(target.sum())
    return target


def action_target(row: pd.Series) -> tuple[str, str, bool, np.ndarray]:
    correct = np.asarray([int(row[f"correct_{model}"]) for model in MODELS], dtype=int)
    has_seizure = bool(int(row.future_has_seizure))
    group = classify_group(row, correct)

    target = np.zeros(len(ACTIONS), dtype=float)
    if group == "background_all_fp":
        target[-1] = 1.0
        return group, "reject", True, target
    if group == "unroutable_seizure":
        # No expert is correct, and rejecting a true seizure is unsafe.  Keep
        # this row for audit/safety analysis but do not invent a policy action.
        return group, "unroutable", False, target
    if group == "background_all_safe":
        # All experts are correct; there is no preferred expert, but Reject is
        # explicitly not a valid target.  A uniform soft target is neutral.
        target[: len(MODELS)] = 1.0 / len(MODELS)
        return group, "expert_uniform_all_safe", True, target

    target[: len(MODELS)] = expert_soft_target(row, correct, has_seizure)
    return group, "expert_soft", True, target


def build_partition(partition: str, source_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_parquet(source_path)
    if source.empty or set(source.partition.astype(str)) != {partition}:
        raise RuntimeError(f"bad source {source_path}")

    required = {
        "sample_id",
        "partition",
        "patient_id",
        "recording",
        "cache_path",
        "cache_row",
        "future_start_s",
        "future_end_s",
        "future_has_seizure",
        "correct_mask",
    }
    required |= {
        f"{field}_{model}"
        for field in ("correct", "confidence", "delay_s", "event_count", "threshold", "probability_sha256")
        for model in MODELS
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for record in source.itertuples(index=False):
        row = pd.Series(record._asdict())
        group, target_kind, eligible, target = action_target(row)
        action = ACTIONS[int(np.argmax(target))] if eligible else "unroutable"
        decision_time = float(record.future_start_s)
        output: dict[str, object] = {
            "sample_id": f"{partition}|{record.patient_id}|{record.recording}|{int(record.cache_row)}|causal10|four_action",
            "source_sample_id_v1": str(record.sample_id),
            "partition": partition,
            "patient_id": str(record.patient_id),
            "recording": str(record.recording),
            "decision_index_v1": int(record.cache_row),
            "decision_time_s": decision_time,
            "context_start_s": decision_time - CONTEXT_SECONDS,
            "context_end_s": decision_time,
            "future_start_s": decision_time,
            "future_end_s": float(record.future_end_s),
            # Metadata used to audit and stratify the policy dataset.  This is
            # not emitted as risk_label and is not an independent risk target.
            "future_has_seizure": int(bool(record.future_has_seizure)),
            "raw_cache_path_v2": str(Path("raw_cache_10s") / partition / Path(str(record.recording)).with_suffix(".npz")),
            "source_cache_path_v1": str(record.cache_path),
            "correct_mask": str(record.correct_mask),
            "correct_count": int(sum(int(row[f"correct_{model}"]) for model in MODELS)),
            "outcome_group": group,
            "policy_action_target_kind": target_kind,
            "policy_action_eligible": int(eligible),
            "policy_action": action,
            "policy_action_id": int(np.argmax(target)) if eligible else -1,
            "policy_target_entropy": float(-np.sum(target[target > 0] * np.log(target[target > 0]))) if eligible else 0.0,
        }
        for index, action_name in enumerate(ACTIONS):
            output[f"policy_target_{action_name}"] = float(target[index])
        for model in MODELS:
            output.update(
                {
                    f"correct_{model}": int(row[f"correct_{model}"]),
                    f"confidence_{model}": float(row[f"confidence_{model}"]),
                    f"delay_s_{model}": float(row[f"delay_s_{model}"]),
                    f"event_count_{model}": int(row[f"event_count_{model}"]),
                    f"threshold_{model}": float(row[f"threshold_{model}"]),
                    f"probability_sha256_{model}": str(row[f"probability_sha256_{model}"]),
                }
            )
        rows.append(output)

    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        raise RuntimeError(f"duplicate ids in {partition}")
    if "risk_label" in frame.columns:
        raise RuntimeError("risk_label must not be present in four-action dataset")

    target_columns = [f"policy_target_{action}" for action in ACTIONS]
    target_sums = frame[target_columns].sum(axis=1).to_numpy()
    eligible = frame.policy_action_eligible.astype(bool).to_numpy()
    if not np.allclose(target_sums[eligible], 1.0, atol=1e-6):
        raise RuntimeError(f"eligible target rows do not sum to one in {partition}")
    if not np.allclose(target_sums[~eligible], 0.0, atol=1e-6):
        raise RuntimeError(f"ineligible target rows are not masked in {partition}")
    if not np.all(frame.loc[frame.outcome_group.eq("background_all_fp"), "policy_action"].eq("reject")):
        raise RuntimeError("all-FP background must target reject")
    if float(frame.loc[frame.future_has_seizure.eq(1), "policy_target_reject"].sum()) != 0.0:
        raise RuntimeError("seizure rows must never have Reject target mass")

    eligible_frame = frame.loc[eligible]
    action_counts = frame.policy_action.value_counts().reindex(ACTIONS + ("unroutable",), fill_value=0).astype(int)
    group_counts = frame.outcome_group.value_counts().sort_index().astype(int)
    action_mass = eligible_frame[target_columns].sum().to_dict() if len(eligible_frame) else {x: 0.0 for x in target_columns}
    audit = {
        "partition": partition,
        "source_label": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_label_sha256": sha256(source_path),
        "rows": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "records": int(frame.recording.nunique()),
        "context_contract": {
            "seconds": CONTEXT_SECONDS,
            "sampling_hz": 128,
            "channels": 16,
            "samples": 1280,
            "context_end": "decision_time_s",
        },
        "future_horizon_seconds": FUTURE_SECONDS,
        "action_order": list(ACTIONS),
        "risk_head": False,
        "risk_label_emitted": False,
        "future_has_seizure_role": "audit_and_safety_stratification_metadata_only",
        "outcome_group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "policy_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "policy_action_mass": {str(k): float(action_mass.get(f"policy_target_{k}", 0.0)) for k in ACTIONS},
        "policy_action_eligible_rows": int(frame.policy_action_eligible.sum()),
        "unroutable_seizure_rows": int(frame.outcome_group.eq("unroutable_seizure").sum()),
        "background_all_fp_rows": int(frame.outcome_group.eq("background_all_fp").sum()),
        "seizure_rows": int(frame.future_has_seizure.sum()),
        "background_rows": int((frame.future_has_seizure == 0).sum()),
        "reject_target_mass_on_seizure": float(frame.loc[frame.future_has_seizure.eq(1), "policy_target_reject"].sum()),
        "eval_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return frame, audit


def write_distribution_files(frames: dict[str, pd.DataFrame], audits: dict[str, dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for partition, frame in frames.items():
        total = len(frame)
        counts = frame.policy_action.value_counts().to_dict()
        for action in ACTIONS + ("unroutable",):
            count = int(counts.get(action, 0))
            rows.append({"partition": partition, "target": action, "count": count, "percent_of_all_rows": 100.0 * count / total})
        table = pd.crosstab(frame.outcome_group, frame.policy_action).reindex(columns=ACTIONS + ("unroutable",), fill_value=0)
        for group, values in table.iterrows():
            group_total = int(values.sum())
            for action in ACTIONS + ("unroutable",):
                count = int(values.get(action, 0))
                group_rows.append(
                    {
                        "partition": partition,
                        "outcome_group": str(group),
                        "target": action,
                        "count": count,
                        "percent_within_outcome_group": 100.0 * count / group_total if group_total else 0.0,
                    }
                )

    pd.DataFrame(rows).to_csv(OUT / "policy_action_distribution.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUT / "policy_action_by_outcome_group.csv", index=False)

    mass_rows = []
    for partition, audit in audits.items():
        for action in ACTIONS:
            mass_rows.append({"partition": partition, "target": action, "soft_target_mass": float(audit["policy_action_mass"][action])})
    pd.DataFrame(mass_rows).to_csv(OUT / "policy_action_soft_mass.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", default="train,dev")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_OUT / "labels")
    args = parser.parse_args()
    partitions = tuple(part.strip() for part in args.partitions.split(",") if part.strip())
    if any(partition not in PARTITIONS for partition in partitions):
        raise ValueError(partitions)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing overwrite of non-empty output {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    label_dir = OUT / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    audits: dict[str, dict[str, object]] = {}
    source_hashes: dict[str, str] = {}
    for partition in partitions:
        source_path = (args.source_dir / f"{partition}_formal_fold0_seed17.parquet").resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        frame, audit = build_partition(partition, source_path)
        output_path = label_dir / f"{partition}_four_action_policy.parquet"
        audit_path = label_dir / f"{partition}_four_action_policy_audit.json"
        frame.to_parquet(output_path, index=False)
        output_path.with_suffix(".sha256").write_text(sha256(output_path) + "\n")
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        audit_path.with_suffix(".sha256").write_text(sha256(audit_path) + "\n")
        frames[partition] = frame
        audits[partition] = audit
        source_hashes[partition] = sha256(source_path)

    write_distribution_files(frames, audits)
    manifest = {
        "release_id": NS,
        "source_release_id": SOURCE_NS,
        "dataset": "TUSZ_v2.0.6 official train/dev",
        "status": "four_action_policy_labels_only_pre_sampling",
        "partitions": list(partitions),
        "labels": {partition: f"outputs/reports/{NS}/labels/{partition}_four_action_policy.parquet" for partition in partitions},
        "audits": {partition: f"outputs/reports/{NS}/labels/{partition}_four_action_policy_audit.json" for partition in partitions},
        "distribution_files": [
            f"outputs/reports/{NS}/policy_action_distribution.csv",
            f"outputs/reports/{NS}/policy_action_by_outcome_group.csv",
            f"outputs/reports/{NS}/policy_action_soft_mass.csv",
        ],
        "source_labels": {partition: f"outputs/reports/{SOURCE_NS}/labels/{partition}_formal_fold0_seed17.parquet" for partition in partitions},
        "source_label_sha256": source_hashes,
        "decision_contract": {
            "input": "raw EEG only",
            "channels": 16,
            "sampling_hz": 128,
            "context_seconds": CONTEXT_SECONDS,
            "context_samples": 1280,
            "future_horizon_seconds": FUTURE_SECONDS,
            "expert_outputs_in_router_input": False,
        },
        "label_contract": {
            "policy_actions": list(ACTIONS),
            "background_all_fp": "one_hot_reject",
            "background_selective": "soft target over correct experts",
            "background_all_safe": "uniform soft target over experts; no reject mass",
            "seizure_selective": "soft target over correct experts",
            "seizure_all_correct": "soft target over all experts",
            "unroutable_seizure": "policy masked; never mapped to reject",
            "target_components": "90% future outcome utility + 10% confidence tie-breaker for expert targets",
            "future_has_seizure_not_a_risk_head": True,
        },
        "risk_head": False,
        "eval_read": False,
        "old_results_preserved": True,
        "audits": audits,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.with_suffix(".sha256").write_text(sha256(manifest_path) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nPOLICY ACTION DISTRIBUTION")
    print(pd.read_csv(OUT / "policy_action_distribution.csv").to_string(index=False))
    print("\nPOLICY ACTION BY OUTCOME GROUP")
    print(pd.read_csv(OUT / "policy_action_by_outcome_group.csv").to_string(index=False))


if __name__ == "__main__":
    main()
