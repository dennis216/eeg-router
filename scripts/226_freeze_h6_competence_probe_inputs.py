"""Freeze the input inventory for the H6 competence probe.

This stage is intentionally read-only with respect to the frozen H6 sources.
It audits the existing 10-s -> future-60-s labels as a secondary diagnostic
and records that new same-window labels are required for the primary probe.
TUSZ Eval is never read.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
NAMESPACE = "h6-eeg-conditionable-competence-probe-v1"
OUT = ROOT / "outputs" / "reports" / NAMESPACE
SOURCE = ROOT / "outputs/reports/h6-causal-policy-router-v2-corrected/labels"
RAW_MANIFEST = ROOT / "outputs/reports/h6-causal-policy-router-v2/raw_cache_10s_manifest.json"
EXPERT_MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/train_manifest.json"
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partition_summary(frame: pd.DataFrame) -> dict[str, object]:
    correctness = frame[[f"correct_{model}" for model in MODELS]].astype(int)
    mask = correctness.astype(str).agg("".join, axis=1)
    seizure = frame["future_has_seizure"].astype(int).eq(1)
    actionable = correctness.sum(axis=1).between(1, 2)
    per_patient = frame.groupby("patient_id", observed=True).size()
    return {
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "records": int(frame["recording"].nunique()),
        "mask_counts": {str(key): int(value) for key, value in mask.value_counts().sort_index().items()},
        "mask_counts_background": {
            str(key): int(value) for key, value in mask[~seizure].value_counts().sort_index().items()
        },
        "mask_counts_seizure": {
            str(key): int(value) for key, value in mask[seizure].value_counts().sort_index().items()
        },
        "correct_prevalence": {
            model: float(frame[f"correct_{model}"].astype(float).mean()) for model in MODELS
        },
        "seizure_rows": int(seizure.sum()),
        "background_rows": int((~seizure).sum()),
        "actionable_rows": int(actionable.sum()),
        "all_correct_rows": int((correctness.sum(axis=1) == 3).sum()),
        "all_wrong_rows": int((correctness.sum(axis=1) == 0).sum()),
        "rows_per_patient": {
            "min": int(per_patient.min()),
            "median": float(per_patient.median()),
            "max": int(per_patient.max()),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "input_inventory.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite frozen inventory: {target}")

    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    summaries: dict[str, dict[str, object]] = {}
    for partition in ("train", "dev"):
        path = SOURCE / f"{partition}_causal_policy.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        required = {
            "patient_id", "recording", "context_start_s", "context_end_s",
            "future_start_s", "future_end_s", "future_has_seizure",
            "raw_cache_path_v2", *(f"correct_{model}" for model in MODELS),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{partition} missing columns: {missing}")
        frames[partition] = frame
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        summaries[partition] = partition_summary(frame)

    train_patients = set(frames["train"]["patient_id"].astype(str))
    dev_patients = set(frames["dev"]["patient_id"].astype(str))
    overlap = sorted(train_patients & dev_patients)
    context_lengths = sorted(
        set((frames["train"]["context_end_s"] - frames["train"]["context_start_s"]).round(6))
        | set((frames["dev"]["context_end_s"] - frames["dev"]["context_start_s"]).round(6))
    )
    horizon_lengths = sorted(
        set((frames["train"]["future_end_s"] - frames["train"]["future_start_s"]).round(6))
        | set((frames["dev"]["future_end_s"] - frames["dev"]["future_start_s"]).round(6))
    )

    for path in (RAW_MANIFEST, EXPERT_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    protocol = ROOT / "protocols/h6-eeg-conditionable-competence-probe-v1.md"
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    source_hashes[str(protocol.relative_to(ROOT))] = sha256(protocol)

    payload = {
        "release_id": NAMESPACE,
        "stage": "input_inventory",
        "created_utc": utc_now(),
        "status": "PASS_SECONDARY_LABEL_AUDIT_PRIMARY_LABEL_BUILD_REQUIRED" if not overlap else "FAIL_PATIENT_OVERLAP",
        "eval_read": False,
        "model_training": False,
        "patient_overlap_train_dev": overlap,
        "existing_label_contract": {
            "context_lengths_seconds": context_lengths,
            "target_horizon_lengths_seconds": horizon_lengths,
            "interpretation": "existing labels are a 10-s causal context to future-60-s competence forecast",
            "allowed_role": "secondary diagnosis of the existing H6-v2 policy",
            "disallowed_role": "primary same-window competence evidence",
        },
        "primary_probe_requirement": (
            "Build new same-window competence labels from the frozen expert probabilities, "
            "frozen event rules, and the exact completed 10-s EEG decision window."
        ),
        "partitions": summaries,
        "source_hashes": source_hashes,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "input_inventory.sha256").write_text(f"{sha256(target)}  input_inventory.json\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
