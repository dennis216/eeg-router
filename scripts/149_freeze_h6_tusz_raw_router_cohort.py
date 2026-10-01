"""Freeze and audit the official TUSZ Train/Dev/Eval cohort for H6.

This is a provenance step only.  It never reads expert predictions, never
trains a model, and never changes the official TUSZ partition.  Existing
partition manifests are copied by reference (not overwritten) and their
hashes plus the frozen dataset tree-inventory hash are recorded together.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MANIFEST_DIR = OUT / "manifests"
FREEZE = OUT / "cohort_freeze_manifest.json"
DATASET_INVENTORY = ROOT / "outputs/reports/v3-compact-extension-v1/tuh_tusz_v2_0_6_dataset_manifest.json"
PARTITIONS = ("train", "dev", "eval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if FREEZE.exists():
        raise RuntimeError(f"refusing to overwrite frozen cohort: {FREEZE}")
    if not DATASET_INVENTORY.exists():
        raise FileNotFoundError(DATASET_INVENTORY)
    source_inventory = json.loads(DATASET_INVENTORY.read_text())
    partition_rows = {}
    all_patients: dict[str, str] = {}
    all_records: set[str] = set()
    problems: list[str] = []
    for partition in PARTITIONS:
        path = MANIFEST_DIR / f"{partition}_manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        patients = [str(value) for value in payload.get("patients", [])]
        records = payload.get("records", [])
        if payload.get("partition") != partition:
            problems.append(f"partition field mismatch: {path}")
        if len(patients) != int(payload.get("patient_count", -1)):
            problems.append(f"patient count mismatch: {path}")
        if len(records) != int(payload.get("record_count", -1)):
            problems.append(f"record count mismatch: {path}")
        for patient in patients:
            previous = all_patients.get(patient)
            if previous is not None and previous != partition:
                problems.append(f"patient appears in multiple official partitions: {patient}")
            all_patients[patient] = partition
        for record in records:
            relative = str(record["relative_edf"])
            if relative in all_records:
                problems.append(f"duplicate EDF across partitions: {relative}")
            all_records.add(relative)
            if str(record["patient_id"]) not in patients:
                problems.append(f"record patient missing from patient list: {relative}")
        partition_rows[partition] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "patient_count": len(patients),
            "record_count": len(records),
            "decision_count": int(sum(int(r.get("decision_count", 0)) for r in records)),
            "edf_hash_types": sorted({str(r.get("edf_hash_type", "sha256")) for r in records}),
        }
    if problems:
        raise RuntimeError("; ".join(problems))
    payload = {
        "release_id": NAMESPACE,
        "status": "frozen",
        "dataset": "TUSZ_v2.0.6",
        "dataset_root": "/mnt/d/TUH_EEG/TUSZ_v2.0.6",
        "official_partitions": partition_rows,
        "total_patients": len(all_patients),
        "total_records": len(all_records),
        "patient_disjoint": True,
        "record_disjoint": True,
        "source_dataset_inventory": str(DATASET_INVENTORY.relative_to(ROOT)),
        "source_dataset_inventory_sha256": sha256(DATASET_INVENTORY),
        "source_tree_inventory_sha256": source_inventory.get("tree_inventory_sha256"),
        "raw_input_contract": {
            "sampling_hz": 128,
            "channels": 16,
            "context_seconds": 60,
            "context_shape": [16, 7680],
            "future_horizon_seconds": 60,
            "router_excludes_expert_outputs": True,
        },
        "edf_hash_policy": "existing partition files retain their recorded hash policy; fast manifests use tree inventory + size/mtime fingerprints",
        "selection_used_model_output": False,
        "selection_used_test_performance": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    FREEZE.with_suffix(".sha256").write_text(sha256(FREEZE) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
