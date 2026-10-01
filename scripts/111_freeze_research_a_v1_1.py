"""Freeze the completed Research A, baseline, and HMM evidence release.

This creates a new, additive release index and evidence snapshot.  It never
overwrites the earlier Research A release or any run output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
OUT = ROOT / "outputs" / "reports" / "research-a-release-v1.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_record(path: Path) -> dict:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def count_fdr_significant(path: Path) -> tuple[int, int]:
    total = significant = undefined = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            q = row.get("q_bh", "")
            if not q or q.lower() == "nan":
                undefined += 1
            elif float(q) < 0.05:
                significant += 1
    return total, significant, undefined


def main() -> None:
    primary = ROOT / "outputs/reports/research-a-release-v1/research_a_freeze_manifest.json"
    primary_audit = ROOT / "runs/v3-groupkfold-confirmatory-v1/groupkfold_confirmatory_audit.json"
    p1 = ROOT / "outputs/reports/v3-groupkfold-baselines-v1"
    p2 = ROOT / "outputs/reports/h2-dynamic-confirmatory-v1"
    p3 = ROOT / "outputs/reports/tuar-tuev-status-v1/manifest.json"

    required = [
        primary,
        primary_audit,
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/oof_release_manifest.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h1_groupkfold_summary.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h2_h3_groupkfold_summary.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h2_stability_manifest.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h4_groupkfold_gates.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h5_groupkfold_descriptive_transfer_manifest.json",
        ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/tusz_groupkfold_external_v1_scoring_manifest.json",
        p1 / "baseline_audit_manifest.json",
        p1 / "baseline_run_metrics.csv",
        p1 / "baseline_summary.csv",
        p1 / "no_duplicate_overlap_preflight.json",
        p2 / "manifest.json",
        p2 / "hmm_persistence_null_test.csv",
        p2 / "hmm_dynamic_model_association_fdr.csv",
        p2 / "hmm_persistence_patient_values.csv",
        p3,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing freeze evidence: " + ", ".join(missing))

    primary_audit_data = json.loads(primary_audit.read_text())
    p1_data = json.loads((p1 / "baseline_audit_manifest.json").read_text())
    p2_data = json.loads((p2 / "manifest.json").read_text())
    p3_data = json.loads(p3.read_text())
    if primary_audit_data.get("matrix_complete") is not True or primary_audit_data.get("completed_runs") != 45:
        raise RuntimeError("primary 45-run matrix is not complete")
    if p1_data.get("matrix_complete") is not True or p1_data.get("completed_runs") != 60:
        raise RuntimeError("standard baseline 60-run matrix is not audit-complete")
    if p2_data.get("status") != "frozen_assignment_audit_complete" or p2_data.get("refit") is not False:
        raise RuntimeError("HMM confirmatory audit is not frozen and read-only")
    if p3_data.get("model_training") is not False or p3_data.get("scientific_labels_created") is not False:
        raise RuntimeError("TUAR/TUEV status audit is not status-only")

    association_path = p2 / "hmm_dynamic_model_association_fdr.csv"
    association_rows, fdr_significant, undefined_association_rows = count_fdr_significant(association_path)
    records = [file_record(path) for path in required]

    code_root = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts")
    code_files = [
        code_root / "61_run_baseline.py",
        code_root / "64_run_conv_baseline.py",
        code_root / "106_audit_standard_groupkfold_baselines.py",
        code_root / "107_run_h2_dynamic_confirmatory_audit.py",
        code_root / "110_preflight_no_duplicate_overlap.py",
        code_root / "111_freeze_research_a_v1_1.py",
    ]
    code_records = [file_record(path) for path in code_files if path.exists()]

    release = {
        "release_id": "research-a-release-v1.1",
        "parent_release": "research-a-release-v1",
        "status": "frozen_for_manuscript_and_follow_up_experiments",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_namespace": "v3-groupkfold-confirmatory-v1",
        "baseline_namespace": "v3-groupkfold-baselines-v1",
        "hmm_namespace": "h2-dynamic-confirmatory-v1",
        "primary_claim": "false alarms are partly temporally shared and architecture-dependent",
        "standard_split": "sklearn GroupKFold outer + patient GroupShuffleSplit inner",
        "group_unit": "canonical patient_id",
        "primary_runs": 45,
        "baseline_runs": 60,
        "test_evaluation_policy": "one evaluation per run; threshold validation-only",
        "old_compact_results_status": "preserved as historical and excluded from confirmatory ranking",
        "p1": {
            "audit_status": "PASS",
            "undefined_median_delay_runs": int(p1_data.get("undefined_median_delay_runs", 0)),
        },
        "p2": {
            "audit_status": p2_data.get("status"),
            "refit": False,
            "outer_test_used_for_selection": False,
            "test_patients": p2_data.get("test_patients"),
            "test_rows": p2_data.get("test_rows"),
            "patient_bootstrap_replicates": p2_data.get("patient_bootstrap_replicates"),
            "signflip_permutations": p2_data.get("signflip_permutations"),
            "association_rows": association_rows,
            "fdr_significant_rows": fdr_significant,
            "undefined_association_rows": undefined_association_rows,
        },
        "p3": {
            "status": p3_data.get("status"),
            "model_training": False,
            "scientific_labels_created": False,
        },
        "claim_boundaries": [
            "H2 static phenotype stability remains exploratory.",
            "HMM states are latent statistical states, not artifact labels.",
            "H4 has not demonstrated a reliable functional router gain.",
            "H5 is descriptive transfer only, not functional transfer success.",
            "TUAR/TUEV status audit is not an ontology result.",
        ],
        "records": records,
        "code_snapshot": code_records,
        "next_namespace_policy": "new artifact ontology, H6, or Siena TTT work must use a new namespace and cannot overwrite this release",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    evidence = OUT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for path in required:
        destination = evidence / rel(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (OUT / "research_a_release_v1_1_manifest.json").write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    (OUT / "README.txt").write_text(
        "Research A release v1.1 is frozen. Evidence is copied under evidence/ and indexed by SHA256.\n"
        "Do not overwrite this release. New artifact ontology, H6, or Siena TTT work requires a new namespace.\n"
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
