"""Create an immutable audit index for the Research A release.

This is an administrative freeze artifact.  It does not rerun models, access
test data, or modify any prior result.  It records the standard GroupKFold
release files and their hashes so subsequent baseline/HMM work cannot be
mistaken for a change to the main study.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("BFA_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs" / "reports" / "research-a-release-v1"
STANDARD = ROOT / "outputs" / "reports" / "v3-groupkfold-confirmatory-v1"
AUDIT = ROOT / "runs" / "v3-groupkfold-confirmatory-v1" / "groupkfold_confirmatory_audit.json"

KEY_FILES = [
    AUDIT,
    STANDARD / "oof_release_manifest.json",
    STANDARD / "h1_groupkfold_summary.json",
    STANDARD / "h2_h3_groupkfold_summary.json",
    STANDARD / "h2_stability_manifest.json",
    STANDARD / "h4_groupkfold_gates.json",
    STANDARD / "h4_groupkfold_router_manifest.json",
    STANDARD / "h4_groupkfold_router_test.csv",
    STANDARD / "h5_groupkfold_descriptive_transfer_manifest.json",
    STANDARD / "tusz_groupkfold_external_v1_scoring_manifest.json",
    STANDARD / "tusz_groupkfold_external_validation_report_manifest.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)
    audit = json.loads(AUDIT.read_text())
    if audit.get("matrix_complete") is not True or audit.get("completed_runs") != 45:
        raise RuntimeError("Research A freeze requires a complete 45-run standard matrix")
    missing = [rel(path) for path in KEY_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(missing))

    records = []
    for path in KEY_FILES:
        records.append({"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    release = {
        "release_id": "research-a-release-v1",
        "status": "frozen_for_manuscript_and_baseline_addendum",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_namespace": "v3-groupkfold-confirmatory-v1",
        "primary_claim": "false alarms are partly temporally shared and architecture-dependent",
        "standard_split": "sklearn GroupKFold outer + patient GroupShuffleSplit inner",
        "group_unit": "canonical patient_id",
        "primary_runs": 45,
        "test_evaluation_policy": "one evaluation per run; threshold validation-only",
        "old_compact_results_status": "preserved_as_historical_and_not_used_for_primary_ranking",
        "tusz_status": "frozen_external_inference_and_scoring",
        "records": records,
        "next_namespace_policy": "all new baseline/HMM work must use a new namespace and cannot overwrite this release",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "research_a_freeze_manifest.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "README.txt").write_text(
        "Research A is frozen. New baseline/HMM experiments must use separate namespaces.\n"
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
