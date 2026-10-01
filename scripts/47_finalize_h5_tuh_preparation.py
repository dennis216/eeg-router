"""Finalize release metadata for descriptive H5 and TUH preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    path = REPORT_DIR / "release_manifest.json"
    release = json.loads(path.read_text())
    h5_manifest = json.loads((REPORT_DIR / "h5_descriptive_transfer_manifest.json").read_text())
    tuh_status = json.loads((REPORT_DIR / "tuh_external_status.json").read_text())
    release["status"] = "h4_preliminary_negative_h5_descriptive_complete_tuh_preparation_pending_data"
    release["analyses"]["h5"] = {
        "status": "descriptive_transfer_only",
        "functional_success_claim": False,
        "manifest": "outputs/reports/v3-compact-extension-v1/h5_descriptive_transfer_manifest.json",
        "manifest_sha256": sha256(REPORT_DIR / "h5_descriptive_transfer_manifest.json"),
        "summary": "outputs/reports/v3-compact-extension-v1/h5_descriptive_transfer.csv",
        "summary_sha256": sha256(REPORT_DIR / "h5_descriptive_transfer.csv"),
        "assignments": "outputs/reports/v3-compact-extension-v1/h5_descriptive_transfer_assignments.parquet",
        "assignments_sha256": sha256(REPORT_DIR / "h5_descriptive_transfer_assignments.parquet"),
        "directions": ["SC_to_T", "ST_to_C", "CT_to_S"],
        "splits": [17, 42],
        "target_labels_used": False,
        "target_architecture_excluded_from_fit": True,
        "interpretation": "coverage, unknown rate, and distribution shift only; no functional transfer claim",
    }
    release["analyses"]["tuh_external_validation"] = {
        "status": "preparation_complete_data_missing",
        "status_file": "outputs/reports/v3-compact-extension-v1/tuh_external_status.json",
        "status_sha256": sha256(REPORT_DIR / "tuh_external_status.json"),
        "channel_mapping_smoke": "outputs/reports/v3-compact-extension-v1/tuh_channel_mapping_smoke.json",
        "channel_mapping_smoke_sha256": sha256(REPORT_DIR / "tuh_channel_mapping_smoke.json"),
        "tuh_files_found": tuh_status["tuh_available"],
        "retrained": False,
        "threshold_retuned": False,
        "external_performance_reported": False,
    }
    release["interpretation"] = [
        "preliminary",
        "compute-matched",
        "h4-preliminary-negative",
        "h4-test-results-locked",
        "h5-descriptive-transfer-only",
        "tuh-preparation-complete-but-data-missing",
        "no_functional_h5_claim",
        "no_external_performance_claim",
        "no_full_v3_completion_claim",
    ]
    path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": release["status"], "h5_status": release["analyses"]["h5"]["status"], "tuh_status": release["analyses"]["tuh_external_validation"]["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
