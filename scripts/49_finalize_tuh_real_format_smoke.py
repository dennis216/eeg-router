"""Update TUH and release status after the frozen real-data format smoke."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    smoke_path = REPORT_DIR / "tuh_real_format_smoke.json"
    dataset_path = REPORT_DIR / "tuh_tusz_v2_0_6_dataset_manifest.json"
    records_path = REPORT_DIR / "tuh_real_format_smoke_records.csv"
    status_path = REPORT_DIR / "tuh_external_status.json"
    release_path = REPORT_DIR / "release_manifest.json"

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not smoke["all_pass"]:
        raise RuntimeError("real format smoke did not pass; release status must not advance")

    old_status = json.loads(status_path.read_text(encoding="utf-8"))
    status = {
        **old_status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6",
        "dataset_root": dataset["dataset_root"],
        "dataset_root_wsl": dataset.get("dataset_root_wsl"),
        "status": "real_format_smoke_passed_model_inference_smoke_pending",
        "tuh_available": True,
        "existing_candidate_roots": [{
            "path": dataset["dataset_root"],
            "edf_files": dataset["edf_files"],
            "edf_gib": dataset["edf_gib"],
        }],
        "dataset_inventory": {
            "manifest": dataset_path.name,
            "manifest_sha256": sha256(dataset_path),
            "tree_inventory_sha256": dataset["tree_inventory_sha256"],
            "edf_files": dataset["edf_files"],
            "patients_by_partition": dataset["patients_by_partition"],
            "download_completeness_claimed": dataset["download_completeness_claimed"],
        },
        "real_format_smoke": {
            "status": "passed_with_signal_quality_warnings" if smoke["records_with_warnings"] else "passed",
            "selected_patients": smoke["selected_patients"],
            "selected_records": smoke["selected_records"],
            "records_passed": smoke["records_passed"],
            "records_failed": smoke["records_failed"],
            "records_with_warnings": smoke["records_with_warnings"],
            "all_pass": smoke["all_pass"],
            "smoke_file": smoke_path.name,
            "smoke_sha256": sha256(smoke_path),
            "records_file": records_path.name,
            "records_sha256": sha256(records_path),
            "warnings_interpretation": "Three selected records contain >1000 uV excursions; these are signal-QC warnings, not EDF/annotation/channel-format failures.",
        },
        "construct_validation": "pending_frozen_model_inference_and_construct_mapping",
        "model_retraining": False,
        "threshold_retuning": False,
        "external_performance_reported": False,
        "next_gate": "freeze TUH real-data adapter, then run 1-2 eval-record three-model inference smoke",
        "notes": "TUSZ files are present and the deterministic nine-patient real format smoke passed. No external performance number is reported yet.",
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["status"] = "h4_preliminary_negative_h5_descriptive_complete_tuh_format_smoke_passed"
    release.setdefault("analyses", {})["tuh_external_validation"] = {
        "status": status["status"],
        "status_file": f"outputs/reports/v3-compact-extension-v1/{status_path.name}",
        "status_sha256": sha256(status_path),
        "dataset_manifest": f"outputs/reports/v3-compact-extension-v1/{dataset_path.name}",
        "dataset_manifest_sha256": sha256(dataset_path),
        "real_format_smoke": f"outputs/reports/v3-compact-extension-v1/{smoke_path.name}",
        "real_format_smoke_sha256": sha256(smoke_path),
        "real_format_smoke_passed": True,
        "signal_quality_warning_records": smoke["records_with_warnings"],
        "model_inference_smoke_complete": False,
        "external_performance_reported": False,
        "retrained": False,
        "threshold_retuned": False,
        "tuh_files_found": True,
    }
    remaining = [x for x in release.get("remaining_work", []) if x != "tuh-preparation-complete-but-data-missing"]
    for item in ("tuh-frozen-model-inference-smoke", "tuh-frozen-external-validation"):
        if item not in remaining:
            remaining.append(item)
    release["remaining_work"] = remaining
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "release_status": release["status"],
        "tuh_status": status["status"],
        "format_smoke_passed": True,
        "warning_records": smoke["records_with_warnings"],
        "next_gate": status["next_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
