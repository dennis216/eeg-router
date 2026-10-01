"""Advance frozen release state after the TUH model-inference smoke passes."""

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
    result_path = REPORT_DIR / "tuh_frozen_inference_smoke.json"
    probability_path = REPORT_DIR / "tuh_frozen_inference_smoke_probabilities.csv"
    cohort_path = REPORT_DIR / "tuh_inference_smoke_cohort.json"
    adapter_path = REPORT_DIR / "tuh_adapter_manifest.json"
    status_path = REPORT_DIR / "tuh_external_status.json"
    release_path = REPORT_DIR / "release_manifest.json"
    result = json.loads(result_path.read_text())
    if not result["all_pass"] or result["result_count"] != 6:
        raise RuntimeError("TUH frozen inference smoke is not complete")

    status = json.loads(status_path.read_text())
    status.update({
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_inference_smoke_passed_external_subset_pending",
        "model_inference_smoke": {
            "status": "passed_with_calibration_warning",
            "models": ["singlem", "cbramod", "tcn_gat"],
            "records": 2,
            "model_record_results": 6,
            "all_pass": True,
            "result_file": result_path.name,
            "result_sha256": sha256(result_path),
            "probability_file": probability_path.name,
            "probability_sha256": sha256(probability_path),
            "cohort_file": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "adapter_file": adapter_path.name,
            "adapter_sha256": sha256(adapter_path),
            "warning": "CBraMod produced one exactly saturated probability in the seizure-present smoke segment; retained for external calibration audit.",
        },
        "external_performance_reported": False,
        "model_retraining": False,
        "threshold_retuning": False,
        "next_gate": "freeze a patient-ID-hashed TUSZ eval subset and run continuous three-model external validation",
        "notes": "Real format and frozen three-model inference smokes passed. No TUH performance claim has been made.",
    })
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    release = json.loads(release_path.read_text())
    release["status"] = "h4_preliminary_negative_h5_descriptive_complete_tuh_inference_smoke_passed"
    tuh = release.setdefault("analyses", {}).setdefault("tuh_external_validation", {})
    tuh.update({
        "status": status["status"],
        "status_file": f"outputs/reports/v3-compact-extension-v1/{status_path.name}",
        "status_sha256": sha256(status_path),
        "model_inference_smoke_complete": True,
        "model_inference_smoke": f"outputs/reports/v3-compact-extension-v1/{result_path.name}",
        "model_inference_smoke_sha256": sha256(result_path),
        "external_performance_reported": False,
        "retrained": False,
        "threshold_retuned": False,
    })
    remaining = [x for x in release.get("remaining_work", []) if x != "tuh-frozen-model-inference-smoke"]
    if "tuh-frozen-external-validation" not in remaining:
        remaining.append("tuh-frozen-external-validation")
    release["remaining_work"] = remaining
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "release_status": release["status"],
        "tuh_status": status["status"],
        "inference_smoke_passed": True,
        "external_performance_reported": False,
        "next_gate": status["next_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
