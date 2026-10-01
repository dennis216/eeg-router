"""Freeze the H4 outcome as preliminary_negative without changing results."""

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
    router_path = REPORT_DIR / "h4_router_manifest.json"
    router = json.loads(router_path.read_text())
    router["status"] = "preliminary_negative"
    router["functional_success"] = False
    router["test_results_locked"] = True
    router["test_retuning_forbidden"] = True
    router["negative_result_reason"] = (
        "learned router did not meet the pre-specified FA reduction and sensitivity-loss gate consistently across the six held-out split-seed test runs; all phenotype and no-phenotype runs are retained"
    )
    router["preserved_comparisons"] = ["router_phenotype", "router_no_phenotype", "best_single", "equal_weight_average", "validation_fixed_weights"]
    router_path.write_text(json.dumps(router, indent=2, sort_keys=True) + "\n")

    gates_path = REPORT_DIR / "gates.json"
    gates = json.loads(gates_path.read_text())
    gates["outcome_status"] = "preliminary_negative"
    gates["functional_success"] = False
    gates["test_results_locked"] = True
    gates["test_retuning_forbidden"] = True
    gates["negative_result_reason"] = router["negative_result_reason"]
    gates_path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")

    freeze = {
        "release_id": "v3-compact-extension-v1",
        "analysis": "H4_functional_router",
        "status": "preliminary_negative",
        "functional_success": False,
        "branch_selected_by_validation_gate": gates["branch"],
        "test_results_locked": True,
        "test_retuning_forbidden": True,
        "preserved_methods": router["preserved_comparisons"],
        "reason": router["negative_result_reason"],
        "artifacts": {
            "gates_sha256": sha256(gates_path),
            "router_manifest_sha256": sha256(router_path),
            "test_results_sha256": sha256(REPORT_DIR / "h4_router_test_results.csv"),
            "validation_workpoints_sha256": sha256(REPORT_DIR / "h4_router_validation_workpoints.csv"),
            "patient_bootstrap_sha256": sha256(REPORT_DIR / "h4_router_patient_bootstrap.csv"),
            "waterfall_sha256": sha256(REPORT_DIR / "h4_router_waterfall.csv"),
        },
        "next_experiment": "h4-expanded-v1 requires a separately frozen train-fit design and a new holdout or TUH validation; current test remains confirmatory-locked and cannot be reused for tuning",
    }
    freeze_path = REPORT_DIR / "h4_negative_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
