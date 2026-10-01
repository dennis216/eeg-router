"""Freeze the full TUSZ eval confirmation cohort and its patient-level audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
PREFIX = "tusz_eval_full_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rel(name: str) -> str:
    return f"outputs/reports/v3-compact-extension-v1/{name}"


def main() -> None:
    cohort_path = REPORT_DIR / f"{PREFIX}_manifest.json"
    results_path = REPORT_DIR / f"{PREFIX}_results_manifest.json"
    audit_path = REPORT_DIR / f"{PREFIX}_audit_manifest.json"
    results_csv = REPORT_DIR / f"{PREFIX}_results.csv"
    qc_csv = REPORT_DIR / f"{PREFIX}_signal_qc.csv"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not results.get("full_tusz_eval") or results.get("status") != "preliminary_frozen_external_eval_complete":
        raise RuntimeError("full TUSZ results are not complete and frozen")
    if audit.get("status") != "patient_waterfall_and_calibration_audit_complete":
        raise RuntimeError("patient-level audit is not complete")

    qc = pd.read_csv(qc_csv)
    qc_summary = {
        "records": int(len(qc)),
        "p99_abs_gt_1000_uv_records": int((qc["p99_abs_uv"] > 1000).sum()),
        "max_abs_gt_1000_uv_records": int((qc["max_abs_uv"] > 1000).sum()),
        "p99_abs_uv_median": float(qc["p99_abs_uv"].median()),
        "p99_abs_uv_p95": float(qc["p99_abs_uv"].quantile(0.95)),
        "p99_abs_uv_p99": float(qc["p99_abs_uv"].quantile(0.99)),
        "max_abs_uv_p99": float(qc["max_abs_uv"].quantile(0.99)),
        "interpretation": "signal-QC warnings are retained and are not used to delete records or retune thresholds",
    }

    audit["decision_notes"] = {
        "external_subset_is_not_full_eval": True,
        "full_eval_expansion_completed": True,
        "next_gate": "optional frozen construct validation on TUAR, TUEV or TUSL; no TUSZ threshold tuning",
    }
    audit["full_eval"] = {
        "cohort_manifest_sha256": sha256(cohort_path),
        "results_manifest_sha256": sha256(results_path),
        "results_csv_sha256": sha256(results_csv),
        "signal_qc_summary": qc_summary,
    }
    audit["status"] = "full_eval_patient_waterfall_and_calibration_audit_complete"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status_path = REPORT_DIR / "tuh_external_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "preliminary_frozen_external_full_eval_complete"
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    status["construct_validation"] = "pending_frozen_model_inference_and_construct_mapping"
    status["external_full_validation"] = {
        "cohort_id": cohort["cohort_id"],
        "patients": cohort["patient_count"],
        "records": cohort["record_count"],
        "duration_h": sum(row["duration_s"] for row in cohort["records"]) / 3600.0,
        "raw_annotation_events": sum(len(row["seizure_events"]) for row in cohort["records"]),
        "manifest": cohort_path.name,
        "manifest_sha256": sha256(cohort_path),
        "results_manifest": results_path.name,
        "results_manifest_sha256": sha256(results_path),
        "audit_manifest": audit_path.name,
        "audit_manifest_sha256": sha256(audit_path),
        "full_tusz_eval": True,
        "status": results["status"],
        "model_training": False,
        "threshold_retuning": False,
        "signal_qc_summary": qc_summary,
    }
    status["external_performance_reported"] = True
    status["model_retraining"] = False
    status["threshold_retuning"] = False
    status["next_gate"] = "optional frozen construct validation on TUAR, TUEV or TUSL; no TUSZ threshold tuning"
    status["notes"] = (
        "Full TUSZ eval confirmation cohort is complete across all 43 eval patients; "
        "models, thresholds and CHB test artifacts remained frozen. Seizure-detection "
        "performance is preliminary external validation, not a clinical validation claim."
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    release_path = REPORT_DIR / "release_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    tuh = release.setdefault("analyses", {}).setdefault("tuh_external_validation", {})
    tuh.update(
        {
            "status": status["status"],
            "status_file": rel(status_path.name),
            "status_sha256": sha256(status_path),
            "full_tusz_eval": True,
            "full_tusz_manifest": rel(cohort_path.name),
            "full_tusz_manifest_sha256": sha256(cohort_path),
            "full_tusz_results": rel(results_path.name),
            "full_tusz_results_sha256": sha256(results_path),
            "full_tusz_audit": rel(audit_path.name),
            "full_tusz_audit_sha256": sha256(audit_path),
            "full_tusz_results_csv": rel(results_csv.name),
            "full_tusz_results_csv_sha256": sha256(results_csv),
            "external_performance_reported": True,
            "retrained": False,
            "threshold_retuned": False,
            "signal_quality_warning_records": qc_summary["max_abs_gt_1000_uv_records"],
        }
    )
    release["status"] = "h4_preliminary_negative_h5_descriptive_complete_tuh_full_eval_complete"
    release["interpretation"] = [
        item
        for item in release.get("interpretation", [])
        if item not in {"tuh-preparation-complete-but-data-missing", "no_external_performance_claim"}
    ]
    for item in ["tuh-full-eval-preliminary-frozen", "tusz-detection-only-external-validation", "no_external_construct_claim"]:
        if item not in release["interpretation"]:
            release["interpretation"].append(item)
    release["remaining_work"] = [
        item for item in release.get("remaining_work", []) if item != "tuh-full-eval-optional-confirmation"
    ]
    if "optional-tuh-construct-validation-TUAR-TUEV-TUSL" not in release["remaining_work"]:
        release["remaining_work"].append("optional-tuh-construct-validation-TUAR-TUEV-TUSL")
    release["external_audit"] = {
        "cohort": rel(cohort_path.name),
        "results": rel(results_path.name),
        "audit": rel(audit_path.name),
        "audit_sha256": sha256(audit_path),
        "status": "complete",
    }
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "release_status": release["status"],
                "tuh_status": status["status"],
                "cohort": cohort["cohort_id"],
                "patients": cohort["patient_count"],
                "records": cohort["record_count"],
                "results_manifest_sha256": sha256(results_path),
                "audit_manifest_sha256": sha256(audit_path),
                "qc": qc_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
