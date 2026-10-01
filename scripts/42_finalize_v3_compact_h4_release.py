"""Finalize hashes and release metadata after the H4 router evaluation."""

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
    gates = json.loads((REPORT_DIR / "gates.json").read_text())
    release_path = REPORT_DIR / "release_manifest.json"
    release = json.loads(release_path.read_text())
    h4_outputs = {
        "gates": "gates.json",
        "validation_single_model": "h4_validation_single_model.csv",
        "validation_fp_jaccard": "h4_validation_fp_jaccard.csv",
        "validation_oracle": "h4_validation_oracle.csv",
        "train_event_manifest": "train_event_table_manifest.json",
        "event_feature_manifest": "h4_event_feature_manifest.json",
        "router_manifest": "h4_router_manifest.json",
        "router_test_results": "h4_router_test_results.csv",
        "router_validation_workpoints": "h4_router_validation_workpoints.csv",
        "router_patient_bootstrap": "h4_router_patient_bootstrap.csv",
        "router_waterfall": "h4_router_waterfall.csv",
        "feature_dictionary": "feature_dictionary.csv",
    }
    h4_hashes = {name: sha256(REPORT_DIR / filename) for name, filename in h4_outputs.items()}
    frozen_inputs = {
        "release_id": "v3-compact-extension-v1",
        "gate": "H4_functional_router_gate",
        "decision_partition": "validation_only",
        "test_used_for_branch_selection": False,
        "event_table_manifest_sha256": sha256(REPORT_DIR / "event_table_manifest.json"),
        "validation_event_table_manifest_sha256": sha256(REPORT_DIR / "validation_event_table_manifest.json"),
        "h2_summary_sha256": sha256(REPORT_DIR / "h2_h3_summary.json"),
        "h2_cluster_profiles_sha256": sha256(REPORT_DIR / "h2_cluster_profiles.csv"),
        "h3_fingerprint_sha256": sha256(REPORT_DIR / "h3_model_fingerprint.csv"),
        "feature_dictionary_sha256": sha256(REPORT_DIR / "feature_dictionary.csv"),
        "split17_manifest_sha256": sha256(ROOT / "manifests" / "splits" / "split_17.json"),
        "split42_manifest_sha256": sha256(ROOT / "manifests" / "splits" / "split_42.json"),
        "window_manifest_sha256": sha256(ROOT / "manifests" / "windows.parquet"),
        "pilot_config_sha256": sha256(ROOT / "configs" / "pilot" / "bsn_pilot_v1.yaml"),
        "branch": gates["branch"],
        "oracle_gate_pass": gates["oracle_gate"]["pass"],
        "fp_jaccard_gate_pass": gates["fp_jaccard_gate"]["pass"],
        "feature_availability_pass": gates["feature_availability_pass"],
    }
    (REPORT_DIR / "h4_frozen_inputs.json").write_text(json.dumps(frozen_inputs, indent=2, sort_keys=True) + "\n")
    h4_hashes["frozen_inputs"] = sha256(REPORT_DIR / "h4_frozen_inputs.json")
    release["status"] = "frozen_primary_matrix_h1_h2_h3_h4_complete_h5_tuh_pending"
    release["analyses"]["h4"] = {
        "status": "complete_preliminary_compute_budgeted",
        "branch": gates["branch"],
        "gate": "outputs/reports/v3-compact-extension-v1/gates.json",
        "gate_sha256": h4_hashes["gates"],
        "frozen_inputs": "outputs/reports/v3-compact-extension-v1/h4_frozen_inputs.json",
        "router_manifest": "outputs/reports/v3-compact-extension-v1/h4_router_manifest.json",
        "router_manifest_sha256": h4_hashes["router_manifest"],
        "test_results": "outputs/reports/v3-compact-extension-v1/h4_router_test_results.csv",
        "test_results_sha256": h4_hashes["router_test_results"],
        "validation_workpoints": "outputs/reports/v3-compact-extension-v1/h4_router_validation_workpoints.csv",
        "patient_bootstrap": "outputs/reports/v3-compact-extension-v1/h4_router_patient_bootstrap.csv",
        "waterfall": "outputs/reports/v3-compact-extension-v1/h4_router_waterfall.csv",
        "train_fit_subset": "manifests/h4_train_fit_index_split{17,42}.parquet",
        "train_fit_events": "outputs/events/v3-compact-extension-v1_train_events.parquet",
        "train_fit_event_sha256": json.loads((REPORT_DIR / "train_event_table_manifest.json").read_text())["event_table_sha256"],
        "test_evaluation_count_per_method_run": 1,
        "test_used_for_selection": False,
        "interpretation": "router candidate evaluated under a 5000-window train-only compute budget; not a convergence-matched full-v3 result",
    }
    release["feature_library"]["h4_event_features_sha256"] = sha256(ROOT / "outputs" / "events" / "v3-compact-extension-v1_h4_event_features.parquet")
    release["feature_library"]["h4_event_features_rows"] = 80104
    release["feature_library"]["feature_dictionary_sha256"] = h4_hashes["feature_dictionary"]
    release["interpretation"] = [
        "preliminary",
        "compute-matched",
        "h4-router-train-fit-uses-5000-window-train-only-subset",
        "h4-router-test-evaluated-once-per-method-run",
        "h5-and-tuh-not-run",
        "no_full_v3_completion_claim",
        "no_architecture_superiority_claim_from_pilot_alone",
    ]
    release["h4_hashes"] = h4_hashes
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release": str(release_path), "status": release["status"], "branch": gates["branch"], "h4_hashes": h4_hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
