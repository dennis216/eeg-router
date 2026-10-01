"""Read-only audit for the completed independent TUSZ H3 replication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
RUN_ID = "h3-external-tusz-v1"
BASE = ROOT / "outputs" / "reports" / RUN_ID
SOURCE = ROOT / "outputs" / "reports" / "v3-groupkfold-confirmatory-v1"
OUT = BASE / "h3_external_tusz_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    feature_path = BASE / "tusz_h3_fp_features.parquet"
    feature_manifest_path = BASE / "tusz_h3_fp_features_manifest.json"
    event_path = SOURCE / "tusz_groupkfold_external_v1_events.parquet"
    cohort_path = SOURCE / "tusz_groupkfold_external_manifest.json"
    scoring_path = SOURCE / "tusz_groupkfold_external_v1_scoring_manifest.json"
    preflight_path = BASE / "no_duplicate_overlap_preflight.json"
    summary_path = BASE / "h3_external_tusz_summary_metrics.csv"
    fold_path = BASE / "h3_external_tusz_cv_fold_metrics.csv"
    boot_path = BASE / "h3_external_tusz_patient_bootstrap.csv"
    perm_path = BASE / "h3_external_tusz_patient_permutation.csv"
    assign_path = BASE / "tusz_patient_groupkfold_assignments.csv"
    required = [feature_path, feature_manifest_path, event_path, cohort_path, scoring_path, preflight_path, summary_path, fold_path, boot_path, perm_path, assign_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    features = pd.read_parquet(feature_path)
    events = pd.read_parquet(event_path)
    fp = events[~events["is_true_positive"].astype(bool)].copy()
    feature_keys = ["model", "fold", "seed", "patient_id", "recording", "event_index"]
    key_source = fp[feature_keys].astype(str).sort_values(feature_keys).reset_index(drop=True)
    key_feature = features[feature_keys].astype(str).sort_values(feature_keys).reset_index(drop=True)
    key_match = bool(key_source.equals(key_feature))

    summary = pd.read_csv(summary_path)
    fold = pd.read_csv(fold_path)
    bootstrap = pd.read_csv(boot_path)
    permutation = pd.read_csv(perm_path)
    assignments = pd.read_csv(assign_path)
    scoring = json.loads(scoring_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    feature_manifest = json.loads(feature_manifest_path.read_text())
    preflight = json.loads(preflight_path.read_text())

    numeric = features.select_dtypes(include=[np.number])
    checks = {
        "preflight_status_pass": preflight.get("status") == "PASS",
        "no_duplicate_target_records": preflight.get("target_records") == 0,
        "feature_rows_equal_frozen_fp_rows": len(features) == len(fp) == 36698,
        "feature_key_set_matches_frozen_fp_events": key_match,
        "feature_patients_43": features["patient_id"].nunique() == 43,
        "feature_recordings_847": features["recording"].nunique() == 847,
        "feature_models_exact": sorted(features["model"].unique().tolist()) == ["cbramod", "singlem", "tcn_gat"],
        "feature_numeric_finite": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "feature_extraction_no_fit_or_selection": feature_manifest.get("fit_or_selection_performed") is False,
        "cohort_frozen_43_patients_880_records": cohort.get("patient_count") == 43 and cohort.get("record_count") == 880,
        "frozen_external_scoring_45_runs": scoring.get("inference_runs_scored") == 45 and scoring.get("expected_inference_runs") == 45,
        "frozen_detector_model_training_false": scoring.get("model_training") is False,
        "frozen_thresholds_not_retuned": scoring.get("threshold_retuning") is False,
        "external_labels_not_used_for_selection": scoring.get("external_labels_used_for_selection") is False,
        "patient_assignment_complete_once": len(assignments) == 43 and assignments["patient_id"].is_unique and set(assignments["cv_fold"]) == set(range(5)),
        "summary_90_units": len(summary) == 90 and summary["status"].eq("ok").all(),
        "summary_all_test_selection_false": summary["test_used_for_selection"].eq(False).all(),
        "summary_correct_auroc_chance_0_5": summary["chance_macro_auroc"].eq(0.5).all(),
        "summary_15_checkpoint_units": summary[["external_fold", "external_seed"]].drop_duplicates().shape[0] == 15,
        "five_inner_patient_folds_per_unit": len(fold) == 450 and fold["status"].eq("ok").all(),
        "patient_bootstrap_2000_per_unit": len(bootstrap) == 90 and bootstrap["bootstrap_replicates"].eq(2000).all() and bootstrap["bootstrap_valid"].eq(2000).all(),
        "conditional_patient_permutation_1000_per_unit": len(permutation) == 90 and permutation["permutations"].eq(1000).all() and permutation["permutations_valid"].eq(1000).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "analysis_id": RUN_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "interpretation": {
            "external_replication": "TUSZ-only patient-group cross-validation with the CHB H3 classifier specification fixed; this tests reproducibility of detector-source information, not zero-shot CHB classifier transfer.",
            "patient_grouping": "all EDFs and event rows for a canonical TUSZ patient remain in one of five inner test folds",
            "patient_centering": "label-free within-partition patient means, reported as a sensitivity analysis",
            "permutation": "secondary conditional held-out label permutation within patient; it does not refit the classifier and is not a training-null p-value",
            "claim_boundary": "source predictability supports detector-dependent output structure; it does not identify a physiological mechanism or prove causation",
        },
        "hashes": {
            "feature_table_sha256": sha256(feature_path),
            "feature_manifest_sha256": sha256(feature_manifest_path),
            "source_event_table_sha256": sha256(event_path),
            "cohort_manifest_sha256": sha256(cohort_path),
            "scoring_manifest_sha256": sha256(scoring_path),
            "patient_assignment_sha256": sha256(assign_path),
            "summary_sha256": sha256(summary_path),
            "fold_metrics_sha256": sha256(fold_path),
            "bootstrap_sha256": sha256(boot_path),
            "permutation_sha256": sha256(perm_path),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
