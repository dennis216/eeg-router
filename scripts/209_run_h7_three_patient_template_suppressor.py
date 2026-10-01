"""Score the frozen H7 specific-template suppressor on the three-patient pilot.

This script does not fit templates, retrain a detector, or select a policy from
test data.  It filters the already frozen H7 exploratory event scores to the
same three canonical patients used by the paused fine-tuning pilot and writes
a reproducible, clearly exploratory subset report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
SUBSET = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/h7_three_patient_subset_manifest.json"
SOURCE_SCORES = ROOT / (
    "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-"
    "exploratory-tusz-holdout-score-v1/holdout_event_scores.parquet"
)
SOURCE_WATERFALL = ROOT / (
    "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-"
    "exploratory-tusz-holdout-score-v1/patient_waterfall.csv"
)
POLICY = ROOT / (
    "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-"
    "dev-policy-freeze-v1/manifest.json"
)
OUT = ROOT / "outputs/reports/h7-template-suppressor-3patients-specific-v1"
METHOD = "specific"
ALPHA = 1.25
BOOTSTRAP_REPLICATES = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_row(scope: str, method: str, rows: pd.DataFrame) -> dict:
    truth_rows = rows[rows.truth_events > 0].copy()
    duration_all = float(rows.holdout_duration_s.sum())
    duration_truth = float(truth_rows.holdout_duration_s.sum())
    truth_total = int(rows.truth_events.sum())
    fp_all = int(rows[f"{method}_fp"].sum())
    tp_all = int(rows[f"{method}_tp"].sum())
    fp_truth = int(truth_rows[f"{method}_fp"].sum())
    tp_truth = int(truth_rows[f"{method}_tp"].sum())
    out = {
        "scope": scope,
        "method": method,
        "patients": int(rows.patient_id.nunique()),
        "patient_model_units": int(len(rows)),
        "patients_with_truth": int(truth_rows.patient_id.nunique()),
        "truth_events": truth_total,
        "fp_events_all_units": fp_all,
        "tp_events_all_units": tp_all,
        "pooled_sensitivity": float(tp_all / truth_total) if truth_total else None,
        "pooled_fa_per_24h_all_units": float(fp_all / duration_all * 86400.0) if duration_all else None,
        "macro_fa_per_24h_all_units": float(rows[f"{method}_fa24"].mean()) if len(rows) else None,
        "macro_fa_per_24h_truth_units": float(truth_rows[f"{method}_fa24"].mean()) if len(truth_rows) else None,
        "macro_sensitivity_truth_units": float(truth_rows[f"{method}_sensitivity"].mean()) if len(truth_rows) else None,
        "fp_events_truth_units": fp_truth,
        "tp_events_truth_units": tp_truth,
        "pooled_fa_per_24h_truth_units": float(fp_truth / duration_truth * 86400.0) if duration_truth else None,
    }
    return out


def _patient_aggregate(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    # Aggregate all three detector units before bootstrap so a replicate is a
    # true patient resample rather than a detector-unit resample.
    agg = rows.groupby("patient_id", as_index=False).agg(
        truth_events=("truth_events", "max"),
        duration_s=("holdout_duration_s", "max"),
        fp=(f"{method}_fp", "sum"),
        tp=(f"{method}_tp", "sum"),
    )
    agg["fa24"] = agg.fp / agg.duration_s * 86400.0
    agg["sensitivity"] = agg.tp / agg.truth_events.replace(0, np.nan)
    return agg


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise RuntimeError(f"Refusing overwrite: {OUT}")
    for path in (SUBSET, SOURCE_SCORES, SOURCE_WATERFALL, POLICY):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    subset = json.loads(SUBSET.read_text())
    units = subset.get("units", [])
    patients = sorted({str(row["patient_id"]) for row in units})
    models = sorted({str(row["model"]) for row in units})
    if len(patients) != 3 or len(units) != 9 or len(models) != 3:
        raise ValueError(f"Expected 3 patients x 3 models, got patients={patients}, units={len(units)}, models={models}")

    policy = json.loads(POLICY.read_text())
    primary = policy.get("primary_policy", {})
    # The specific policy is a frozen secondary policy selected on DEV.  The
    # run is intentionally not allowed to substitute an alpha from test data.
    if float(next(x["alpha"] for x in policy.get("prespecified_secondary_ablation_policies", []) if x["method"] == METHOD)) != ALPHA:
        raise ValueError("Frozen specific alpha does not match the declared pilot policy")

    scores = pd.read_parquet(SOURCE_SCORES)
    scores = scores[
        scores.partition.eq("dev")
        & scores.phase.eq("holdout")
        & scores.patient_id.isin(patients)
        & scores.model.isin(models)
    ].copy()
    if scores.empty or not scores.shared_suppressed.dtype == bool:
        raise ValueError("Missing frozen event scores or unexpected suppression column type")
    if "specific_suppressed" not in scores.columns:
        raise ValueError("Frozen specific-template scores are missing")
    scores["specific_suppressed"] = scores["specific_suppressed"].astype(bool)
    scores.to_parquet(OUT / "specific_template_event_scores.parquet", index=False)

    waterfall = pd.read_csv(SOURCE_WATERFALL)
    waterfall = waterfall[
        waterfall.partition.eq("dev")
        & waterfall.patient_id.isin(patients)
        & waterfall.model.isin(models)
    ].copy()
    if len(waterfall) != 9:
        raise ValueError(f"Expected nine patient-model rows, found {len(waterfall)}")
    # The source scorer used the frozen alpha=1.25 specific policy.  Verify the
    # arithmetic on the copied rows before publishing the subset report.
    if not np.allclose(
        waterfall["specific_fa24"].to_numpy(dtype=float),
        waterfall["specific_fp"].to_numpy(dtype=float) / waterfall["holdout_duration_s"].to_numpy(dtype=float) * 86400.0,
    ):
        raise ValueError("Source waterfall arithmetic check failed")
    waterfall.to_csv(OUT / "patient_waterfall.csv", index=False)

    metric_rows = []
    for method in ("baseline", METHOD):
        metric_rows.append(_metric_row("all_three_patients", method, waterfall))
        metric_rows.append(_metric_row("patients_with_test_truth", method, waterfall[waterfall.truth_events > 0]))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    # Patient-level bootstrap is provided as a descriptive uncertainty display;
    # with only three patients it is not a confirmatory inferential result.
    rng = np.random.default_rng(20260831)
    boot_rows = []
    for method in ("baseline", METHOD):
        agg = _patient_aggregate(waterfall, method)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sample = agg.iloc[rng.integers(0, len(agg), len(agg))]
            truth = int(sample.truth_events.sum())
            duration = float(sample.duration_s.sum())
            boot_rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "pooled_sensitivity": float(sample.tp.sum() / truth) if truth else np.nan,
                    "pooled_fa_per_24h": float(sample.fp.sum() / duration * 86400.0) if duration else np.nan,
                    "macro_sensitivity": float(sample.sensitivity.mean()),
                    "macro_fa_per_24h": float(sample.fa24.mean()),
                }
            )
    pd.DataFrame(boot_rows).to_csv(OUT / "patient_bootstrap.csv", index=False)

    manifest = {
        "release_id": OUT.name,
        "status": "completed_three_patient_specific_template_score",
        "scope": "same three-patient pilot as paused H7 fine-tuning queue",
        "method": METHOD,
        "policy_alpha": ALPHA,
        "policy_selection": "frozen secondary policy selected on validation/development data; no test-based selection",
        "primary_frozen_policy_in_source": primary,
        "patients": patients,
        "models": models,
        "patient_model_units": int(len(waterfall)),
        "test_scores_reused_from_prior_exploratory_run": True,
        "test_evaluation_count": 1,
        "no_post_test_selection": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "exploratory_small_sample": True,
        "source_subset_sha256": sha256(SUBSET),
        "source_event_scores_sha256": sha256(SOURCE_SCORES),
        "source_waterfall_sha256": sha256(SOURCE_WATERFALL),
        "source_policy_manifest_sha256": sha256(POLICY),
        "files": {
            "specific_template_event_scores": "specific_template_event_scores.parquet",
            "patient_waterfall": "patient_waterfall.csv",
            "metrics": "metrics.csv",
            "patient_bootstrap": "patient_bootstrap.csv",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
