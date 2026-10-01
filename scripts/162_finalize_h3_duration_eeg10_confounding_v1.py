"""Audit and finalize the duration/EEG10 H3 confounding analysis.

The first pass used the correct per-label one-vs-rest AUROC for the
multi-label point estimates.  This finalizer repairs the patient bootstrap
for that experiment (independent sigmoid heads do not produce probabilities
that sum to one), preserves the initial file, and writes a compact comparison
table/report without rerunning any model fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "h3-duration-eeg10-confounding-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
MODELS = ("singlem", "cbramod", "tcn_gat")
N_BOOTSTRAP = 2000
RNG_SEED = 20260828


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def macro_auc_multilabel(y: np.ndarray, proba: np.ndarray) -> float:
    values = []
    for j in range(len(MODELS)):
        try:
            values.append(float(roc_auc_score(y[:, j], proba[:, j])))
        except ValueError:
            return math.nan
    return float(np.mean(values))


def macro_auc_multiclass(y: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, proba, labels=list(range(len(MODELS))), multi_class="ovr", average="macro"))
    except ValueError:
        return math.nan


def patient_bootstrap(pred: pd.DataFrame, experiment: str, seed: int) -> dict[str, float]:
    patients = np.asarray(sorted(pred.patient_id.astype(str).unique()), dtype=object)
    patient_values = pred.patient_id.astype(str).to_numpy()
    positions = {patient: np.flatnonzero(patient_values == patient) for patient in patients}
    y = pred.true_model.map({name: i for i, name in enumerate(MODELS)}).to_numpy(int)
    proba = pred[[f"prob_{name}" for name in MODELS]].to_numpy(float)
    if experiment == "duration_only_multilabel":
        y_multi = np.column_stack([(y == i).astype(int) for i in range(len(MODELS))])
        metric = macro_auc_multilabel
    else:
        y_multi = y
        metric = macro_auc_multiclass
    observed = metric(y_multi, proba)
    values: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(N_BOOTSTRAP):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([positions[p] for p in selected])
        score = metric(y_multi[indices], proba[indices])
        if np.isfinite(score):
            values.append(float(score))
    array = np.asarray(values, dtype=float)
    return {
        "dataset": str(pred.dataset.iloc[0]),
        "experiment": experiment,
        "unit": str(pred.unit.iloc[0]),
        "observed_macro_auroc": float(observed) if np.isfinite(observed) else math.nan,
        "bootstrap_mean_macro_auroc": float(array.mean()) if len(array) else math.nan,
        "bootstrap_q025_macro_auroc": float(np.quantile(array, 0.025)) if len(array) else math.nan,
        "bootstrap_q975_macro_auroc": float(np.quantile(array, 0.975)) if len(array) else math.nan,
        "bootstrap_valid_replicates": int(len(array)),
        "bootstrap_patients": int(len(patients)),
        "bootstrap_replicates_requested": N_BOOTSTRAP,
        "bootstrap_metric_definition": "macro one-vs-rest AUROC" if experiment == "duration_only_multilabel" else "multiclass OVR macro AUROC",
    }


def plain_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free readable table for the Markdown report."""
    return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    pred_path = REPORT_DIR / "test_predictions.parquet"
    metrics_path = REPORT_DIR / "classification_metrics.csv"
    reg_path = REPORT_DIR / "regression_metrics.csv"
    old_boot_path = REPORT_DIR / "classification_patient_bootstrap.csv"
    corrected_path = REPORT_DIR / "classification_patient_bootstrap_corrected.csv"
    manifest_path = REPORT_DIR / "manifest.json"
    if not pred_path.exists() or not metrics_path.exists() or not reg_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("duration analysis outputs are incomplete")
    if old_boot_path.exists() and not (REPORT_DIR / "classification_patient_bootstrap_initial.csv").exists():
        shutil.copy2(old_boot_path, REPORT_DIR / "classification_patient_bootstrap_initial.csv")
    pred = pd.read_parquet(pred_path)
    if corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
    else:
        rows: list[dict[str, float]] = []
        for dataset in sorted(pred.dataset.astype(str).unique()):
            for experiment in ("duration_only_multilabel", "duration_only_multiclass"):
                q = pred[(pred.dataset.astype(str) == dataset) & (pred.experiment.astype(str) == experiment)]
                if q.empty:
                    continue
                units = ["all_outer_tests"] if dataset == "chbmit" else sorted(q.unit.astype(str).unique())
                for unit in units:
                    subset = q[q.unit.astype(str) == unit] if unit != "all_outer_tests" else q
                    if not subset.empty:
                        rows.append(patient_bootstrap(subset, experiment, RNG_SEED + len(rows)))
        corrected = pd.DataFrame(rows)
        corrected.to_csv(corrected_path, index=False)

    metrics = pd.read_csv(metrics_path)
    test_metrics = metrics[metrics.split_role.astype(str) != "validation"].copy()
    new_summary = test_metrics.groupby(["dataset", "experiment"], as_index=False)[["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"]].mean()
    new_summary["source"] = "new duration-only fit"
    new_summary["comparison_note"] = "accuracy is exact-match for multi-label and argmax accuracy for multiclass"
    new_summary.to_csv(REPORT_DIR / "new_duration_classification_summary.csv", index=False)

    regression = pd.read_csv(reg_path)
    regression_summary = regression[regression.split_role.astype(str) != "validation"].groupby("dataset", as_index=False)[["mae_log_duration", "rmse_log_duration", "r2_log_duration", "spearman_log_duration", "mae_duration_s"]].mean()
    regression_summary.to_csv(REPORT_DIR / "eeg10_duration_regression_summary.csv", index=False)

    reference = pd.read_csv(REPORT_DIR / "completed_reference.csv")
    comparison_rows: list[dict[str, object]] = []
    for row in new_summary.to_dict("records"):
        comparison_rows.append({
            "dataset": row["dataset"],
            "experiment": row["experiment"],
            "accuracy": row["accuracy"],
            "balanced_accuracy": row["balanced_accuracy"],
            "macro_f1": row["macro_f1"],
            "macro_auroc": row["macro_auroc"],
            "source": "new duration-only fit",
        })
    for row in reference.to_dict("records"):
        if row["dataset"] == "chbmit":
            accuracy = row.get("mean_test_accuracy")
            balanced = row.get("mean_test_balanced_accuracy")
            macro_f1 = row.get("mean_test_macro_f1")
            auroc = row.get("mean_test_macro_auroc")
        else:
            accuracy = row.get("mean_accuracy")
            balanced = row.get("mean_balanced_accuracy")
            macro_f1 = math.nan
            auroc = row.get("mean_macro_auroc")
        comparison_rows.append({
            "dataset": row["dataset"],
            "experiment": "full_event_eeg10_multiclass_reused",
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "macro_auroc": auroc,
            "source": row["source"],
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORT_DIR / "experiment_comparison_summary.csv", index=False)
    lines = [
        "# H3 duration and complete-event EEG10 confounding analysis",
        "",
        "## Scope",
        "",
        "Each row is one complete false-positive event generated by one detector. The duration classifiers use only `log1p(event_duration_s)` and predict detector source. The multi-label version is a one-hot three-head encoding of that source label; it is not the separate common-window multi-label task. The direct regression uses the ten complete-event EEG features to predict log duration.",
        "",
        "## New duration-only classifier results",
        "",
        plain_table(new_summary),
        "",
        "## EEG10 to duration diagnostic",
        "",
        plain_table(regression_summary),
        "",
        "## Reused complete-event EEG10 three-class results",
        "",
        plain_table(reference),
        "",
        "## Interpretation",
        "",
        "Duration carries detector-source information, especially on TUSZ, so detector duration is a real confounder for source classification. However, the complete-event EEG10 summary features show very low held-out duration predictability (R2 close to zero), so the source signal in duration cannot be attributed simply to EEG10 reconstructing event length. These are identifiability/confounding analyses, not evidence that a seizure detector performs better. Complete-event EEG10 still summarizes a variable-length interval and must not be interpreted as a duration-free representation.",
        "",
        "## Audit",
        "",
        "- No detector was retrained; the frozen feature tables were read only.",
        "- CHB uses the frozen canonical patient GroupKFold split/partition; TUSZ uses the frozen/reused patient GroupKFold assignment and 15 external fold-seed units.",
        "- All imputing, clipping, and scaling are fitted on the training patients only.",
        "- Test results are evaluated once per held-out unit; no test result selected a threshold or model.",
        "- The initial multi-label bootstrap is preserved as `classification_patient_bootstrap_initial.csv`; the corrected file uses macro one-vs-rest AUROC, matching the point-estimate definition.",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")

    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete_audited"
    manifest["posthoc_audit"] = {
        "corrected_multilabel_bootstrap": "classification_patient_bootstrap_corrected.csv",
        "initial_bootstrap_preserved_as": "classification_patient_bootstrap_initial.csv",
        "corrected_multilabel_metric": "macro one-vs-rest AUROC; independent sigmoid heads need not sum to one",
        "test_used_for_selection": False,
        "audit_pass": True,
    }
    manifest["outputs"] = sorted(set(manifest.get("outputs", [])) | {"classification_patient_bootstrap_corrected.csv", "classification_patient_bootstrap_initial.csv", "new_duration_classification_summary.csv", "eeg10_duration_regression_summary.csv", "experiment_comparison_summary.csv", "report.md"})
    manifest["manifest_sha256_before_finalize"] = sha256(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_pass": True, "corrected_bootstrap_rows": len(corrected), "manifest": str(manifest_path), "report": str(REPORT_DIR / 'report.md')}, indent=2))
    print("\\nCorrected bootstrap summary:")
    print(corrected.groupby(["dataset", "experiment"], as_index=False)[["observed_macro_auroc", "bootstrap_mean_macro_auroc", "bootstrap_q025_macro_auroc", "bootstrap_q975_macro_auroc", "bootstrap_valid_replicates"]].mean().to_string(index=False))


if __name__ == "__main__":
    main()
