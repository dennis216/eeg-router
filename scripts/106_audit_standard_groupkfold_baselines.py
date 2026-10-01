"""Audit and summarize the standard GroupKFold baseline addendum."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("BFA_ROOT", str(Path(__file__).resolve().parent.parent)))
NAMESPACE = "v3-groupkfold-baselines-v1"
RUN_ROOT = ROOT / "runs" / "baselines" / NAMESPACE
OUT = ROOT / "outputs" / "reports" / NAMESPACE
MODELS = ("eegnet", "psd_catboost", "deepconvnet", "shallowconvnet")
FOLDS = range(5)
SEEDS = (17, 42, 3407)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def numeric_or_nan(value):
    """Preserve undefined metrics (for example delay with no true positives)."""
    return float("nan") if value is None else float(value)


def main() -> None:
    rows = []
    failures = []
    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                run = RUN_ROOT / model / f"{model}_split{fold}_seed{seed}"
                completed_path = run / "completed.json"
                manifest_path = run / "run_manifest.json"
                if not completed_path.exists() or not manifest_path.exists():
                    failures.append(f"missing {model} fold={fold} seed={seed}")
                    continue
                completed = json.loads(completed_path.read_text())
                manifest = json.loads(manifest_path.read_text())
                test = completed.get("test") or {}
                score = test.get("score") or {}
                if not test or test.get("test_evaluation_count") != 1:
                    failures.append(f"test policy failed {model} fold={fold} seed={seed}")
                if test.get("threshold_source") != "validation_only":
                    failures.append(f"threshold source failed {model} fold={fold} seed={seed}")
                split_manifest = str(manifest.get("split_manifest", ""))
                if "groupkfold_cv_v1" not in split_manifest:
                    failures.append(f"wrong split manifest {model} fold={fold} seed={seed}")
                rows.append({
                    "model": model, "fold": fold, "seed": seed,
                    "run_dir": str(run.relative_to(ROOT)).replace("\\", "/"),
                    "sensitivity": numeric_or_nan(score.get("event_sensitivity")),
                    "fa_per_24h": numeric_or_nan(score.get("fa_per_24h")),
                    "median_delay_s": numeric_or_nan(score.get("median_delay_s")),
                    "median_delay_defined": score.get("median_delay_s") is not None,
                    "threshold": numeric_or_nan(score.get("threshold")),
                    "test_evaluation_count": int(test.get("test_evaluation_count", -1)),
                    "threshold_source": str(test.get("threshold_source")),
                    "completed_sha256": sha256(completed_path),
                    "manifest_sha256": sha256(manifest_path),
                })
    expected = len(MODELS) * len(tuple(FOLDS)) * len(SEEDS)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "baseline_run_metrics.csv", index=False)
    summary = frame.groupby("model", as_index=False).agg(
        runs=("model", "size"), sensitivity_mean=("sensitivity", "mean"),
        sensitivity_median=("sensitivity", "median"), fa_per_24h_mean=("fa_per_24h", "mean"),
        fa_per_24h_median=("fa_per_24h", "median"), delay_mean_s=("median_delay_s", "mean"),
        delay_median_s=("median_delay_s", "median"),
    ) if not frame.empty else pd.DataFrame()
    summary.to_csv(OUT / "baseline_summary.csv", index=False)
    report = {
        "namespace": NAMESPACE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": expected, "completed_runs": len(rows),
        "models": list(MODELS), "folds": list(FOLDS), "seeds": list(SEEDS),
        "matrix_complete": len(rows) == expected and not failures,
        "failures": failures,
        "split_policy": "standard GroupKFold outer manifests with patient-level inner validation",
        "threshold_policy": "validation_only", "test_evaluation_count_required": 1,
        "undefined_median_delay_runs": int(frame["median_delay_defined"].eq(False).sum()) if not frame.empty else 0,
        "run_metrics_sha256": sha256(OUT / "baseline_run_metrics.csv"),
        "summary_sha256": sha256(OUT / "baseline_summary.csv"),
    }
    (OUT / "baseline_audit_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["matrix_complete"]:
        raise SystemExit(2)


if __name__ == "__main__": main()
