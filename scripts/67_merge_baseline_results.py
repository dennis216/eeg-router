from __future__ import annotations

"""Merge the original and supplemental baseline matrices for reporting."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OLD_ROOT = ROOT / "runs" / "baselines" / "v3-compact-baselines"
NEW_ROOT = ROOT / "runs" / "baselines" / "v3-compact-conv-baselines"
OUT_ROOT = ROOT / "runs" / "baselines" / "v3-compact-baselines-merged"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD_ROOT / "baseline_results.csv")
    old["baseline_family"] = old.model.map({"eegnet": "neural_baseline", "psd_catboost": "feature_baseline"})
    new = pd.read_csv(NEW_ROOT / "baseline_results.csv")
    new["baseline_family"] = new.model.map({"deepconvnet": "neural_baseline", "shallowconvnet": "neural_baseline"})
    table = pd.concat([old, new], ignore_index=True).sort_values(["model", "split", "seed"])
    table.to_csv(OUT_ROOT / "all_baseline_results.csv", index=False)
    summary = (
        table.groupby(["model", "baseline_family"])
        .agg(
            runs=("seed", "count"),
            test_sensitivity_mean=("test_event_sensitivity", "mean"),
            test_sensitivity_sd=("test_event_sensitivity", "std"),
            test_fa_per_24h_mean=("test_fa_per_24h", "mean"),
            test_fa_per_24h_median=("test_fa_per_24h", "median"),
            validation_auroc_mean=("validation_window_auroc", "mean"),
            validation_auroc_sd=("validation_window_auroc", "std"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_ROOT / "baseline_summary.csv", index=False)
    manifest = {
        "namespace": "v3-compact-baselines-merged",
        "source_namespaces": ["v3-compact-baselines", "v3-compact-conv-baselines"],
        "run_count": int(len(table)),
        "source_run_counts": {
            "v3-compact-baselines": int(len(old)),
            "v3-compact-conv-baselines": int(len(new)),
        },
        "all_results_csv": str((OUT_ROOT / "all_baseline_results.csv").relative_to(ROOT)),
        "summary_csv": str((OUT_ROOT / "baseline_summary.csv").relative_to(ROOT)),
    }
    (OUT_ROOT / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
