from __future__ import annotations

"""Audit and summarize the frozen v3 compact baseline matrix."""

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "runs" / "baselines" / "v3-compact-baselines"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    expected = [
        (model, split, seed)
        for model in ("eegnet", "psd_catboost")
        for split in (17, 42)
        for seed in (17, 42, 3407)
    ]
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for model, split, seed in expected:
        run_dir = RUN_ROOT / model / f"{model}_split{split}_seed{seed}"
        completed_path = run_dir / "completed.json"
        if not completed_path.exists():
            problems.append(f"missing completed.json: {run_dir}")
            continue
        completed = json.loads(completed_path.read_text())
        validation = completed.get("validation", {})
        operating = validation.get("selected_event_operating_point", {})
        test = completed.get("test", {})
        score = test.get("score", {})
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        lock = json.loads((run_dir / "threshold_lock.json").read_text())
        test_probability_path = run_dir / "test" / "probabilities.parquet"
        validation_probability_path = run_dir / "validation" / "probabilities.parquet"
        model_path = run_dir / ("checkpoints/final.pt" if model == "eegnet" else "model.cbm")
        model_hash_path = model_path.with_suffix(model_path.suffix + ".sha256")
        if test.get("test_evaluation_count") != 1:
            problems.append(f"test evaluation count != 1: {run_dir}")
        if test.get("threshold_source") != "validation_only":
            problems.append(f"test threshold source is not validation_only: {run_dir}")
        if manifest.get("test_accessed") is not True:
            problems.append(f"manifest test_accessed is not true: {run_dir}")
        if lock.get("source") != "validation_only":
            problems.append(f"threshold lock source is not validation_only: {run_dir}")
        for path in (test_probability_path, validation_probability_path, model_path, model_hash_path):
            if not path.exists():
                problems.append(f"missing artifact: {path}")
        if test_probability_path.exists() and sha256_file(test_probability_path) != test.get("probability_sha256"):
            problems.append(f"test probability hash mismatch: {run_dir}")
        if validation_probability_path.exists() and sha256_file(validation_probability_path) != validation.get("probability_sha256"):
            problems.append(f"validation probability hash mismatch: {run_dir}")
        rows.append(
            {
                "model": model,
                "split": split,
                "seed": seed,
                "validation_window_auroc": validation.get("window_auroc"),
                "validation_window_average_precision": validation.get("window_average_precision"),
                "validation_event_sensitivity": operating.get("event_sensitivity"),
                "validation_fa_per_24h": operating.get("fa_per_24h"),
                "locked_threshold": operating.get("threshold"),
                "test_event_sensitivity": score.get("event_sensitivity"),
                "test_fa_per_24h": score.get("fa_per_24h"),
                "test_true_positive_events": score.get("true_positive_events"),
                "test_false_alarm_events": score.get("false_alarm_events"),
                "test_truth_events": score.get("truth_events"),
                "test_median_delay_s": score.get("median_delay_s"),
                "test_evaluation_count": test.get("test_evaluation_count"),
                "model_sha256": sha256_file(model_path) if model_path.exists() else None,
                "test_probability_sha256": test.get("probability_sha256"),
                "validation_probability_sha256": validation.get("probability_sha256"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "split", "seed"])
    table.to_csv(RUN_ROOT / "baseline_results.csv", index=False)
    summary = (
        table.groupby("model")
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
    audit = {
        "namespace": "v3-compact-baselines",
        "expected_run_count": len(expected),
        "observed_run_count": len(rows),
        "problems": problems,
        "matrix_complete": len(rows) == len(expected) and not problems,
        "summary": summary.to_dict(orient="records"),
        "results_csv": str((RUN_ROOT / "baseline_results.csv").relative_to(ROOT)),
    }
    (RUN_ROOT / "baseline_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
