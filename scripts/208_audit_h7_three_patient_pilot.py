"""Audit the reduced H7 three-patient-per-detector pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/root/b_false_alarm_atlas")
METHODS = ("head_only", "lora", "full")
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.unlink(missing_ok=True)
    temporary.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    out = ROOT / "outputs/reports" / args.namespace
    report: dict[str, Any] = {
        "release_id": "h7-patient-finetune-comparison-v1",
        "namespace": args.namespace,
        "phase_names": {"fit": "training_set", "dev": "validation_set", "holdout": "test_set"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    failures: list[str] = []
    queue_path = out / "queue_seed17_manifest.json"
    subset_path = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/h7_three_patient_subset_manifest.json"
    label_path = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/labels/manifest.json"
    if not queue_path.exists():
        failures.append("missing queue manifest")
        queue = {}
    else:
        queue = json.loads(queue_path.read_text())
        report["checks"]["queue"] = {
            "status": queue.get("status"),
            "expected_runs": len(queue.get("expected_runs", [])),
            "completed_runs": len(queue.get("completed_runs", [])),
            "evaluate_after_each_run": queue.get("evaluate_after_each_run"),
        }
        if queue.get("status") != "complete" or len(queue.get("expected_runs", [])) != 27:
            failures.append("queue is not complete 27-run pilot")
    if not subset_path.exists():
        failures.append("missing three-patient subset manifest")
        subset = {}
    else:
        subset = json.loads(subset_path.read_text())
        units = subset.get("units", [])
        counts = {model: sum(1 for item in units if item.get("model") == model) for model in MODELS}
        report["checks"]["subset"] = {"sha256": sha256(subset_path), "units": len(units), "counts_by_model": counts}
        if len(units) != 9 or any(counts[model] != 3 for model in MODELS):
            failures.append("subset does not contain three units per model")
    if label_path.exists():
        report["checks"]["label_manifest_sha256"] = sha256(label_path)
    expected = queue.get("expected_runs", [])
    runs_root = out / "runs"
    evaluations = out / "per_run_evaluations"
    complete = 0
    eval_complete = 0
    run_failures: list[dict[str, Any]] = []
    for item in expected:
        name = f"{item['model']}__{item['method']}__{item['partition']}__{item['patient_id']}__seed{item['seed']}"
        run_dir = runs_root / name
        manifest_path = run_dir / "manifest.json"
        progress_path = run_dir / "progress.json"
        checks: list[str] = []
        try:
            manifest = json.loads(manifest_path.read_text())
            progress = json.loads(progress_path.read_text())
        except Exception as exc:
            run_failures.append({"run": name, "checks": [f"unreadable manifest/progress: {exc}"]})
            continue
        if progress.get("status") != "complete": checks.append("run not complete")
        if not (run_dir / "checkpoint.pt").exists(): checks.append("missing selected checkpoint")
        if manifest.get("training_set_read") is not True or manifest.get("validation_set_read") is not True or manifest.get("test_set_read") is not False:
            checks.append("dataset read flags")
        if manifest.get("requested_updates") != 2000: checks.append("requested max updates")
        if not (0 < int(manifest.get("updates", 0)) <= 2000): checks.append("invalid actual updates")
        if manifest.get("early_stopping", {}).get("enabled") is not True: checks.append("early stopping not enabled")
        if not int(manifest.get("best_validation_update", 0)) > 0: checks.append("missing validation-selected checkpoint")
        history_path = run_dir / "history.json"
        if not history_path.exists():
            checks.append("missing training history")
        else:
            history = json.loads(history_path.read_text())
            if any(int(row.get("positive_n", -1)) != 32 or int(row.get("negative_n", -1)) != 32 for row in history):
                checks.append("unbalanced training batch")
            if not any("validation_loss" in row for row in history):
                checks.append("missing validation loss history")
        if not checks: complete += 1
        evaluation_path = evaluations / f"{name}.json"
        if not evaluation_path.exists():
            checks.append("missing immediate validation/test evaluation")
        else:
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation.get("test_evaluation_count") != 1 or evaluation.get("test_used_for_selection") is not False:
                checks.append("test evaluation audit")
            else:
                eval_complete += 1
        if checks:
            run_failures.append({"run": name, "checks": checks})
    report["checks"]["runs"] = {"expected": len(expected), "complete": complete, "evaluations_complete": eval_complete, "failures": run_failures[:50]}
    if len(expected) == 27 and complete != 27:
        failures.append(f"only {complete}/27 runs complete")
    if len(expected) == 27 and eval_complete != 27:
        failures.append(f"only {eval_complete}/27 immediate evaluations complete")
    if run_failures:
        failures.append(f"{len(run_failures)} run audit issues")
    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    atomic_json(out / "pilot_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
