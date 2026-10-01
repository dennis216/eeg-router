"""Audit H7 patient-local fine-tuning outputs without reading EEG predictions.

This is a post-fit integrity check.  It verifies the immutable label manifest,
the 216 seed-17 run matrix, balanced fit sampling, checkpoint steps and the
absence of dev/holdout reads in training manifests.  It deliberately does not
run a model or choose a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
MODELS = ("singlem", "cbramod", "tcn_gat")
METHODS = ("head_only", "lora", "full")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="h7-patient-finetune-comparison-v1-formal-seed17-v2")
    args = parser.parse_args()
    out = ROOT / "outputs/reports" / args.namespace
    labels = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/labels"
    report: dict[str, Any] = {"release_id": "h7-patient-finetune-comparison-v1", "namespace": args.namespace, "created_utc": datetime.now(timezone.utc).isoformat(), "checks": {}}
    failures: list[str] = []
    label_manifest = labels / "manifest.json"
    if not label_manifest.exists():
        failures.append("missing label manifest")
    else:
        lm = json.loads(label_manifest.read_text())
        report["checks"]["label_manifest"] = {"status": lm.get("status"), "sha256": sha256(label_manifest), "units": lm.get("counts", {}).get("eligible_patient_detector_units"), "cross_phase_edfs": lm.get("counts", {}).get("cross_phase_edfs")}
        if lm.get("status") != "PASS" or lm.get("counts", {}).get("eligible_patient_detector_units") != 72:
            failures.append("label manifest status/count mismatch")
    queue_manifest_path = out / "queue_seed17_manifest.json"
    progress_path = out / "queue_seed17_progress.json"
    if not queue_manifest_path.exists():
        failures.append("missing queue manifest")
    else:
        queue = json.loads(queue_manifest_path.read_text())
        report["checks"]["queue"] = {"status": queue.get("status"), "expected_runs": len(queue.get("expected_runs", [])), "completed_runs": len(queue.get("completed_runs", []))}
        if queue.get("status") != "complete" or len(queue.get("expected_runs", [])) != 216:
            failures.append("queue is not complete 216-run matrix")
    if progress_path.exists():
        report["checks"]["progress"] = json.loads(progress_path.read_text())
    runs_root = out / "runs"
    run_paths = sorted(runs_root.glob("*/manifest.json")) if runs_root.exists() else []
    expected = 216
    complete = 0
    run_failures: list[dict[str, Any]] = []
    for manifest_path in run_paths:
        run_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            progress = json.loads((run_dir / "progress.json").read_text())
        except Exception as exc:
            run_failures.append({"run": run_dir.name, "error": f"unreadable manifest/progress: {exc}"})
            continue
        if progress.get("status") == "complete" and int(progress.get("update", -1)) == 1000:
            complete += 1
        checks = []
        if manifest.get("fit_only") is not True or manifest.get("dev_read") is not False or manifest.get("holdout_read") is not False:
            checks.append("partition-read flags")
        if manifest.get("updates") != 1000 or manifest.get("batch_size") != 64:
            checks.append("training contract")
        if not (run_dir / "checkpoint.pt").exists():
            checks.append("final checkpoint")
        for step in (100, 250, 500, 750, 1000):
            if not (run_dir / f"checkpoint_step_{step:04d}.pt").exists():
                checks.append(f"checkpoint {step}")
        history_path = run_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history and any(int(row.get("positive_n", -1)) != 32 or int(row.get("negative_n", -1)) != 32 for row in history):
                checks.append("unbalanced fit batch")
        else:
            checks.append("history")
        if checks:
            run_failures.append({"run": run_dir.name, "checks": checks})
    report["checks"]["runs"] = {"run_directories": len(run_paths), "complete_step1000": complete, "expected": expected, "failures": run_failures[:50]}
    if len(run_paths) not in (0, expected):
        failures.append(f"unexpected run count {len(run_paths)}")
    if len(run_paths) == expected and complete != expected:
        failures.append(f"only {complete}/{expected} runs complete")
    if run_failures:
        failures.append(f"{len(run_failures)} run integrity failures")
    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    atomic_json(out / "formal_seed17_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
