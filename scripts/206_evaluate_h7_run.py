"""Immediate validation/test evaluation for one completed H7 pilot run.

This is deliberately a per-run diagnostic.  The training weights are selected
using the validation set only; the test set is then evaluated once with that
frozen threshold.  The resulting test numbers are never fed back to training,
patient selection, or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_scorer(namespace: str):
    os.environ["H7_FT_NAMESPACE"] = namespace
    path = ROOT / "scripts/200_h7_score_and_gate.py"
    spec = importlib.util.spec_from_file_location("h7_immediate_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not (run_dir / "checkpoint.pt").exists():
        raise FileNotFoundError(run_dir / "checkpoint.pt")
    output = ROOT / "outputs/reports" / args.namespace / "per_run_evaluations" / f"{run_dir.name}.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing evaluation: {output}")

    parts = run_dir.name.split("__")
    if len(parts) != 5 or not parts[-1].startswith("seed"):
        raise ValueError(f"unexpected run directory name: {run_dir.name}")
    model, method, partition, patient, seed_text = parts
    seed = int(seed_text.removeprefix("seed"))
    scorer = load_scorer(args.namespace)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    records = scorer.source_records()
    source_threshold = scorer.source_threshold(model)

    # The adapted checkpoint was selected from validation loss.  Use the same
    # fixed run checkpoint for both validation and test prediction.
    validation = scorer.predict_run(run_dir, "dev", device, batch_size=args.batch_size)
    test = scorer.predict_run(run_dir, "holdout", device, batch_size=args.batch_size)
    frozen_validation = scorer.predict_unit(
        model, "head_only", partition, patient, "dev", device,
        scorer.source_checkpoint(model), batch_size=args.batch_size,
        output_method="frozen_original", seed=seed,
    )
    frozen_metric, _ = scorer.score_table(frozen_validation, records, partition, patient, "dev", source_threshold)
    candidates: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metric, _ = scorer.score_table(validation, records, partition, patient, "dev", float(threshold))
        if metric["event_sensitivity"] >= frozen_metric["event_sensitivity"] - 0.03:
            candidates.append(metric)
    selected = min(candidates, key=lambda row: (row["fa_per_24h"], -row["threshold"])) if candidates else None
    threshold = float(selected["threshold"] if selected is not None else source_threshold)
    validation_metric, _ = scorer.score_table(validation, records, partition, patient, "dev", threshold)
    test_metric, _ = scorer.score_table(test, records, partition, patient, "holdout", threshold)

    payload = {
        "release_id": "h7-patient-finetune-comparison-v1",
        "namespace": args.namespace,
        "run": run_dir.name,
        "model": model,
        "method": method,
        "partition": partition,
        "patient_id": patient,
        "seed": seed,
        "phase_names": {"fit": "training_set", "dev": "validation_set", "holdout": "test_set"},
        "evaluation_type": "immediate_per_run_diagnostic",
        "checkpoint": str((run_dir / "checkpoint.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256(run_dir / "checkpoint.pt"),
        "threshold": threshold,
        "threshold_source": "validation_set_only" if selected is not None else "frozen_source_validation_only_fallback",
        "validation_gate_passed": selected is not None,
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "frozen_validation": {k: v for k, v in frozen_metric.items() if k != "patient_metrics"},
        "validation": {k: v for k, v in validation_metric.items() if k != "patient_metrics"},
        "test": {k: v for k, v in test_metric.items() if k != "patient_metrics"},
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json(output, payload)
    print(json.dumps({"status": "complete", "run": run_dir.name, "validation": payload["validation"], "test": payload["test"], "threshold": threshold}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
