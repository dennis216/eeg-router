"""Resumable, immutable H7 fine-tuning queue.

The queue is intentionally separate from ``199_h7_patient_finetune_comparison``
so a formal run has one auditable manifest and one process-level lock.  It
never overwrites a completed or partial run directory.  Only fit rows are
read by the training worker; dev/holdout are handled by script 200 after the
queue has finished.

Example::

    H7_FT_NAMESPACE=h7-patient-finetune-comparison-v1-final \
      .venv/bin/python scripts/201_queue_h7_finetune_comparison.py \
      --seed 17 --updates 1000
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")
DEFAULT_NAMESPACE = "h7-patient-finetune-comparison-v1-final"
MODELS = ("singlem", "cbramod", "tcn_gat")
METHODS = ("head_only", "lora", "full")
SEEDS = (17, 42, 3407)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_subset_manifest(value: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    path_text = value or os.environ.get("H7_UNIT_SUBSET_MANIFEST")
    if not path_text:
        return None, None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("subset manifest must contain a non-empty units list")
    units: list[tuple[str, str, str]] = []
    for item in raw_units:
        if not isinstance(item, dict):
            raise ValueError("subset units must be objects")
        unit = (str(item["model"]), str(item["partition"]), str(item["patient_id"]))
        units.append(unit)
    if len(set(units)) != len(units):
        raise ValueError("subset manifest contains duplicate units")
    counts = {model: sum(1 for unit in units if unit[0] == model) for model in MODELS}
    if any(counts[model] != 3 for model in MODELS):
        raise ValueError(f"three-patient subset required per model; observed {counts}")
    normalized = {**payload, "units": [{"model": m, "partition": p, "patient_id": q} for m, p, q in units], "counts_by_model": counts}
    return path, normalized


def load_training_module():
    path = ROOT / "scripts/199_h7_patient_finetune_comparison.py"
    spec = importlib.util.spec_from_file_location("h7_finetune_queue_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device_from_arg(value: str | None) -> torch.device:
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def acquire_lock(path: Path, namespace: str, seed: int) -> None:
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "namespace": namespace, "seed": seed, "started_utc": now()}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"queue lock exists; refusing duplicate launch: {path}") from exc
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def release_lock(path: Path) -> None:
    # Keep a completion record rather than silently deleting the evidence.
    if path.exists():
        completed = path.with_name(path.name + ".released")
        os.replace(path, completed)


def run_queue(args: argparse.Namespace) -> None:
    namespace = os.environ.get("H7_FT_NAMESPACE", DEFAULT_NAMESPACE)
    if args.seed not in SEEDS:
        raise ValueError(args.seed)
    out = ROOT / "outputs/reports" / namespace
    # A preflight report may already have created the namespace directory.  It
    # is safe to reuse that directory only when it contains no experiment
    # outputs; any existing run/smoke artifact is treated as a hard stop.
    out.mkdir(parents=True, exist_ok=True)
    unexpected = [p for p in out.iterdir() if p.name not in {"no_duplicate_overlap_preflight.json", "formal_seed17.log"}]
    if unexpected:
        raise RuntimeError(f"refusing to reuse non-empty namespace: {unexpected}")
    lock = out / "queue.lock"
    acquire_lock(lock, namespace, args.seed)
    queue_manifest = out / f"queue_seed{args.seed}_manifest.json"
    progress_path = out / f"queue_seed{args.seed}_progress.json"
    worker = None
    try:
        # The required guard is run before the queue itself is entered.  The
        # caller should also retain its stdout in the experiment log.
        preflight = ROOT / "scripts/110_preflight_no_duplicate_overlap.py"
        import subprocess

        check = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(preflight), "--namespace", namespace, "--write-report"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (out / "preflight_stdout.json").write_text(check.stdout)
        if check.returncode != 0:
            raise RuntimeError(f"preflight failed; see {out / 'preflight_stdout.json'}")
        worker = load_training_module()
        labels = worker.load_label_table()
        eligible = (
            labels[labels.phase == "fit"]
            .groupby(["model", "partition", "patient_id"], sort=True)
            .size()
            .reset_index(name="n")
        )
        units = [
            (str(row.model), str(row.partition), str(row.patient_id))
            for row in eligible.itertuples(index=False)
        ]
        subset_path, subset_payload = resolve_subset_manifest(args.subset_manifest)
        if subset_payload is not None:
            requested = [
                (str(item["model"]), str(item["partition"]), str(item["patient_id"]))
                for item in subset_payload["units"]
            ]
            missing = sorted(set(requested) - set(units))
            if missing:
                raise RuntimeError(f"subset units are not eligible in the frozen label table: {missing}")
            units = [unit for unit in units if unit in set(requested)]
        expected = [
            {"model": model, "method": method, "partition": partition, "patient_id": patient, "seed": args.seed}
            for model, partition, patient in units
            for method in METHODS
        ]
        atomic_json(
            queue_manifest,
            {
                "release_id": "h7-patient-finetune-comparison-v1",
                "namespace": namespace,
                "stage": "formal_fit_queue",
                "status": "running",
                "seed": args.seed,
                "updates": args.updates,
                "models": MODELS,
                "methods": METHODS,
                "fit_only": False,
                "dev_read": True,
                "holdout_read": False,
                "training_set_read": True,
                "validation_set_read": True,
                "test_set_read": False,
                "phase_names": {"fit": "training_set", "dev": "validation_set", "holdout": "test_set"},
                "early_stopping": {
                    "enabled": True,
                    "monitor": "validation_balanced_bce",
                    "selection": "best_validation_checkpoint",
                    "max_updates": args.updates,
                    "every_updates": int(os.environ.get("H7_EARLY_STOP_EVERY", "100")),
                    "patience": int(os.environ.get("H7_EARLY_STOP_PATIENCE", "3")),
                },
                "evaluate_after_each_run": bool(args.evaluate_after_run),
                "label_manifest": "outputs/reports/h7-patient-finetune-comparison-v1/labels/manifest.json",
                "patient_subset_manifest": str(subset_path.relative_to(ROOT)) if subset_path is not None else None,
                "patient_subset_manifest_sha256": sha256(subset_path) if subset_path is not None else None,
                "patient_subset": subset_payload,
                "expected_runs": expected,
                "preflight_stdout": str((out / "preflight_stdout.json").relative_to(ROOT)),
                "created_utc": now(),
            },
        )
        atomic_json(progress_path, {"status": "running", "completed": 0, "total": len(expected), "current": None, "updated_utc": now()})
        done: list[dict[str, Any]] = []
        for index, item in enumerate(expected, start=1):
            run_dir = out / "runs" / f"{item['model']}__{item['method']}__{item['partition']}__{item['patient_id']}__seed{args.seed}"
            if run_dir.exists():
                progress = {}
                try:
                    progress = json.loads((run_dir / "progress.json").read_text())
                except Exception:
                    pass
                if progress.get("status") == "complete" and (run_dir / "checkpoint.pt").exists():
                    done.append({**item, "status": "already_complete", "run": str(run_dir.relative_to(ROOT))})
                    atomic_json(progress_path, {"status": "running", "completed": len(done), "total": len(expected), "current": item, "updated_utc": now()})
                    continue
                raise RuntimeError(f"refusing to overwrite incomplete run: {run_dir}")
            atomic_json(progress_path, {"status": "running", "completed": len(done), "total": len(expected), "current": item, "updated_utc": now()})
            started = time.monotonic()
            result = worker.run_one_train(
                item["model"], item["method"], item["partition"], item["patient_id"],
                args.seed, args.updates, device_from_arg(args.device), smoke=False,
            )
            if args.evaluate_after_run:
                evaluator = ROOT / "scripts/206_evaluate_h7_run.py"
                command = [
                    str(ROOT / ".venv/bin/python"), str(evaluator),
                    "--namespace", namespace,
                    "--run-dir", str(run_dir.relative_to(ROOT)),
                    "--device", str(args.device or ("cuda" if torch.cuda.is_available() else "cpu")),
                ]
                evaluation = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
                print(evaluation.stdout, end="", flush=True)
                if evaluation.returncode != 0:
                    print(evaluation.stderr, end="", flush=True)
                    raise RuntimeError(f"immediate validation/test evaluation failed for {run_dir.name}")
            done.append({**item, "status": "complete", "run": result["checkpoint"], "elapsed_s": time.monotonic() - started})
            atomic_json(progress_path, {"status": "running", "completed": len(done), "total": len(expected), "current": item, "updated_utc": now()})
            print(json.dumps({"completed": len(done), "total": len(expected), **item}, sort_keys=True), flush=True)
        atomic_json(queue_manifest, {**json.loads(queue_manifest.read_text()), "status": "complete", "completed_runs": done, "completed_utc": now()})
        atomic_json(progress_path, {"status": "complete", "completed": len(done), "total": len(expected), "current": None, "updated_utc": now()})
    except Exception as exc:
        atomic_json(progress_path, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "updated_utc": now()})
        try:
            payload = json.loads(queue_manifest.read_text())
        except Exception:
            payload = {}
        atomic_json(queue_manifest, {**payload, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "failed_utc": now()})
        raise
    finally:
        release_lock(lock)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--subset-manifest", default=None, help="JSON manifest listing exactly three patient units per model")
    parser.add_argument("--evaluate-after-run", action="store_true", help="run validation and test scoring immediately after each completed run")
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    run_queue(args)


if __name__ == "__main__":
    main()
