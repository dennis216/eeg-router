from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bfa.training.matrix import MatrixRun, content_addressed_run_id

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "experiment" / "v3_unified.yaml"
NAMESPACE = "v3-full"
MATRIX_ROOT = ROOT / "runs" / NAMESPACE
MATRIX_PATH = MATRIX_ROOT / "matrix.parquet"
IDENTITY_PATH = MATRIX_ROOT / "matrix_identity.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prioritised_unified_runs(
    config: dict[str, Any], *, config_sha256: str, code_commit: str
) -> list[MatrixRun]:
    experiment = config["experiment"]
    split_hashes = {
        split_seed: sha256_file(ROOT / "manifests" / "splits" / f"split_{split_seed}.json")
        for split_seed in experiment["split_seeds"]
    }

    def build(model: str, split_seed: int, model_seed: int) -> MatrixRun:
        return MatrixRun(
            suite="mandatory",
            configuration="unified",
            model=model,
            split_seed=split_seed,
            model_seed=model_seed,
            run_id=content_addressed_run_id(
                configuration="unified",
                model=model,
                split_seed=split_seed,
                model_seed=model_seed,
                config_sha256=config_sha256,
                split_sha256=split_hashes[split_seed],
                code_commit=code_commit,
            ),
        )

    model_order = ["tcn_gat", "cbramod", "singlem"]
    run_order: list[MatrixRun] = [build(model, 17, 17) for model in model_order]
    for seed in (42, 3407):
        run_order.extend(build(model, 17, seed) for model in model_order)
    for split_seed in (42, 3407, 2026, 777):
        for model_seed in (17, 42, 3407):
            run_order.extend(build(model, split_seed, model_seed) for model in model_order)
    if len(run_order) != 45 or len({run.run_id for run in run_order}) != 45:
        raise AssertionError("full v3 unified matrix must contain 45 unique runs")
    return run_order


def initial_status(runs: list[MatrixRun]) -> pd.DataFrame:
    rows = [
        {
            "priority": priority,
            "run_id": run.run_id,
            "suite": run.suite,
            "configuration": run.configuration,
            "model": run.model,
            "split_seed": run.split_seed,
            "model_seed": run.model_seed,
            "status": "pending",
            "attempts": 0,
            "started_at": None,
            "completed_at": None,
            "last_error": None,
        }
        for priority, run in enumerate(runs, start=1)
    ]
    return pd.DataFrame(rows)


def write_status(table: pd.DataFrame) -> None:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX_PATH.with_suffix(".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(MATRIX_PATH)


def load_or_create_status(runs: list[MatrixRun]) -> pd.DataFrame:
    expected = initial_status(runs)
    if not MATRIX_PATH.exists():
        write_status(expected)
        return expected
    current = pd.read_parquet(MATRIX_PATH)
    if set(current.run_id) != set(expected.run_id):
        raise RuntimeError("existing matrix state does not match the frozen 45-run plan")
    return (
        expected.drop(columns=["status", "attempts", "started_at", "completed_at", "last_error"])
        .merge(
            current[["run_id", "status", "attempts", "started_at", "completed_at", "last_error"]],
            on="run_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("priority", kind="stable")
        .reset_index(drop=True)
    )


def run_directory(run: MatrixRun) -> Path:
    return MATRIX_ROOT / "unified" / run.run_id


def verify_completed_run(run: MatrixRun) -> dict[str, Any]:
    run_dir = run_directory(run)
    required = [
        run_dir / "completed.json",
        run_dir / "run_manifest.json",
        run_dir / "threshold_lock.json",
        run_dir / "test" / "probabilities.parquet",
        run_dir / "test" / "metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("completed run is missing artifacts: " + ", ".join(missing))
    completed = json.loads((run_dir / "completed.json").read_text())
    lock = json.loads((run_dir / "threshold_lock.json").read_text())
    test = json.loads((run_dir / "test" / "metrics.json").read_text())
    checkpoint = run_dir / lock["checkpoint"]
    if sha256_file(checkpoint) != lock["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for {run.run_id}")
    if sha256_file(run_dir / "test" / "probabilities.parquet") != test["probability_sha256"]:
        raise RuntimeError(f"test-probability hash mismatch for {run.run_id}")
    if int(test["test_evaluation_count"]) != 1:
        raise RuntimeError(f"test count is not one for {run.run_id}")
    if completed["run_id"] != run.run_id:
        raise RuntimeError(f"completed run identity mismatch for {run.run_id}")
    return completed


def active_process(run: MatrixRun) -> bool:
    pattern = (
        "scripts/28_run_v3_unified.py --model "
        f"{run.model} --split-seed {run.split_seed} --model-seed {run.model_seed}"
    )
    check = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False, text=True)
    return check.returncode == 0


def cache_complete(model: str, config: dict[str, Any]) -> bool:
    recordings = pd.read_parquet(ROOT / config["inputs"]["recordings_manifest"])
    root = Path(config["inputs"]["foundation_feature_cache_root"]) / model
    return all(
        (root / Path(relative).with_suffix(".npy")).is_file()
        for relative in recordings.relative_path
    )


def signal_cache_complete(config: dict[str, Any]) -> bool:
    recordings = pd.read_parquet(ROOT / config["inputs"]["recordings_manifest"])
    root = Path(config["inputs"]["foundation_signal_cache_root"])
    return all(
        (root / model / Path(relative).with_suffix(".npy")).is_file()
        for model in ("singlem", "cbramod")
        for relative in recordings.relative_path
    )


def cache_builder_active() -> bool:
    check = subprocess.run(
        ["pgrep", "-f", "10_cache_pilot_official_views.py"],
        capture_output=True,
        check=False,
        text=True,
    )
    return check.returncode == 0


def ensure_feature_cache(model: str, config: dict[str, Any]) -> None:
    while not signal_cache_complete(config):
        if not cache_builder_active():
            raise RuntimeError(
                "v3 foundation signal cache is incomplete and its builder is not active"
            )
        print("V3_QUEUE_WAIT_SIGNAL_CACHE", flush=True)
        time.sleep(60)
    if cache_complete(model, config):
        return
    log_path = ROOT / "logs" / f"v3_feature_cache_{model}.log"
    command = [
        sys.executable,
        "scripts/13_cache_frozen_features.py",
        "--model",
        model,
        "--input-root",
        str(config["inputs"]["foundation_signal_cache_root"]),
        "--output-root",
        str(config["inputs"]["foundation_feature_cache_root"]),
        "--profile",
        "v3-official-noclip",
        "--config",
        "configs/experiment/v3_unified.yaml",
    ]
    print("V3_FEATURE_CACHE_START " + " ".join(command), flush=True)
    with log_path.open("a") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0 or not cache_complete(model, config):
        raise RuntimeError(f"v3 feature cache failed for {model}; see {log_path}")


def gpu_memory_gib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip().splitlines()[0]) / 1024.0


def tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return "log not created"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def set_status(table: pd.DataFrame, run: MatrixRun, status: str, **values: object) -> pd.DataFrame:
    mask = table.run_id == run.run_id
    if mask.sum() != 1:
        raise KeyError(run.run_id)
    table.loc[mask, "status"] = status
    for key, value in values.items():
        table.loc[mask, key] = value
    write_status(table)
    return table


def launch_run(run: MatrixRun, *, max_vram_gib: float) -> tuple[int, str]:
    memory = gpu_memory_gib()
    if memory > max_vram_gib:
        raise RuntimeError(
            f"GPU memory {memory:.2f} GiB exceeds queue limit {max_vram_gib:.2f} GiB"
        )
    log_path = ROOT / "logs" / f"v3_{run.run_id}.log"
    command = [
        sys.executable,
        "scripts/28_run_v3_unified.py",
        "--model",
        run.model,
        "--split-seed",
        str(run.split_seed),
        "--model-seed",
        str(run.model_seed),
        "--run-id",
        run.run_id,
    ]
    run_dir = run_directory(run)
    if run_dir.exists():
        command.append("--resume")
    print("V3_MATRIX_START " + " ".join(command), flush=True)
    with log_path.open("a") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return completed.returncode, tail(log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-vram-gib", type=float, default=28.0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_file(CONFIG_PATH)
    if IDENTITY_PATH.exists():
        identity = json.loads(IDENTITY_PATH.read_text())
        if identity["config_sha256"] != config_sha:
            raise RuntimeError("matrix configuration changed after identity was frozen")
    else:
        identity = {
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": config_sha,
            "code_commit": git_commit(),
            "frozen_at": utc_now(),
        }
        IDENTITY_PATH.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    runs = prioritised_unified_runs(
        config, config_sha256=config_sha, code_commit=identity["code_commit"]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "count": len(runs),
                    "identity": identity,
                    "first_nine": [run.__dict__ for run in runs[:9]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    state_existed = MATRIX_PATH.exists()
    table = load_or_create_status(runs)
    if not args.resume and state_existed:
        raise RuntimeError("matrix state exists; use --resume")
    for run in runs:
        row = table.loc[table.run_id == run.run_id].iloc[0]
        if row.status in {"completed", "blocked"}:
            continue
        while True:
            completed_path = run_directory(run) / "completed.json"
            if completed_path.exists():
                verify_completed_run(run)
                table = set_status(table, run, "completed", completed_at=utc_now(), last_error=None)
                print(f"V3_MATRIX_COMPLETE run_id={run.run_id}", flush=True)
                break
            if active_process(run):
                table = set_status(table, run, "running", started_at=utc_now())
                print(f"V3_MATRIX_WAIT_ACTIVE run_id={run.run_id}", flush=True)
                time.sleep(args.poll_seconds)
                continue
            current = table.loc[table.run_id == run.run_id].iloc[0]
            attempts = int(current.attempts)
            if attempts >= 2:
                table = set_status(
                    table,
                    run,
                    "blocked",
                    last_error="two attempts failed or the pre-existing run was incomplete",
                )
                print(f"V3_MATRIX_BLOCKED run_id={run.run_id}", flush=True)
                break
            if run.model in {"singlem", "cbramod"}:
                ensure_feature_cache(run.model, config)
            table = set_status(
                table,
                run,
                "running",
                attempts=attempts + 1,
                started_at=utc_now(),
                last_error=None,
            )
            returncode, log_tail = launch_run(run, max_vram_gib=args.max_vram_gib)
            if returncode == 0:
                continue
            table = set_status(table, run, "failed_once", last_error=log_tail)
            print(f"V3_MATRIX_FAILED_ONCE run_id={run.run_id}", flush=True)
    summary = table.status.value_counts(dropna=False).to_dict()
    print("V3_MATRIX_TERMINAL " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
