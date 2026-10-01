"""Generate outer-train discovery probabilities for standard H2/H3.

The formal runs persisted validation and outer-test probabilities.  H2/H3
require a separate discovery fit, so this script performs inference only on
the frozen outer-train patients using each run's final 5,000-update
checkpoint.  It never reads outer-test probabilities to choose a model,
threshold, scaler, PCA, cluster count, or classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLDS = tuple(range(5))
SEEDS = (17, 42, 3407)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner():
    runner_path = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/11_run_bsn_pilot.py")
    spec = importlib.util.spec_from_file_location("standard_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_one(runner, windows: pd.DataFrame, fold: int, model: str, seed: int, workers: int) -> None:
    split = json.loads((ROOT / "manifests" / "groupkfold_cv_v1" / f"fold_{fold}.json").read_text())
    train_index = windows[
        windows.patient.isin(split["train"]) & windows.train_eligible
    ].reset_index(drop=True)
    run_dir = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
    checkpoint_path = run_dir / "checkpoints" / "step_05000.pt"
    output_dir = run_dir / "train_discovery"
    output_path = output_dir / "probabilities_step_05000.parquet"
    manifest_path = output_dir / "inference_manifest.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            int(manifest.get("rows", -1)) == len(train_index)
            and manifest.get("checkpoint_sha256") == _sha256(checkpoint_path)
        ):
            print(f"DISCOVERY_INFERENCE_EXISTS fold={fold} model={model} seed={seed} rows={len(train_index)}", flush=True)
            return
    print(f"DISCOVERY_INFERENCE_START fold={fold} model={model} seed={seed} rows={len(train_index)}", flush=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = runner._model(model).cuda()
    head = runner.SharedContextHead().cuda()
    encoder.load_state_dict(state["encoder"], strict=False)
    head.load_state_dict(state["head"], strict=True)
    encoder.eval()
    head.eval()
    probabilities = runner._predict(
        encoder,
        head,
        train_index,
        model,
        batch_size=64,
        workers=workers,
    )
    table = runner._probability_table(train_index, probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    manifest = {
        "release_id": RUN_ID,
        "partition": "outer_train_discovery",
        "fold": fold,
        "seed": seed,
        "model": model,
        "optimizer_update": 5000,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "probability_sha256": _sha256(output_path),
        "rows": int(len(table)),
        "patients": sorted(table.patient.astype(str).unique().tolist()),
        "test_read": False,
        "threshold_fit": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    del encoder, head, state, probabilities, table
    torch.cuda.empty_cache()
    print(f"DISCOVERY_INFERENCE_DONE fold={fold} model={model} seed={seed} rows={len(train_index)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all 45 runs")
    args = parser.parse_args()
    os.environ.setdefault("BFA_ROOT", str(ROOT))
    os.environ.setdefault("BFA_CONFIG", str(ROOT / "configs" / "standard" / "v3_groupkfold_confirmatory_v1.yaml"))
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    runner = _load_runner()
    jobs = [(fold, model, seed) for fold in FOLDS for model in MODELS for seed in SEEDS]
    if args.max_runs:
        jobs = jobs[: args.max_runs]
    for fold, model, seed in jobs:
        _run_one(runner, windows, fold, model, seed, max(0, int(args.workers)))
    print(f"DISCOVERY_INFERENCE_COMPLETE jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
