"""Run final-checkpoint inference on the frozen H2/H3 discovery subset."""

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
SUBSET_ID = "outer-train-discovery-subset-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner():
    runner_path = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/11_run_bsn_pilot.py")
    spec = importlib.util.spec_from_file_location("standard_runner_subset", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_one(runner, fold: int, model: str, seed: int, workers: int) -> None:
    subset_path = ROOT / "manifests" / "groupkfold_cv_v1" / f"discovery_subset_fold_{fold}.parquet"
    subset = pd.read_parquet(subset_path)
    run_dir = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
    checkpoint_path = run_dir / "checkpoints" / "step_05000.pt"
    output_dir = run_dir / "train_discovery_subset"
    output_path = output_dir / "probabilities_step_05000.parquet"
    manifest_path = output_dir / "inference_manifest.json"
    if output_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if (
            int(existing.get("rows", -1)) == len(subset)
            and existing.get("checkpoint_sha256") == _sha256(checkpoint_path)
            and existing.get("subset_sha256") == _sha256(subset_path)
        ):
            print(f"SUBSET_INFERENCE_EXISTS fold={fold} model={model} seed={seed} rows={len(subset)}", flush=True)
            return
    print(f"SUBSET_INFERENCE_START fold={fold} model={model} seed={seed} rows={len(subset)}", flush=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = runner._model(model).cuda()
    head = runner.SharedContextHead().cuda()
    encoder.load_state_dict(state["encoder"], strict=False)
    head.load_state_dict(state["head"], strict=True)
    encoder.eval()
    head.eval()
    probabilities = runner._predict(encoder, head, subset, model, batch_size=64, workers=workers)
    table = runner._probability_table(subset, probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    metadata = json.loads((ROOT / "manifests" / "groupkfold_cv_v1" / f"discovery_subset_fold_{fold}.json").read_text())
    manifest = {
        "release_id": RUN_ID,
        "subset_id": SUBSET_ID,
        "partition": "outer_train_discovery_subset",
        "fold": fold,
        "seed": seed,
        "model": model,
        "optimizer_update": 5000,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "subset_sha256": _sha256(subset_path),
        "probability_sha256": _sha256(output_path),
        "rows": int(len(table)),
        "patients": sorted(table.patient.astype(str).unique().tolist()),
        "subset_rows_manifest": int(metadata["rows"]),
        "test_read": False,
        "threshold_fit": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    del encoder, head, state, probabilities, table
    torch.cuda.empty_cache()
    print(f"SUBSET_INFERENCE_DONE fold={fold} model={model} seed={seed} rows={len(subset)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("BFA_ROOT", str(ROOT))
    os.environ.setdefault("BFA_CONFIG", str(ROOT / "configs" / "standard" / "v3_groupkfold_confirmatory_v1.yaml"))
    runner = _load_runner()
    jobs = [(fold, model, seed) for fold in FOLDS for model in MODELS for seed in SEEDS]
    if args.max_runs:
        jobs = jobs[: args.max_runs]
    for fold, model, seed in jobs:
        _run_one(runner, fold, model, seed, max(0, int(args.workers)))
    print(f"SUBSET_INFERENCE_COMPLETE jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
