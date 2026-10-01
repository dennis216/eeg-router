from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
REPORT = ROOT / "outputs/reports" / NAMESPACE
RUNS = ROOT / "runs" / NAMESPACE


def load_v3_helpers():
    path = ROOT / "scripts/28_run_v3_unified.py"
    spec = importlib.util.spec_from_file_location("v3_unified_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_v3_helpers()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def partition_index(windows: pd.DataFrame, patients: list[str], *, training: bool) -> pd.DataFrame:
    mask = windows.patient.isin(patients)
    if training:
        mask &= windows.train_eligible
    else:
        mask &= ~windows.warmup
    out = windows.loc[mask].reset_index(drop=True)
    if out.empty or out.label.nunique(dropna=True) < 2:
        raise RuntimeError(f"partition has insufficient data: n={len(out)} labels={out.label.value_counts(dropna=False).to_dict()}")
    return out


def build_config() -> dict[str, Any]:
    return {
        "inputs": {
            "window_manifest": str(ROOT / "manifests/windows.parquet"),
            "recordings_manifest": str(ROOT / "manifests/recordings.parquet"),
            "seizures_manifest": str(ROOT / "manifests/seizures.parquet"),
            "tcn_cache_root": "/mnt/d/EEGData/bfa_cache_v3",
            "foundation_feature_cache_root": "/mnt/d/EEGData/bfa_features_v3_official_noclip",
            "quality_cache_root": "/mnt/d/EEGData/bfa_cache_v3",
        },
        "training": {
            "micro_batch_size": 16,
            "gradient_accumulation": 4,
            "target_positive_fraction": 0.30,
            "dataloader_workers": 4,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=range(5))
    ap.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.smoke and args.updates > 25:
        args.updates = 25
    split_path = REPORT / f"round_{args.round}_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text())
    run_name = f"training_smoke/round_{args.round}/{args.model}" if args.smoke else f"formal/round_{args.round}/{args.model}"
    run_dir = RUNS / run_name
    progress_path = run_dir / "progress.json"
    if (run_dir / "completed.json").exists() and not args.smoke:
        print(f"ALREADY_COMPLETE round={args.round} model={args.model}", flush=True)
        return
    if run_dir.exists() and not args.resume and not args.smoke:
        raise RuntimeError(f"run exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_config()
    windows = pd.read_parquet(ROOT / "manifests/windows.parquet")
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    train = partition_index(windows, split["train"], training=True)
    dev = partition_index(windows, split["validation"], training=False)
    test = partition_index(windows, split["test"], training=False)
    for frame, name in ((train, "train"), (dev, "router_dev"), (test, "final_test")):
        frame.to_parquet(run_dir / f"{name}_index.parquet", index=False)
    if not torch.cuda.is_available():
        raise RuntimeError("BIOT router experiment requires CUDA for expert training")
    v3.configure_determinism(args.seed)
    encoder = v3.build_encoder(args.model).cuda()
    head = v3.SharedContextHead().cuda()
    trainable = [p for m in (encoder, head) for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=1e-5)
    positives = float((train.label == 1).sum())
    negatives = float((train.label == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device="cuda")
    write_json(
        run_dir / "run_manifest.json",
        {
            "protocol_id": NAMESPACE,
            "round": args.round,
            "model": args.model,
            "seed": args.seed,
            "updates": args.updates,
            "smoke": args.smoke,
            "split_manifest": str(split_path.relative_to(ROOT)),
            "train_patients": split["train"],
            "router_dev_patients": split["validation"],
            "final_test_patients": split["test"],
            "train_rows": len(train),
            "router_dev_rows": len(dev),
            "final_test_rows": len(test),
            "threshold_source": "expert_train_only",
            "test_accessed": False,
            "checkpoint_reuse": False,
            "started_at_unix": time.time(),
        },
    )
    write_json(progress_path, {"stage": "training", "round": args.round, "model": args.model, "update": 0, "total_updates": args.updates})
    loader, sampler = v3.make_loader(
        train,
        args.model,
        config,
        model_seed=args.seed,
        split_seed=100 + args.round,
        epoch=1,
        updates=args.updates,
    )
    started = time.monotonic()
    loss, _ = v3.train_epoch(
        encoder,
        head,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        updates=args.updates,
        accumulation=4,
        pos_weight=pos_weight,
        gradient_clip_norm=1.0,
        epoch=1,
        total_epochs=1,
    )
    write_json(progress_path, {"stage": "training_complete", "round": args.round, "model": args.model, "update": args.updates, "total_updates": args.updates, "loss": loss, "elapsed_seconds": time.monotonic() - started})
    checkpoint = run_dir / "expert_checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "update": args.updates,
            "seed": args.seed,
            "round": args.round,
            "model": args.model,
            "protocol_id": NAMESPACE,
            "threshold_source": "expert_train_only",
            "split_sha256": sha256_file(split_path),
        },
        checkpoint,
    )
    (run_dir / "expert_checkpoint.pt.sha256").write_text(sha256_file(checkpoint) + "\n")
    if args.smoke:
        # Smoke only verifies the data and output path; it never evaluates final test.
        train_smoke = train.groupby("patient", sort=False).head(8).reset_index(drop=True)
        dev_smoke = dev.groupby("patient", sort=False).head(8).reset_index(drop=True)
        train_prob = v3.predict(encoder, head, train_smoke, args.model, config, batch_size=8, workers=0)
        dev_prob = v3.predict(encoder, head, dev_smoke, args.model, config, batch_size=8, workers=0)
        pd.DataFrame({"probability": train_prob}).to_parquet(run_dir / "smoke_train_probabilities.parquet", index=False)
        pd.DataFrame({"probability": dev_prob}).to_parquet(run_dir / "smoke_router_dev_probabilities.parquet", index=False)
        write_json(progress_path, {"stage": "smoke_complete", "round": args.round, "model": args.model, "update": args.updates, "test_accessed": False, "status": "PASS"})
        print(f"BIOT_EXPERT_SMOKE_OK round={args.round} model={args.model} updates={args.updates}", flush=True)
        return
    write_json(progress_path, {"stage": "predict_expert_train", "round": args.round, "model": args.model, "update": args.updates, "total_updates": args.updates})
    train_prob = v3.predict(encoder, head, train, args.model, config, batch_size=16, workers=4)
    train_table = v3.probability_table(train, train_prob)
    train_table.to_parquet(run_dir / "expert_train_probabilities.parquet", index=False)
    sweep, selected = v3.threshold_sweep(train_table, seizures, recordings)
    sweep.to_parquet(run_dir / "expert_train_threshold_sweep.parquet", index=False)
    threshold = float(selected["threshold"])
    write_json(
        run_dir / "threshold_lock.json",
        {
            "source": "expert_train_only",
            "threshold": threshold,
            "selected_event_operating_point": selected,
            "threshold_sweep_sha256": sha256_file(run_dir / "expert_train_threshold_sweep.parquet"),
        },
    )
    write_json(progress_path, {"stage": "predict_router_dev", "round": args.round, "model": args.model, "threshold": threshold, "update": args.updates, "test_accessed": False})
    dev_prob = v3.predict(encoder, head, dev, args.model, config, batch_size=16, workers=4)
    dev_table = v3.probability_table(dev, dev_prob)
    dev_table.to_parquet(run_dir / "router_dev_probabilities.parquet", index=False)
    dev_score = v3.score_probabilities(dev_table, seizures, recordings, threshold)
    write_json(run_dir / "router_dev_metrics.json", dev_score)
    write_json(progress_path, {"stage": "predict_final_test", "round": args.round, "model": args.model, "threshold": threshold, "update": args.updates, "test_accessed": True})
    test_prob = v3.predict(encoder, head, test, args.model, config, batch_size=16, workers=4)
    test_table = v3.probability_table(test, test_prob)
    test_table.to_parquet(run_dir / "final_test_probabilities.parquet", index=False)
    test_score = v3.score_probabilities(test_table, seizures, recordings, threshold)
    write_json(run_dir / "final_test_metrics.json", test_score)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    manifest.update({"test_accessed": True, "threshold": threshold, "checkpoint_sha256": sha256_file(checkpoint), "completed_at_unix": time.time()})
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(progress_path, {"stage": "complete", "round": args.round, "model": args.model, "threshold": threshold, "update": args.updates, "test_accessed": True, "status": "PASS"})
    write_json(run_dir / "completed.json", {"protocol_id": NAMESPACE, "round": args.round, "model": args.model, "status": "PASS", "test_evaluation_count": 1, "checkpoint_sha256": sha256_file(checkpoint), "completed_at_unix": time.time()})
    print(json.dumps({"round": args.round, "model": args.model, "threshold": threshold, "router_dev": dev_score, "final_test": test_score}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
