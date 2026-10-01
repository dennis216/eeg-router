from __future__ import annotations

"""Run compact DeepConvNet and ShallowConvNet baselines on the frozen CHB matrix."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
BASELINE61_PATH = ROOT / "scripts" / "61_run_baseline.py"
_spec = importlib.util.spec_from_file_location("bfa_baseline61", BASELINE61_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot import baseline helpers from {BASELINE61_PATH}")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

RawWindowDataset = _baseline.RawWindowDataset
PatientBalancedSampler = _baseline.PatientBalancedSampler
balanced_smoke = _baseline.balanced_smoke
evaluate_test = _baseline.evaluate_test
evaluate_validation = _baseline.evaluate_validation
indexes_for_split = _baseline.indexes_for_split
probability_table = _baseline.probability_table
sha256_file = _baseline.sha256_file
configure_seed = _baseline.configure_seed

CONFIG_PATH = ROOT / "configs" / "experiment" / "v3_unified.yaml"
NAMESPACE = "v3-compact-conv-baselines"


class DeepConvNet(nn.Module):
    """Four-block DeepConvNet-style EEG classifier."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.unsqueeze(1))
        return self.classifier(self.pool(x).flatten(1)).squeeze(-1)


class SquareLog(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x.square(), min=1e-4))


class ShallowConvNet(nn.Module):
    """ShallowConvNet/FBCSP-style temporal-spatial EEG classifier."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False),
            nn.Conv2d(40, 40, kernel_size=(channels, 1), bias=False),
            nn.BatchNorm2d(40),
            SquareLog(),
            nn.AvgPool2d((1, 75)),
            nn.Dropout(0.5),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.unsqueeze(1))
        return self.classifier(self.pool(x).flatten(1)).squeeze(-1)


def model_for(name: str) -> nn.Module:
    if name == "deepconvnet":
        return DeepConvNet()
    if name == "shallowconvnet":
        return ShallowConvNet()
    raise ValueError(name)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def predict(
    model: nn.Module,
    index: pd.DataFrame,
    cache_root: Path,
    workers: int,
) -> np.ndarray:
    dataset = RawWindowDataset(index, cache_root, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output.append(torch.sigmoid(model(batch["x"].cuda(non_blocking=True))).float().cpu().numpy())
            if batch_number % 500 == 0 or batch_number == len(loader):
                print(f"CONV_EVAL_PROGRESS batches={batch_number}/{len(loader)}", flush=True)
    return np.concatenate(output)


def run(
    model_name: str,
    split_seed: int,
    model_seed: int,
    updates: int,
    workers: int,
    smoke: bool,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    split_path = ROOT / "manifests" / "splits" / f"split_{split_seed}.json"
    split = json.loads(split_path.read_text())
    windows = pd.read_parquet(ROOT / config["inputs"]["window_manifest"])
    recordings = pd.read_parquet(ROOT / config["inputs"]["recordings_manifest"])
    seizures = pd.read_parquet(ROOT / config["inputs"]["seizures_manifest"])
    train_index, validation_index, test_index = indexes_for_split(windows, split)
    if smoke:
        validation_index = balanced_smoke(validation_index)
    effective_updates = updates
    run_id = f"{model_name}_split{split_seed}_seed{model_seed}"
    namespace = NAMESPACE + ("-smoke" if smoke else "")
    run_dir = ROOT / "runs" / "baselines" / namespace / model_name / run_id
    if (run_dir / "completed.json").exists():
        print(f"CONV_ALREADY_COMPLETE run_id={run_id}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory exists and is incomplete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(config["inputs"]["tcn_cache_root"])
    code_path = Path(__file__)
    manifest: dict[str, Any] = {
        "protocol": config["protocol"],
        "addendum": "v3_compact_conv_baselines",
        "run_id": run_id,
        "model": model_name,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "split_manifest": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256_file(split_path),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "code": str(code_path.relative_to(ROOT)),
        "code_sha256": sha256_file(code_path),
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "updates": effective_updates,
        "test_accessed": False,
        "smoke": smoke,
        "normalization": "per-channel window z-score, clip [-8,8]",
        "threshold_policy": "validation-only target sensitivity 0.80; unchanged v3 eventization",
        "started_at_unix": time.time(),
    }
    save_json(run_dir / "run_manifest.json", manifest)
    print("CONV_START " + json.dumps({"model": model_name, "split": split_seed, "seed": model_seed, "updates": effective_updates, "train_rows": len(train_index), "validation_rows": len(validation_index), "test_rows": len(test_index)}, sort_keys=True), flush=True)

    configure_seed(model_seed)
    model = model_for(model_name).cuda()
    sampler = PatientBalancedSampler(
        train_index,
        batch_size=16,
        positive_fraction=0.30,
        seed=model_seed * 1_000_000 + split_seed * 1_000,
        epoch_size=effective_updates * 4 * 16,
    )
    loader = DataLoader(
        RawWindowDataset(train_index, cache_root, normalize=True),
        batch_size=16,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    positives = float((train_index.label == 1).sum())
    negatives = float((train_index.label == 0).sum())
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_updates, eta_min=1e-5)
    iterator = iter(loader)
    accumulation = 4
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    model.train()
    for update in range(1, effective_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for _ in range(accumulation):
            batch = next(iterator)
            x = batch["x"].cuda(non_blocking=True)
            y = batch["y"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                raw_loss = functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
                (raw_loss / accumulation).backward()
            losses.append(float(raw_loss.detach().cpu()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite convolutional baseline gradient")
        optimizer.step()
        scheduler.step()
        if update % 25 == 0 or update == effective_updates:
            row = {"update": update, "loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started, "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            history.append(row)
            print("CONV_TRAIN_PROGRESS " + json.dumps(row, sort_keys=True), flush=True)
    checkpoint = run_dir / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "updates": effective_updates, "manifest": manifest}, checkpoint)
    (checkpoint.with_suffix(".pt.sha256")).write_text(sha256_file(checkpoint) + "\n")
    save_json(run_dir / "training_history.json", {"checkpoints": history})

    validation_table = probability_table(validation_index, predict(model, validation_index, cache_root, workers))
    validation_metrics = evaluate_validation(validation_table, seizures, recordings, run_dir)
    threshold = float(validation_metrics["selected_event_operating_point"]["threshold"])
    save_json(
        run_dir / "threshold_lock.json",
        {
            "source": "validation_only",
            "threshold": threshold,
            "selected_event_operating_point": validation_metrics["selected_event_operating_point"],
            "checkpoint": str(checkpoint.relative_to(run_dir)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "validation_probability_sha256": validation_metrics["probability_sha256"],
        },
    )
    if smoke:
        save_json(run_dir / "completed.json", {"smoke": True, "validation": validation_metrics})
        print(f"CONV_SMOKE_OK model={model_name} updates={effective_updates}", flush=True)
        return
    manifest["test_accessed"] = True
    manifest["validation_threshold"] = threshold
    save_json(run_dir / "run_manifest.json", manifest)
    test_table = probability_table(test_index, predict(model, test_index, cache_root, workers))
    test_metrics = evaluate_test(test_table, seizures, recordings, run_dir, threshold)
    save_json(
        run_dir / "completed.json",
        {"validation": validation_metrics, "test": test_metrics, "checkpoint_sha256": sha256_file(checkpoint)},
    )
    print("CONV_COMPLETE " + json.dumps({"model": model_name, "split": split_seed, "seed": model_seed, "test": test_metrics}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["deepconvnet", "shallowconvnet"], required=True)
    parser.add_argument("--split-seed", type=int, choices=[17, 42], required=True)
    parser.add_argument("--model-seed", type=int, choices=[17, 42, 3407], required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("convolutional baselines require CUDA")
    run(args.model, args.split_seed, args.model_seed, args.smoke_updates or args.updates, args.workers, args.smoke_updates > 0)


if __name__ == "__main__":
    main()
