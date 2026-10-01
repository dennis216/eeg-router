from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from bfa.models.shared_head import SharedContextHead
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.training.dataset import CachedContextDataset
from bfa.training.sampler import PatientBalancedSampler


class GPUMonitor:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                utilization, memory_mib = result.stdout.strip().splitlines()[0].split(",")
                self.samples.append(
                    {
                        "elapsed_s": time.monotonic() - started,
                        "gpu_utilization_percent": float(utilization.strip()),
                        "gpu_memory_used_mib": float(memory_mib.strip()),
                    }
                )
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GPUMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _cuda_timed(operation: object) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()  # type: ignore[operator]
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("outputs/profiling/tcn_gat_200batch"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    split = json.loads(Path(f"manifests/splits/split_{args.split}.json").read_text())
    windows = pd.read_parquet("manifests/windows.parquet")
    train_index = windows[
        windows.patient.isin(split["train"]) & windows.train_eligible
    ].reset_index(drop=True)
    sampler = PatientBalancedSampler(
        train_index,
        batch_size=args.batch_size,
        positive_fraction=0.30,
        seed=args.seed,
        epoch_size=args.batches * args.batch_size,
    )
    dataset = CachedContextDataset(
        train_index, Path("/mnt/d/EEGData/bfa_cache_v3"), "tcn_gat"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    device = torch.device("cuda")
    encoder = MultiScaleTCNGAT().to(device)
    head = SharedContextHead().to(device)
    trainable = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    positives = float((train_index.label == 1).sum())
    negatives = float((train_index.label == 0).sum())
    pos_weight = torch.tensor(negatives / positives, device=device)
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    records = []
    optimizer.zero_grad(set_to_none=True)

    with GPUMonitor() as monitor:
        profile_started = time.monotonic()
        for batch_index in range(args.batches):
            batch_started = time.monotonic()
            batch = next(iterator)
            data_load_s = time.monotonic() - batch_started
            gpu_batch: dict[str, torch.Tensor] = {}

            def transfer() -> None:
                gpu_batch["x"] = batch["x"].to(device, non_blocking=True)
                gpu_batch["quality"] = batch["quality"].to(device, non_blocking=True)
                gpu_batch["y"] = batch["y"].to(device, non_blocking=True)

            cpu_to_gpu_s = _cuda_timed(transfer)
            state: dict[str, torch.Tensor] = {}

            def forward() -> None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embeddings = encoder.forward_window_sequence(
                        gpu_batch["x"], gpu_batch["quality"]
                    )
                    logits = head(embeddings)
                    state["loss"] = functional.binary_cross_entropy_with_logits(
                        logits, gpu_batch["y"], pos_weight=pos_weight
                    ) / args.accumulation

            forward_s = _cuda_timed(forward)
            backward_s = _cuda_timed(state["loss"].backward)
            optimizer_s = 0.0
            gradient_norm = float("nan")
            if (batch_index + 1) % args.accumulation == 0:
                def update() -> None:
                    nonlocal gradient_norm
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                optimizer_s = _cuda_timed(update)
            total_s = time.monotonic() - batch_started
            records.append(
                {
                    "batch": batch_index,
                    "data_load_s": data_load_s,
                    "cpu_to_gpu_s": cpu_to_gpu_s,
                    "forward_s": forward_s,
                    "backward_s": backward_s,
                    "optimizer_s": optimizer_s,
                    "total_s": total_s,
                    "loss_unscaled": float(state["loss"].detach()) * args.accumulation,
                    "positive_fraction": float(gpu_batch["y"].float().mean()),
                    "gradient_norm": gradient_norm,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            if (batch_index + 1) % 10 == 0:
                print(f"TCN_PROFILE_PROGRESS {batch_index + 1}/{args.batches}", flush=True)
        profile_total_s = time.monotonic() - profile_started

    args.output.mkdir(parents=True, exist_ok=True)
    batches = pd.DataFrame(records)
    gpu_samples = pd.DataFrame(monitor.samples)
    batches.to_parquet(args.output / "batches.parquet", index=False)
    gpu_samples.to_parquet(args.output / "gpu_samples.parquet", index=False)
    measured = batches[batches.batch >= args.warmup_batches]
    config = {
        "batches": args.batches,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulation,
        "effective_batch_size": args.batch_size * args.accumulation,
        "workers": args.workers,
        "warmup_batches": args.warmup_batches,
        "seed": args.seed,
        "split": args.split,
        "train_rows": len(train_index),
        "total_batches_per_full_epoch": int(np.ceil(len(train_index) / args.batch_size)),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    timings = {}
    for column in (
        "data_load_s",
        "cpu_to_gpu_s",
        "forward_s",
        "backward_s",
        "optimizer_s",
        "total_s",
    ):
        timings[column] = {
            "mean": float(measured[column].mean()),
            "p50": float(measured[column].quantile(0.50)),
            "p95": float(measured[column].quantile(0.95)),
        }
    summary = {
        "config": config,
        "config_sha256": config_hash,
        "profile_wall_s": profile_total_s,
        "timings": timings,
        "gpu_utilization_mean_percent": float(gpu_samples.gpu_utilization_percent.mean()),
        "gpu_utilization_p50_percent": float(gpu_samples.gpu_utilization_percent.quantile(0.5)),
        "gpu_utilization_below_50_fraction": float(
            (gpu_samples.gpu_utilization_percent < 50).mean()
        ),
        "gpu_memory_used_peak_mib_nvidia_smi": float(gpu_samples.gpu_memory_used_mib.max()),
        "torch_peak_allocated_gib": float(batches.peak_allocated_gib.max()),
        "torch_peak_reserved_gib": float(batches.peak_reserved_gib.max()),
        "estimated_full_epoch_hours": float(
            timings["total_s"]["mean"] * config["total_batches_per_full_epoch"] / 3600
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
