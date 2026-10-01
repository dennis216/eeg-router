"""Train the causal raw-EEG TUSZ pre-router.

Only the past 60-s, 16-channel, 128-Hz context is loaded here.  Expert
probabilities are intentionally absent from the dataset and model inputs;
they were used offline by script 151 to make future competence targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RawContextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.cache_root = cache_root
        self._last_path: str | None = None
        self._last_array: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path_key = str(row["cache_path"])
        if path_key != self._last_path:
            path = self.cache_root.parent.parent / path_key
            # cache_path is relative to the H6 report root, not raw_cache.
            if not path.exists():
                path = OUT / path_key
            self._last_array = np.load(path, allow_pickle=False)["x"]
            self._last_path = path_key
        assert self._last_array is not None
        x = self._last_array[int(row["cache_row"])].astype(np.float32, copy=False)
        targets = np.array([row[f"correct_{m}"] for m in MODELS], dtype=np.float32)
        weights = np.array([row[f"target_weight_{m}"] for m in MODELS], dtype=np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(targets),
            torch.tensor(float(row["reject_label"]), dtype=torch.float32),
            torch.tensor(float(row["future_has_seizure"]), dtype=torch.float32),
            torch.from_numpy(weights),
        )


class RawPreRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.competence_head = nn.Linear(128, 3)
        self.reject_head = nn.Linear(128, 1)
        self.seizure_risk_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (16, 7680):
            raise ValueError(f"expected raw context [B,16,7680], got {tuple(x.shape)}")
        representation = self.features(x).squeeze(-1)
        return {
            "competence_logits": self.competence_head(representation),
            "reject_logit": self.reject_head(representation).squeeze(-1),
            "seizure_risk_logit": self.seizure_risk_head(representation).squeeze(-1),
        }


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    patient_counts = frame.groupby("patient_id")["sample_id"].transform("count").to_numpy(dtype=float)
    values = np.ones(len(frame), dtype=float)
    for i, row in enumerate(frame.itertuples(index=False)):
        if int(row.reject_label):
            values[i] *= 5.0
        elif int(row.future_has_seizure):
            values[i] *= 2.0
        elif len(set(str(row.correct_mask))) > 1:
            values[i] *= 2.0
    return values / np.maximum(patient_counts, 1.0)


def _balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    per = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    result = []
    for col in range(targets.shape[1]):
        target = targets[:, col]
        positive = target.mean()
        if float(positive) <= 0.0 or float(positive) >= 1.0:
            result.append(per[:, col].mean())
        else:
            weights = torch.where(target > 0.5, 0.5 / positive, 0.5 / (1.0 - positive))
            result.append((per[:, col] * weights).mean())
    return torch.stack(result).mean()


def _head_grad_norm(model: RawPreRouter, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if prefix in name and parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="train")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    labels_path = args.labels.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.json"
    if progress_path.exists() and not args.resume:
        raise RuntimeError(f"refusing to overwrite existing router run: {output_dir}")
    frame = pd.read_parquet(labels_path)
    if frame.empty:
        raise RuntimeError("empty router label table")
    for required in ["sample_id", "patient_id", "cache_path", "cache_row", "correct_mask", "reject_label", "future_has_seizure"]:
        if required not in frame:
            raise KeyError(required)
    cache_root = args.cache_root.resolve()
    dataset = RawContextDataset(frame, cache_root)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(_sample_weights(frame), dtype=torch.double), num_samples=max(len(frame), args.batch_size), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the raw-EEG router")
    device = torch.device("cuda")
    model = RawPreRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_update = 0
    if args.resume:
        candidates = sorted(checkpoint_dir.glob("router_step_*.pt"))
        if candidates:
            checkpoint = torch.load(candidates[-1], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_update = int(checkpoint["update"])
    iterator = iter(loader)
    history: list[dict[str, object]] = []
    started = time.monotonic()
    for update in range(start_update + 1, args.updates + 1):
        try:
            x, correct, reject, seizure, target_weights = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, correct, reject, seizure, target_weights = next(iterator)
        x, correct, reject, seizure, target_weights = [value.to(device, non_blocking=True) for value in (x, correct, reject, seizure, target_weights)]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        competence_loss = _balanced_bce(outputs["competence_logits"], correct)
        # Preserve simultaneous-correct labels while making calibrated/delay
        # priority visible to the competence head.  Rows with no correct expert
        # (unroutable) do not contribute to this ranking term.
        valid = target_weights.sum(dim=1) > 0
        if bool(valid.any()):
            soft_target = target_weights[valid] / target_weights[valid].sum(dim=1, keepdim=True)
            priority_loss = -(soft_target * nn.functional.log_softmax(outputs["competence_logits"][valid], dim=1)).sum(dim=1).mean()
        else:
            priority_loss = competence_loss.new_zeros(())
        reject_loss = nn.functional.binary_cross_entropy_with_logits(outputs["reject_logit"], reject)
        risk_loss = nn.functional.binary_cross_entropy_with_logits(outputs["seizure_risk_logit"], seizure)
        total_loss = competence_loss + 0.05 * priority_loss + 0.2 * reject_loss + 0.1 * risk_loss
        total_loss.backward()
        gradients = {"competence": _head_grad_norm(model, "competence_head"), "reject": _head_grad_norm(model, "reject_head"), "seizure_risk": _head_grad_norm(model, "seizure_risk_head")}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {"update": update, "loss": float(total_loss.detach().cpu()), "competence_bce": float(competence_loss.detach().cpu()), "priority_ce": float(priority_loss.detach().cpu()), "reject_bce": float(reject_loss.detach().cpu()), "seizure_risk_bce": float(risk_loss.detach().cpu()), "reject_rate_batch": float(reject.mean().detach().cpu()), "seizure_rate_batch": float(seizure.mean().detach().cpu()), "gradient_norms": gradients, "elapsed_s": time.monotonic() - started}
        history.append(row)
        if update == 1 or update % 25 == 0 or update == args.updates:
            print(json.dumps({"status": "training", **row}, sort_keys=True), flush=True)
        if update % args.checkpoint_every == 0 or update == args.updates:
            checkpoint = checkpoint_dir / f"router_step_{update:06d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update, "architecture": "16x7680->Conv1d(32,64,128)+BN+ReLU+GAP->heads", "seed": args.seed, "labels_sha256": sha256(labels_path), "created_utc": datetime.now(timezone.utc).isoformat()}, checkpoint)
        progress = {"release_id": NAMESPACE, "partition": args.partition, "status": "running" if update < args.updates else "complete", "update": update, "updates_total": args.updates, "batch_size": args.batch_size, "seed": args.seed, "labels": str(labels_path.relative_to(ROOT)) if labels_path.is_relative_to(ROOT) else str(labels_path), "labels_sha256": sha256(labels_path), "cache_root": str(cache_root), "router_input": {"channels": 16, "sampling_hz": 128, "context_seconds": 60, "shape": [16, 7680], "expert_outputs_in_input": False}, "loss": {"competence_bce": 1.0, "priority_cross_entropy": 0.05, "reject_bce": 0.2, "seizure_risk_bce": 0.1}, "history_tail": history[-25:], "updated_utc": datetime.now(timezone.utc).isoformat()}
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "updates": args.updates, "checkpoint": str((checkpoint_dir / f"router_step_{args.updates:06d}.pt")), "smoke": args.smoke}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
