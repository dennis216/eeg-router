from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from bfa.same_window_moe import ACTIONS, EXPERTS, aggregate_components, eventize_common, score_events, threshold_normalize


ROOT = Path("/root/b_false_alarm_atlas")
DEFAULT_EMBEDDINGS = Path("/mnt/d/EEGData/same_window_router_first_moe_v1/biot_embeddings")
DEFAULT_OUTPUT = ROOT / "outputs/reports/same-window-router-first-moe-v1/router_training"
MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/dev_manifest.json"
TARGET_COLUMNS = [f"policy_target_{action}" for action in ACTIONS]


class EmbeddingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, embeddings: np.memmap) -> None:
        self.frame = frame.reset_index(drop=True)
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        x = np.asarray(self.embeddings[int(row.embedding_row_id)], dtype=np.float32).copy()
        target = row[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(target), int(index)


class HierarchicalBatchSampler(Sampler[list[int]]):
    """50/50 seizure/background with bounded expert and reject dominance."""
    def __init__(self, frame: pd.DataFrame, batch_size: int, batches: int, seed: int) -> None:
        if batch_size % 4:
            raise ValueError("batch size must be divisible by four")
        self.frame = frame.reset_index(drop=True)
        self.batch_size, self.batches, self.seed = batch_size, batches, seed
        self.patient_frequency = self.frame.patient_id.astype(str).value_counts().to_dict()
        self.groups: dict[tuple[str, int], np.ndarray] = {}
        for kind, mask in {
            "seizure": self.frame.label.eq(1),
            "background_nonreject": self.frame.label.eq(0) & ~self.frame.outcome_group.eq("background_all_fp"),
        }.items():
            for action in range(3):
                self.groups[(kind, action)] = np.flatnonzero((mask & self.frame.policy_action_id.eq(action)).to_numpy())
        self.reject = np.flatnonzero(self.frame.outcome_group.eq("background_all_fp").to_numpy())
        if not len(self.reject) or not any(len(self.groups[("seizure", action)]) for action in range(3)):
            raise RuntimeError("hierarchical sampler lacks seizure or background_all_fp support")

    def __len__(self) -> int:
        return self.batches

    def _choice(self, rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
        if not len(pool):
            return np.empty(0, dtype=np.int64)
        patients = self.frame.iloc[pool].patient_id.astype(str)
        weights = np.array([1.0 / self.patient_frequency[p] for p in patients], dtype=float)
        weights /= weights.sum()
        return rng.choice(pool, size=n, replace=True, p=weights)

    def _balanced_experts(self, rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
        available = [action for action in range(3) if len(self.groups[(kind, action)])]
        base, remainder = divmod(n, len(available))
        pieces = []
        for position, action in enumerate(available):
            pieces.append(self._choice(rng, self.groups[(kind, action)], base + int(position < remainder)))
        return np.concatenate(pieces)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        quarter = self.batch_size // 4
        for _ in range(self.batches):
            seizure = self._balanced_experts(rng, "seizure", self.batch_size // 2)
            reject = self._choice(rng, self.reject, quarter)
            background = self._balanced_experts(rng, "background_nonreject", quarter)
            batch = np.concatenate([seizure, reject, background])
            rng.shuffle(batch)
            yield batch.tolist()


class RouterHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def load_bundle(root: Path, partition: str) -> tuple[pd.DataFrame, np.memmap]:
    directory = root / partition
    frame = pd.read_parquet(directory / "index.parquet")
    frame = frame[frame.policy_action_eligible.astype(bool)].reset_index(drop=True)
    if any("future" in column.lower() for column in frame.columns):
        raise RuntimeError("future-policy columns are forbidden")
    shape = (int(pd.read_json(directory / "manifest.json", typ="series")["rows"]), 256)
    embeddings = np.memmap(directory / "embeddings.float16.memmap", mode="r", dtype=np.float16, shape=shape)
    return frame, embeddings


@torch.inference_mode()
def infer(model: RouterHead, frame: pd.DataFrame, embeddings: np.memmap, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    ids = frame.embedding_row_id.to_numpy(np.int64)
    output = []
    for start in range(0, len(ids), batch_size):
        x = np.asarray(embeddings[ids[start : start + batch_size]], dtype=np.float32)
        output.append(torch.softmax(model(torch.from_numpy(x).to(device)), dim=1).cpu().numpy())
    return np.concatenate(output)


def truths(record: dict[str, object]) -> list[tuple[float, float]]:
    result = []
    for item in record.get("seizure_events", []):
        result.append((float(item.get("start_s", item.get("start", 0))), float(item.get("end_s", item.get("end", 0)))) if isinstance(item, dict) else (float(item[0]), float(item[1])))
    return result


def event_metrics(frame: pd.DataFrame, actions: np.ndarray) -> dict[str, float | int]:
    manifest = json.loads(MANIFEST.read_text())
    records = {str(record["relative_edf"]): record for record in manifest["records"]}
    parts = []
    score = np.zeros(len(frame), dtype=float)
    for action_id, model_name in enumerate(EXPERTS):
        mask = actions == action_id
        score[mask] = threshold_normalize(frame.loc[mask, f"probability_{model_name}"].to_numpy(float), frame.loc[mask, f"threshold_{model_name}"].to_numpy(float))
    for recording, positions in frame.groupby("recording", sort=False).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        order = index[np.argsort(frame.iloc[index].window_end_s.to_numpy(float))]
        times = frame.iloc[order].window_end_s.to_numpy(float)
        events = eventize_common(times, score[order])
        record = records[str(recording)]
        parts.append(score_events(events, truths(record), float(times.min()), float(record["duration_s"])))
    return aggregate_components(parts)


def selection_key(metric: dict[str, float | int], best_fixed_sensitivity: float) -> tuple:
    feasible = float(metric["sensitivity"]) >= best_fixed_sensitivity - 0.03
    if feasible:
        return (0, float(metric["false_alarm_time_min_per_24h"]), float(metric["fa_per_24h"]), -float(metric["sensitivity"]))
    return (1, -float(metric["sensitivity"]), float(metric["false_alarm_time_min_per_24h"]), float(metric["fa_per_24h"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batches-per-epoch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    torch.backends.cuda.matmul.allow_tf32 = True
    train, train_embeddings = load_bundle(args.embedding_root, "train")
    val, val_embeddings = load_bundle(args.embedding_root, "val")
    fixed = []
    for action_id, model_name in enumerate(EXPERTS):
        fixed.append((model_name, event_metrics(val, np.full(len(val), action_id, dtype=np.int8))))
    best_fixed_name, best_fixed = min(
        [item for item in fixed if float(item[1]["sensitivity"]) >= max(float(x[1]["sensitivity"]) for x in fixed) - 0.03],
        key=lambda item: (float(item[1]["false_alarm_time_min_per_24h"]), float(item[1]["fa_per_24h"])),
    )
    device = torch.device("cuda")
    model = RouterHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    sampler = HierarchicalBatchSampler(train, args.batch_size, args.batches_per_epoch, 3407)
    loader = DataLoader(EmbeddingDataset(train, train_embeddings), batch_sampler=sampler, num_workers=0, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_key, stale = None, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; rows = 0
        for x, target, _ in loader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = soft_cross_entropy(model(x), target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(x); rows += len(x)
        scheduler.step()
        probability = infer(model, val, val_embeddings, device, max(args.batch_size, 4096))
        action = probability.argmax(axis=1).astype(np.int8)
        metric = event_metrics(val, action)
        row = {"epoch": epoch, "train_loss": total_loss / rows, **metric, "action_distribution": {ACTIONS[i]: int((action == i).sum()) for i in range(4)}, "reject_fraction": float((action == 3).mean()), "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        key = selection_key(metric, float(best_fixed["sensitivity"]))
        if best_key is None or key < best_key:
            best_key, stale = key, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metric": metric, "best_fixed_expert": best_fixed_name, "best_fixed_metric": best_fixed}, args.output / "best.pt")
            pd.DataFrame({"embedding_row_id": val.embedding_row_id, "predicted_action_id": action, **{f"router_probability_{name}": probability[:, i] for i, name in enumerate(ACTIONS)}}).to_parquet(args.output / "best_val_predictions.parquet", index=False)
        else:
            stale += 1
        print(json.dumps(row, sort_keys=True), flush=True)
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    (args.output / "completed.json").write_text(json.dumps({"status": "PASS", "epochs": len(history), "best_fixed_expert": best_fixed_name, "best_fixed_metric": best_fixed, "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
