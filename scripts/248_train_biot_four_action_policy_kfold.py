"""Train the four-action policy head on frozen BIOT embeddings.

This is the policy-only method agreed for the current experiment:

* frozen BIOT/EEG-PREST encoder, followed by the original small MLP router;
* four actions: SingleM, CBraMod, TCN-GAT and Reject;
* no risk target and no risk head;
* one seed (3407), patient-grouped outer K-fold and an inner validation split;
* inverse-frequency balanced action sampling so the head cannot minimize loss
  by always emitting the dominant natural action;
* early stopping on natural validation macro-F1, with loss and per-action
  recall/precision recorded at every epoch.

The outer MIT folds reuse the frozen project GroupKFold manifest.  TUSZ uses
the same GroupKFold + GroupShuffleSplit construction on official Train
patients; official Dev is never used for early stopping and is reported only
as a locked external check after all five fold models are trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path("/root/b_false_alarm_atlas")
EMBED_ROOT = Path("/mnt/d/EEGData/bfa_biot_policy_embeddings_v1")
OUTPUT_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v3-four-action-biot-kfold"
MIT_CV = ROOT / "manifests/groupkfold_cv_v1/cv_manifest.json"
SEED = 3407
ACTIONS = ("singlem", "cbramod", "tcn_gat", "reject")
TARGET_COLUMNS = [f"policy_target_{name}" for name in ACTIONS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, memmap: np.memmap, embedding_ids: np.ndarray, targets: np.ndarray, action_ids: np.ndarray) -> None:
        self.memmap = memmap
        self.embedding_ids = np.asarray(embedding_ids, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.action_ids = np.asarray(action_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.embedding_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = np.asarray(self.memmap[int(self.embedding_ids[index])], dtype=np.float32)
        return torch.from_numpy(row.copy()), torch.from_numpy(self.targets[index]), torch.tensor(self.action_ids[index], dtype=torch.long)


class RouterHead(torch.nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(256, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def load_embedding_bundle(dataset: str) -> tuple[pd.DataFrame, np.memmap]:
    directory = EMBED_ROOT / dataset
    manifest_path = directory / "manifest.json"
    index_path = directory / "index.parquet"
    memmap_path = directory / "embeddings.float16.memmap"
    for path in (manifest_path, index_path, memmap_path):
        if not path.exists():
            raise FileNotFoundError(f"embedding artifact is not ready: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or int(manifest.get("embedding_dim", 0)) != 256:
        raise RuntimeError(f"invalid embedding manifest: {manifest_path}")
    frame = pd.read_parquet(index_path)
    if "embedding_row_id" not in frame.columns or not frame.embedding_row_id.is_unique:
        raise RuntimeError(f"invalid embedding index: {index_path}")
    rows = len(frame)
    memmap = np.memmap(memmap_path, mode="r", dtype=np.float16, shape=(rows, 256))
    return frame, memmap


def _patient_folds_mit() -> list[dict[str, object]]:
    payload = json.loads(MIT_CV.read_text())
    if len(payload.get("folds", [])) != 5:
        raise RuntimeError("MIT CV manifest must contain five folds")
    return payload["folds"]


def _patient_folds_tusz(frame: pd.DataFrame) -> list[dict[str, object]]:
    train_patients = np.asarray(sorted(frame.loc[frame.partition.eq("train"), "patient_id"].astype(str).unique()), dtype=object)
    if len(train_patients) < 5:
        raise RuntimeError("TUSZ Train has fewer than five patients")
    patient_frame = pd.DataFrame({"patient_id": train_patients})
    outer = GroupKFold(n_splits=5)
    folds: list[dict[str, object]] = []
    groups = patient_frame.patient_id.to_numpy()
    for fold, (development_idx, test_idx) in enumerate(outer.split(patient_frame, groups=groups)):
        development = patient_frame.iloc[development_idx].reset_index(drop=True)
        test = sorted(patient_frame.iloc[test_idx].patient_id.astype(str).tolist())
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + fold)
        train_idx, validation_idx = next(inner.split(development, groups=development.patient_id.to_numpy()))
        train = sorted(development.iloc[train_idx].patient_id.astype(str).tolist())
        validation = sorted(development.iloc[validation_idx].patient_id.astype(str).tolist())
        folds.append({"fold": fold, "train": train, "validation": validation, "test": test, "algorithm": "GroupKFold outer + GroupShuffleSplit inner; patient_id only"})
    return folds


def patient_masks(frame: pd.DataFrame, dataset: str, fold: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patients = frame.patient_id.astype(str)
    if dataset == "tusz":
        partition = frame.partition.astype(str)
        train = patients.isin(fold["train"]) & partition.eq("train")
        validation = patients.isin(fold["validation"]) & partition.eq("train")
        test = patients.isin(fold["test"]) & partition.eq("train")
    else:
        train = patients.isin(fold["train"])
        validation = patients.isin(fold["validation"])
        test = patients.isin(fold["test"])
    if (train & validation).any() or (train & test).any() or (validation & test).any():
        raise RuntimeError(f"patient overlap in fold {fold['fold']}")
    if not train.any() or not validation.any() or not test.any():
        raise RuntimeError(f"empty policy partition in fold {fold['fold']}")
    return train.to_numpy(), validation.to_numpy(), test.to_numpy()


def action_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in TARGET_COLUMNS + ["policy_action_id", "embedding_row_id"] if column not in frame.columns]
    if missing:
        raise RuntimeError(f"missing policy columns: {missing}")
    target = frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    if not np.allclose(target.sum(axis=1), 1.0, atol=2e-4):
        raise RuntimeError("policy target rows must sum to one")
    hard = frame.policy_action_id.to_numpy(dtype=np.int64)
    if np.any((hard < 0) | (hard >= len(ACTIONS))):
        raise RuntimeError("ineligible or invalid action appears in embedding index")
    return target, hard


def train_one_epoch(model: RouterHead, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for features, targets, _ in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = soft_cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(features.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_rows += count
    return total_loss / max(total_rows, 1)


@torch.inference_mode()
def evaluate(model: RouterHead, frame: pd.DataFrame, memmap: np.memmap, mask: np.ndarray, target: np.ndarray, hard: np.ndarray, device: torch.device, batch_size: int) -> tuple[dict[str, object], np.ndarray]:
    ids = frame.loc[mask, "embedding_row_id"].to_numpy(dtype=np.int64)
    dataset = EmbeddingDataset(memmap, ids, target[mask], hard[mask])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    prob_parts: list[np.ndarray] = []
    loss_total = 0.0
    row_total = 0
    for features, targets, _ in loader:
        logits = model(features.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        prob_parts.append(probs)
        loss = soft_cross_entropy(logits, targets.to(device, non_blocking=True))
        count = len(features)
        loss_total += float(loss.cpu()) * count
        row_total += count
    probabilities = np.concatenate(prob_parts, axis=0) if prob_parts else np.empty((0, len(ACTIONS)), dtype=np.float32)
    y_true = hard[mask]
    y_pred = probabilities.argmax(axis=1)
    metrics: dict[str, object] = {
        "rows": int(len(y_true)),
        "loss": float(loss_total / max(row_total, 1)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "target_soft_mass": {name: float(target[mask, index].mean()) for index, name in enumerate(ACTIONS)},
        "target_hard_distribution": {name: int((y_true == index).sum()) for index, name in enumerate(ACTIONS)},
        "predicted_distribution": {name: int((y_pred == index).sum()) for index, name in enumerate(ACTIONS)},
        "per_action": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIONS)))).tolist(),
    }
    per_action: dict[str, object] = {}
    for index, name in enumerate(ACTIONS):
        support = int((y_true == index).sum())
        predicted = int((y_pred == index).sum())
        true_positive = int(((y_true == index) & (y_pred == index)).sum())
        per_action[name] = {
            "support": support,
            "predicted": predicted,
            "recall": true_positive / support if support else 0.0,
            "precision": true_positive / predicted if predicted else 0.0,
            "f1": float(f1_score(y_true, y_pred, labels=[index], average="macro", zero_division=0)) if support else 0.0,
        }
    metrics["per_action"] = per_action
    return metrics, probabilities


def write_history(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def train_fold(dataset_name: str, frame: pd.DataFrame, memmap: np.memmap, fold: dict[str, object], args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    fold_id = int(fold["fold"])
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    train_mask, validation_mask, test_mask = patient_masks(frame, dataset_name, fold)
    target, hard = action_arrays(frame)
    train_hard = hard[train_mask]
    counts = np.bincount(train_hard, minlength=len(ACTIONS))
    if np.any(counts == 0):
        raise RuntimeError(f"fold {fold_id} training lacks action class: {counts.tolist()}")
    sample_weights = 1.0 / counts[train_hard].astype(np.float64)
    sample_count = int(args.steps_per_epoch * args.batch_size)
    train_ids = frame.loc[train_mask, "embedding_row_id"].to_numpy(dtype=np.int64)
    train_dataset = EmbeddingDataset(memmap, train_ids, target[train_mask], hard[train_mask])
    generator = torch.Generator().manual_seed(SEED + fold_id)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=sample_count, replacement=True, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterHead(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    history_path = fold_dir / "training_history.jsonl"
    best_metric = -float("inf")
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics, _ = evaluate(model, frame, memmap, validation_mask, target, hard, device, args.eval_batch_size)
        scheduler.step()
        metric = float(validation_metrics["macro_f1"])
        row = {
            "dataset": dataset_name,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_history(history_path, row)
        print(f"BIOT_POLICY_PROGRESS dataset={dataset_name} fold={fold_id} epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={float(validation_metrics['loss']):.6f} val_macro_f1={metric:.4f} val_bal_acc={float(validation_metrics['balanced_accuracy']):.4f} val_dist={validation_metrics['predicted_distribution']}", flush=True)
        improved = metric > best_metric + args.min_delta
        if improved:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "dataset": dataset_name, "fold": fold_id, "seed": SEED, "best_epoch": best_epoch, "best_validation_macro_f1": best_metric, "config": vars(args), "actions": list(ACTIONS)}, fold_dir / "best_policy_head.pt")
        elif epoch >= args.min_epochs:
            stale += 1
            if stale >= args.patience:
                print(f"BIOT_POLICY_EARLY_STOP dataset={dataset_name} fold={fold_id} epoch={epoch} best_epoch={best_epoch} best_val_macro_f1={best_metric:.4f}", flush=True)
                break

    checkpoint = torch.load(fold_dir / "best_policy_head.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    validation_metrics, validation_probabilities = evaluate(model, frame, memmap, validation_mask, target, hard, device, args.eval_batch_size)
    test_metrics, test_probabilities = evaluate(model, frame, memmap, test_mask, target, hard, device, args.eval_batch_size)
    for name, mask, probabilities in (("validation", validation_mask, validation_probabilities), ("test", test_mask, test_probabilities)):
        subset = frame.loc[mask, ["embedding_row_id", "patient_id", "recording", "policy_action_id"]].copy()
        subset["predicted_action_id"] = probabilities.argmax(axis=1).astype(np.int8)
        subset["predicted_action"] = subset.predicted_action_id.map(dict(enumerate(ACTIONS)))
        subset["target_action"] = subset.policy_action_id.map(dict(enumerate(ACTIONS)))
        subset["predicted_max_probability"] = probabilities.max(axis=1).astype(np.float32)
        subset.to_parquet(fold_dir / f"{name}_predictions.parquet", index=False)
    summary = {
        "dataset": dataset_name,
        "fold": fold_id,
        "seed": SEED,
        "fold_definition": fold,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_action_counts": {name: int(counts[index]) for index, name in enumerate(ACTIONS)},
        "balancing": "WeightedRandomSampler inverse frequency of hard policy_action_id; soft policy targets retained for loss",
        "early_stopping": {"monitor": "validation_macro_f1", "best_epoch": best_epoch, "best_value": best_metric, "patience": args.patience, "min_delta": args.min_delta, "max_epochs": args.epochs},
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str((fold_dir / "best_policy_head.pt").relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def aggregate_summaries(dataset: str, summaries: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    confusion = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
    total_rows = 0
    for summary in summaries:
        matrix = np.asarray(summary["test"]["confusion_matrix"], dtype=np.int64)
        confusion += matrix
        total_rows += int(summary["test"]["rows"])
    y_true = np.repeat(np.arange(len(ACTIONS)), confusion.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(len(ACTIONS)), confusion[i]) for i in range(len(ACTIONS))]) if total_rows else np.empty(0, dtype=int)
    # The concatenation above is row-major by true class; build the matching
    # predictions explicitly so aggregate metrics use the confusion counts.
    true_list: list[int] = []
    pred_list: list[int] = []
    for i in range(len(ACTIONS)):
        for j in range(len(ACTIONS)):
            true_list.extend([i] * int(confusion[i, j]))
            pred_list.extend([j] * int(confusion[i, j]))
    y_true = np.asarray(true_list, dtype=int)
    y_pred = np.asarray(pred_list, dtype=int)
    aggregate = {
        "dataset": dataset,
        "folds": len(summaries),
        "seed": SEED,
        "rows": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)) if len(y_true) else float("nan"),
        "per_action": {},
        "confusion_matrix": confusion.tolist(),
        "fold_summaries": [{"fold": s["fold"], "validation_macro_f1": s["validation"]["macro_f1"], "test_macro_f1": s["test"]["macro_f1"], "test_balanced_accuracy": s["test"]["balanced_accuracy"], "test_rows": s["test"]["rows"], "best_epoch": s["early_stopping"]["best_epoch"]} for s in summaries],
    }
    for index, name in enumerate(ACTIONS):
        support = int((y_true == index).sum())
        predicted = int((y_pred == index).sum())
        tp = int(((y_true == index) & (y_pred == index)).sum())
        aggregate["per_action"][name] = {"support": support, "predicted": predicted, "recall": tp / support if support else 0.0, "precision": tp / predicted if predicted else 0.0}
    (output_dir / "aggregate_test_metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return aggregate


@torch.inference_mode()
def score_tusz_dev(frame: pd.DataFrame, memmap: np.memmap, output_dir: Path, summaries: list[dict[str, object]], batch_size: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = frame.partition.eq("dev").to_numpy()
    target, hard = action_arrays(frame)
    ids = frame.loc[mask, "embedding_row_id"].to_numpy(dtype=np.int64)
    dataset = EmbeddingDataset(memmap, ids, target[mask], hard[mask])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_probabilities: list[np.ndarray] = []
    for summary in summaries:
        checkpoint = torch.load(ROOT / summary["checkpoint"], map_location=device, weights_only=False)
        model = RouterHead(hidden=int(checkpoint["config"].get("hidden", 64)), dropout=float(checkpoint["config"].get("dropout", 0.2))).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        parts: list[np.ndarray] = []
        for features, _, _ in loader:
            parts.append(torch.softmax(model(features.to(device, non_blocking=True)), dim=1).cpu().numpy())
        all_probabilities.append(np.concatenate(parts, axis=0))
        del model
    probabilities = np.mean(np.stack(all_probabilities, axis=0), axis=0)
    y_true = hard[mask]
    y_pred = probabilities.argmax(axis=1)
    result = {
        "dataset": "tusz",
        "partition": "official_dev",
        "rows": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(range(len(ACTIONS))), average="macro", zero_division=0)),
        "target_hard_distribution": {name: int((y_true == i).sum()) for i, name in enumerate(ACTIONS)},
        "predicted_distribution": {name: int((y_pred == i).sum()) for i, name in enumerate(ACTIONS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIONS)))).tolist(),
        "fold_ensemble": "uniform mean of five best-fold policy heads; Dev not used for training or early stopping",
    }
    (output_dir / "official_dev_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mit", "tusz"), required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("the requested BIOT policy training requires CUDA")
    frame, memmap = load_embedding_bundle(args.dataset)
    target, hard = action_arrays(frame)
    if args.dataset == "mit":
        folds = _patient_folds_mit()
    else:
        folds = _patient_folds_tusz(frame)
    output_dir = OUTPUT_ROOT / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": "h6-causal-policy-router-v3-four-action-biot-kfold",
        "dataset": args.dataset,
        "status": "training_in_progress",
        "seed": SEED,
        "model": "frozen BIOT/EEG-PREST encoder + original 256->64->4 RouterHead",
        "actions": list(ACTIONS),
        "risk_head": False,
        "risk_label": False,
        "rows": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "folds": folds,
        "training_config": vars(args),
        "balancing": "inverse hard policy_action_id frequency WeightedRandomSampler; target remains soft",
        "early_stopping": {"monitor": "validation_macro_f1", "patience": args.patience, "min_delta": args.min_delta, "minimum_epochs": args.min_epochs},
        "embedding_manifest_sha256": sha256(EMBED_ROOT / args.dataset / "manifest.json"),
        "created_utc": time.time(),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summaries = []
    for fold in folds:
        summary = train_fold(args.dataset, frame, memmap, fold, args, output_dir)
        summaries.append(summary)
    aggregate = aggregate_summaries(args.dataset, summaries, output_dir)
    external = score_tusz_dev(frame, memmap, output_dir, summaries, args.eval_batch_size) if args.dataset == "tusz" and frame.partition.eq("dev").any() else None
    manifest.update({"status": "PASS", "completed_at_unix": time.time(), "aggregate_test": aggregate, "official_dev": external})
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": args.dataset, "aggregate_test": aggregate, "official_dev": external}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
