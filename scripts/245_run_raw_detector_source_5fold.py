"""Strict patient-grouped raw-EEG detector-source classification.

This is a new, self-contained H3 extension.  It predicts which frozen detector
generated a retained false-positive event from raw EEG only.  The source event
table is fixed before any model fitting; no detector probabilities, duration or
model-ID features enter the network.  CHB-MIT uses OOF outer-test FP rows from
the frozen v3 release (seed 3407); TUSZ uses the frozen external eval rows.

The classifier uses the same event-sliding 16-channel 128-Hz representation as
the earlier exploratory raw experiment: one 10-s midpoint window for short
events and all 10-s windows at 2-s stride for longer events, followed by mean
event pooling.  Each fold has an independent patient-level validation split;
the outer test is evaluated exactly once after early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/root/b_false_alarm_atlas")
CACHE_ROOT = Path("/mnt/d/EEGData/bfa_h3_raw128_cnn_v1")
CHB_EVENTS = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/h2h3_events.parquet"
TUSZ_EVENTS = ROOT / "outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL = {m: i for i, m in enumerate(MODELS)}
FS = 128.0
WINDOW_S = 10.0
STRIDE_S = 2.0
WINDOW_N = int(FS * WINDOW_S)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_cache_path(dataset: str, relative_path: str) -> Path:
    p = Path(relative_path)
    if p.suffix.lower() != ".npy":
        p = p.with_suffix(".npy")
    return CACHE_ROOT / dataset / p


def load_events(dataset: str) -> pd.DataFrame:
    if dataset == "chbmit":
        if not CHB_EVENTS.exists():
            raise FileNotFoundError(CHB_EVENTS)
        x = pd.read_parquet(CHB_EVENTS)
        # OOF rows only: each CHB patient belongs to one frozen outer test
        # fold.  A single fixed expert seed prevents triplicating the same
        # event and makes the source labels independent of seed selection.
        x = x[(x["partition"] == "test") & (x["seed"].astype(int) == 3407) & (x["event_type"] == "fp")].copy()
        x = x.rename(columns={"patient": "patient_id"})
        x["relative_path"] = x["relative_path"].astype(str)
        x["recording"] = x["relative_path"]
        keep = ["patient_id", "recording", "relative_path", "model", "start_s", "end_s", "peak_probability", "fold", "seed"]
    elif dataset == "tusz":
        if not TUSZ_EVENTS.exists():
            raise FileNotFoundError(TUSZ_EVENTS)
        x = pd.read_parquet(TUSZ_EVENTS)
        # Eval rows are a single external cohort scored by frozen experts.
        # No labels are used to fit the three experts here.
        x = x[(x["partition"] == "eval") & (x["is_fp"].astype(int) == 1)].copy()
        x["relative_path"] = x["recording"].astype(str)
        x["fold"] = -1
        x["seed"] = 17
        keep = ["patient_id", "recording", "relative_path", "model", "start_s", "end_s", "fold", "seed"]
    else:
        raise ValueError(dataset)
    x = x[x["model"].isin(MODELS)].copy()
    x["model"] = x["model"].astype(str)
    x["label"] = x["model"].map(LABEL).astype(int)
    x["event_id"] = [
        hashlib.sha1(f"{dataset}|{p}|{r}|{m}|{a:.3f}|{b:.3f}".encode()).hexdigest()[:20]
        for p, r, m, a, b in zip(x.patient_id, x.relative_path, x.model, x.start_s, x.end_s, strict=True)
    ]
    # Repeated rows can occur when the same event appears in a materialized
    # table more than once.  Keep one raw event per detector and recording.
    x = x.drop_duplicates("event_id", keep="first").reset_index(drop=True)
    x["cache_path"] = [str(canonical_cache_path(dataset, p)) for p in x.relative_path]
    missing = [p for p in x.cache_path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset}: {len(missing)} raw cache files missing; first={missing[0]}")
    return x[keep + ["label", "event_id", "cache_path"]]


def window_starts(start_s: float, end_s: float, n_samples: int) -> list[int]:
    total_s = n_samples / FS
    start_s = max(0.0, float(start_s))
    end_s = min(total_s, max(start_s, float(end_s)))
    duration = end_s - start_s
    if duration <= WINDOW_S:
        mid = (start_s + end_s) / 2.0
        a = min(max(0.0, mid - WINDOW_S / 2.0), max(0.0, total_s - WINDOW_S))
        return [int(round(a * FS))]
    last = max(0.0, end_s - WINDOW_S)
    vals = list(np.arange(start_s, last + 1e-6, STRIDE_S, dtype=float))
    if not vals or abs(vals[-1] - last) > 1e-4:
        vals.append(last)
    return sorted({int(round(min(max(0.0, a), total_s - WINDOW_S) * FS)) for a in vals})


@dataclass
class EventItem:
    event_id: str
    patient_id: str
    recording: str
    cache_path: str
    start_s: float
    end_s: float
    label: int


class EventDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, rows: pd.DataFrame, max_windows: int | None = None):
        self.items = [EventItem(str(r.event_id), str(r.patient_id), str(r.recording), str(r.cache_path), float(r.start_s), float(r.end_s), int(r.label)) for r in rows.itertuples()]
        self.max_windows = max_windows
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _array(self, path: str) -> np.ndarray:
        arr = self._arrays.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[0] != 16:
                raise ValueError(f"unexpected raw shape {path}: {arr.shape}")
            self._arrays[path] = arr
            # Keep the worker-local cache bounded when an EDF-heavy cohort is
            # used.  Numpy memmaps do not copy the entire file into RAM.
            if len(self._arrays) > 64:
                self._arrays.pop(next(iter(self._arrays)))
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        arr = self._array(item.cache_path)
        starts = window_starts(item.start_s, item.end_s, int(arr.shape[1]))
        if self.max_windows is not None and len(starts) > self.max_windows:
            # Deterministic evenly spaced subsampling prevents very long
            # alarms from dominating an event-level minibatch.
            sel = np.linspace(0, len(starts) - 1, self.max_windows).round().astype(int)
            starts = [starts[int(i)] for i in sel]
        windows = []
        for a in starts:
            z = np.asarray(arr[:, a : a + WINDOW_N], dtype=np.float32)
            if z.shape[-1] != WINDOW_N:
                pad = np.zeros((16, WINDOW_N), dtype=np.float32)
                pad[:, : z.shape[-1]] = z
                z = pad
            z = z - z.mean(axis=1, keepdims=True)
            z = z / 100.0
            windows.append(z)
        return torch.from_numpy(np.stack(windows, axis=0)), item.label, item.event_id


def collate(batch: list[tuple[torch.Tensor, int, str]]):
    xs, ys, ids = zip(*batch, strict=True)
    return list(xs), torch.tensor(ys, dtype=torch.long), list(ids)


class RawSourceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 9, padding=4), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 3)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

    def forward(self, x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        z = self.embed(x)
        chunks = torch.split(z, sizes, dim=0)
        pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        return self.head(pooled)


def event_forward(model: RawSourceNet, xs: list[torch.Tensor], device: torch.device, window_batch: int) -> torch.Tensor:
    sizes = [int(x.shape[0]) for x in xs]
    flat = torch.cat(xs, dim=0).to(device, non_blocking=True)
    chunks = []
    for a in range(0, flat.shape[0], window_batch):
        chunks.append(model.embed(flat[a : a + window_batch]))
    z = torch.cat(chunks, dim=0)
    # Events have different numbers of sliding windows; pool each event before
    # stacking so long events cannot trigger a ragged-tensor failure.
    pooled = torch.stack([chunk.mean(dim=0) for chunk in torch.split(z, sizes, dim=0)], dim=0)
    return model.head(pooled)


def class_weights(labels: Iterable[int]) -> torch.Tensor:
    c = np.bincount(np.asarray(list(labels), dtype=int), minlength=3).astype(float)
    c[c == 0] = 1.0
    w = c.sum() / (3.0 * c)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: RawSourceNet, loader: DataLoader, device: torch.device, window_batch: int) -> tuple[dict[str, float | None], pd.DataFrame]:
    model.eval(); ys: list[int] = []; probs: list[np.ndarray] = []; ids: list[str] = []
    for xs, y, event_ids in loader:
        logits = event_forward(model, xs, device, window_batch)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy()); ys.extend(y.numpy().tolist()); ids.extend(event_ids)
    p = np.concatenate(probs, axis=0) if probs else np.zeros((0, 3)); y = np.asarray(ys, dtype=int)
    pred = p.argmax(1) if len(p) else np.zeros(0, dtype=int)
    out: dict[str, float | None] = {
        "events": float(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) and len(np.unique(y)) > 1 else None,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else None,
        "macro_auroc": None,
    }
    try:
        out["macro_auroc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        pass
    pred_df = pd.DataFrame({"event_id": ids, "true_label": y, "predicted_label": pred, "prob_singlem": p[:, 0] if len(p) else [], "prob_cbramod": p[:, 1] if len(p) else [], "prob_tcn_gat": p[:, 2] if len(p) else []})
    return out, pred_df


def split_train_validation(rows: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = rows.patient_id.to_numpy()
    for attempt in range(50):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        tr, va = next(gss.split(rows, groups=groups))
        if len(np.unique(rows.iloc[tr].label)) >= 2 and len(np.unique(rows.iloc[va].label)) >= 2:
            return tr, va
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(gss.split(rows, groups=groups))


def train_fold(rows: pd.DataFrame, outer_fold: int, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace, out: Path, device: torch.device) -> dict[str, object]:
    train_rows = rows.iloc[train_idx].reset_index(drop=True)
    test_rows = rows.iloc[test_idx].reset_index(drop=True)
    fit_idx, val_idx = split_train_validation(train_rows, args.seed + outer_fold * 101)
    fit_rows = train_rows.iloc[fit_idx].reset_index(drop=True)
    val_rows = train_rows.iloc[val_idx].reset_index(drop=True)
    if args.max_train_events:
        fit_rows = fit_rows.head(args.max_train_events).copy()
    loader_args = dict(batch_size=args.event_batch, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0, collate_fn=collate)
    fit_loader = DataLoader(EventDataset(fit_rows, args.max_windows_per_event), shuffle=True, generator=torch.Generator().manual_seed(args.seed + outer_fold), **loader_args)
    val_loader = DataLoader(EventDataset(val_rows, args.max_windows_per_event), shuffle=False, **loader_args)
    test_loader = DataLoader(EventDataset(test_rows, args.max_windows_per_event), shuffle=False, **loader_args)
    model = RawSourceNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(weight=class_weights(fit_rows.label.tolist()).to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_metric = -math.inf; best_state = None; best_epoch = 0; patience_left = args.patience; history = []
    for epoch in range(args.epochs):
        model.train(); running = 0.0; n = 0; t0 = time.time()
        for xs, y, _ in fit_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                logits = event_forward(model, xs, device, args.window_batch)
                loss = criterion(logits, y.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            running += float(loss.detach().cpu()) * len(y); n += len(y)
        sched.step()
        val_metrics, _ = evaluate(model, val_loader, device, args.window_batch)
        metric = val_metrics.get("macro_auroc")
        metric_val = float(metric) if metric is not None else float(val_metrics.get("balanced_accuracy") or -math.inf)
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, n), "validation": val_metrics, "seconds": time.time() - t0})
        if metric_val > best_metric + args.min_delta:
            best_metric = metric_val; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0 and epoch + 1 >= args.min_epochs:
                break
    if best_state is not None: model.load_state_dict(best_state)
    val_metrics, val_pred = evaluate(model, val_loader, device, args.window_batch)
    test_metrics, test_pred = evaluate(model, test_loader, device, args.window_batch)
    ckpt = out / f"fold{outer_fold}_best.pt"; torch.save({"state_dict": model.state_dict(), "outer_fold": outer_fold, "seed": args.seed, "best_epoch": best_epoch, "dataset": args.dataset}, ckpt)
    val_pred.to_parquet(out / f"fold{outer_fold}_validation_predictions.parquet", index=False); test_pred.to_parquet(out / f"fold{outer_fold}_test_predictions.parquet", index=False)
    pd.DataFrame(history).to_json(out / f"fold{outer_fold}_history.json", orient="records", indent=2)
    return {"outer_fold": outer_fold, "train_events": len(fit_rows), "validation_events": len(val_rows), "test_events": len(test_rows), "train_patients": int(train_rows.patient_id.nunique()), "validation_patients": int(val_rows.patient_id.nunique()), "test_patients": int(test_rows.patient_id.nunique()), "best_epoch": best_epoch, "epochs_run": len(history), "validation": val_metrics, "test": test_metrics, "checkpoint": ckpt.name, "checkpoint_sha256": sha256(ckpt), "status": "complete"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["chbmit", "tusz"], required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--event-batch", type=int, default=16)
    ap.add_argument("--window-batch", type=int, default=1024)
    ap.add_argument("--max-windows-per-event", type=int, default=32)
    ap.add_argument("--max-train-events", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = ROOT / "outputs/reports" / args.namespace
    if (out / "manifest.json").exists():
        raise RuntimeError(f"refusing overwrite: {out}")
    # The mandatory duplicate preflight writes its report into the target
    # namespace before a run starts.  Reuse that directory only when it
    # contains that audit file and no experiment output; never overwrite a
    # prior run or an unrelated artifact.
    if out.exists():
        allowed = {"no_duplicate_overlap_preflight.json"}
        unexpected = {p.name for p in out.iterdir()} - allowed
        if unexpected:
            raise RuntimeError(f"refusing overwrite; existing files in {out}: {sorted(unexpected)}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    rows = load_events(args.dataset)
    if args.smoke:
        args.epochs = min(args.epochs, 1); args.min_epochs = 1; args.patience = 1; args.max_train_events = args.max_train_events or 800; args.event_batch = min(args.event_batch, 8); args.workers = min(args.workers, 2)
    patients = rows.patient_id.to_numpy(); gkf = GroupKFold(n_splits=5); fold_records = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, (tr, te) in enumerate(gkf.split(rows, groups=patients)):
        print(json.dumps({"dataset": args.dataset, "fold": fold, "train_events": len(tr), "test_events": len(te), "device": str(device)}), flush=True)
        fold_records.append(train_fold(rows, fold, tr, te, args, out, device))
        if args.smoke and fold >= 0:
            break
    test_rows = [r["test"] for r in fold_records]
    manifest = {
        "release_id": args.namespace, "status": "smoke_complete" if args.smoke else "complete", "dataset": args.dataset,
        "task": "raw EEG detector-source classification", "target": "one frozen detector source per retained FP event",
        "features": {"raw_eeg_only": True, "channels": 16, "sampling_hz": 128.0, "window_seconds": 10.0, "window_stride_seconds": 2.0, "shape": [16, 1280], "normalization": "per-channel window de-mean then divide by 100", "detector_probability_feature": False, "duration_feature": False},
        "source_protocol": "CHB: frozen v3 outer-test rows, seed 3407; TUSZ: frozen external eval rows; FP-only",
        "split": "ordinary GroupKFold(n_splits=5) by canonical patient; GroupShuffleSplit validation inside outer train",
        "models": MODELS, "seed": args.seed, "outer_folds_completed": len(fold_records), "outer_folds_total": 5,
        "folds": fold_records, "test_evaluation_count": 1, "test_used_for_selection": False, "threshold_source": "none; softmax argmax",
        "source_event_table_sha256": sha256(CHB_EVENTS if args.dataset == "chbmit" else TUSZ_EVENTS), "events": int(len(rows)), "patients": int(rows.patient_id.nunique()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "events_used.parquet").write_parquet(rows, index=False) if False else rows.to_parquet(out / "events_used.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "manifest.sha256").write_text(sha256(out / "manifest.json") + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
