"""H7 patient-local fine-tuning comparison.

The script implements the frozen H7 addendum without touching the earlier
template-suppressor outputs.  It is intentionally resumable and immutable at
the run level.  Training labels come only from the frozen window-label
manifest; the raw EEG is reconstructed from the official TUSZ EDFs.

Usage examples (inside the project WSL environment)::

    python scripts/199_h7_patient_finetune_comparison.py smoke
    python scripts/199_h7_patient_finetune_comparison.py train --seed 17
    python scripts/199_h7_patient_finetune_comparison.py score-dev --seed 17
    python scripts/199_h7_patient_finetune_comparison.py gate --seed 17
    python scripts/199_h7_patient_finetune_comparison.py score-holdout --seed 17

``smoke`` never reads holdout rows.  ``train`` only reads fit rows.  Dev and
holdout are separate commands so a checkpoint cannot silently inspect a test
row during optimization.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.lora import apply_lora, trainable_parameter_report
from bfa.models.shared_head import SharedContextHead
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.preprocessing.tuh import (
    MODEL_VIEW_CONFIG,
    load_tuh_canonical_raw_uv,
    tuh_model_view,
    tuh_quality_features,
)


ROOT = Path("/root/b_false_alarm_atlas")
LABEL_DIR = ROOT / "outputs/reports/h7-patient-finetune-comparison-v1/labels"
OUT = ROOT / "outputs/reports" / os.environ.get("H7_FT_NAMESPACE", "h7-patient-finetune-comparison-v1")
SOURCE = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
CHECKPOINT_ROOT = ROOT / "runs/v3-groupkfold-confirmatory-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
METHODS = ("head_only", "lora", "full")
WINDOWS_PER_CONTEXT = 31
RATE = {"singlem": 128, "cbramod": 200, "tcn_gat": 256}
SEED_VALUES = (17, 42, 3407)
CHECKPOINT_STEPS = (100, 250, 500, 750, 1000, 1500, 2000)
MICROBATCH_CONTEXTS = max(1, int(os.environ.get("H7_MICROBATCH_CONTEXTS", "4")))
MAX_RECORDINGS = max(128, int(os.environ.get("H7_MAX_RECORDINGS", "2048")))
EARLY_STOP_ENABLED = os.environ.get("H7_EARLY_STOP", "1").strip().lower() not in {"0", "false", "no"}
EARLY_STOP_EVERY = max(1, int(os.environ.get("H7_EARLY_STOP_EVERY", "100")))
EARLY_STOP_PATIENCE = max(1, int(os.environ.get("H7_EARLY_STOP_PATIENCE", "3")))
EARLY_STOP_MIN_DELTA = max(0.0, float(os.environ.get("H7_EARLY_STOP_MIN_DELTA", "1e-4")))
VALIDATION_MAX_PER_CLASS = max(64, int(os.environ.get("H7_VALIDATION_MAX_PER_CLASS", "256")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def device_from_arg(value: str | None) -> torch.device:
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_label_table() -> pd.DataFrame:
    path = LABEL_DIR / "patient_detector_window_labels.parquet"
    table = pd.read_parquet(path)
    table = table[table.model.isin(MODELS)].copy()
    table["label"] = table["label"].astype("Int8")
    return table


def load_source_records() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for partition in ("train", "dev", "eval"):
        manifest = json.loads((SOURCE / "manifests" / f"{partition}_manifest.json").read_text())
        for record in manifest["records"]:
            output[str(record["relative_edf"])] = record
    return output


def load_unit_table(table: pd.DataFrame, model: str, patient: str, partition: str) -> pd.DataFrame:
    unit = table[
        (table.model == model)
        & (table.patient_id == patient)
        & (table.partition == partition)
    ].copy()
    unit = unit.sort_values(["recording", "context_end_s"], kind="stable").reset_index(drop=True)
    return unit


class RecordingViewCache:
    """LRU cache for one detector's filtered/resampled EDF views."""

    def __init__(self, model: str, max_recordings: int = 128) -> None:
        self.model = model
        self.max_recordings = max_recordings
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, float]] = OrderedDict()
        self.source_records = load_source_records()

    def get(self, recording: str) -> tuple[np.ndarray, np.ndarray, float]:
        if recording in self._cache:
            value = self._cache.pop(recording)
            self._cache[recording] = value
            return value
        path = DATA_ROOT / recording
        raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(path)
        view = tuh_model_view(raw_uv, source_hz, self.model)
        quality = tuh_quality_features(raw_uv, source_hz)
        value = (view, quality, float(source_hz))
        self._cache[recording] = value
        while len(self._cache) > self.max_recordings:
            self._cache.popitem(last=False)
        return value

    def context(self, recording: str, context_end_s: float) -> tuple[np.ndarray, np.ndarray]:
        view, quality, _ = self.get(recording)
        rate = RATE[self.model]
        # The detector head consumes 31 ten-second windows at a two-second
        # stride.  The first and last windows therefore span 70 seconds
        # (60 seconds of prior context plus the current 10-second decision
        # window), matching the original H7 detector contract.
        start_s = float(context_end_s) - 70.0
        start = int(round(start_s * rate))
        context_n = int(70 * rate)
        if start < 0 or start + context_n > view.shape[-1]:
            raise ValueError(f"context outside recording: {recording} end={context_end_s}")
        context = view[:, start : start + context_n]
        windows = np.stack(
            [context[:, i * 2 * rate : i * 2 * rate + 10 * rate] for i in range(WINDOWS_PER_CONTEXT)],
            axis=0,
        )
        if windows.shape != (WINDOWS_PER_CONTEXT, 16, 10 * rate):
            raise ValueError(f"bad context windows {windows.shape} for {recording}")
        source_hz = self.get(recording)[2]
        q_window = int(round(2.0 * source_hz))
        quality_start = int(round(start_s * source_hz / q_window))
        # tuh_quality_features starts at zero and advances by 2 s.
        quality_start = int(round(start_s / 2.0))
        if quality_start < 0 or quality_start + WINDOWS_PER_CONTEXT > len(quality):
            # Quality is only used by TCN-GAT.  Pad at the boundary rather than
            # changing the raw EEG contract; all official H7 rows are valid.
            q = np.zeros((WINDOWS_PER_CONTEXT, 16, 3), dtype=np.float32)
            stop = min(len(quality), quality_start + WINDOWS_PER_CONTEXT)
            if stop > quality_start:
                q[: stop - quality_start] = quality[quality_start:stop]
        else:
            q = quality[quality_start : quality_start + WINDOWS_PER_CONTEXT]
        return windows.astype(np.float32, copy=False), q.astype(np.float32, copy=False)


def batch_contexts(rows: pd.DataFrame, cache: RecordingViewCache) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        w, q = cache.context(str(row.recording), float(row.context_end_s))
        windows.append(w)
        qualities.append(q)
    return np.stack(windows), np.stack(qualities)


def build_validation_sample(unit_all: pd.DataFrame, seed: int, max_per_class: int = VALIDATION_MAX_PER_CLASS) -> pd.DataFrame:
    """Build a deterministic, class-balanced validation sample for early stopping.

    The sample is drawn only from the phase labelled ``dev`` in the frozen
    manifest (the user-facing validation set).  It never reads test rows and
    is kept fixed for the whole patient-local run so validation loss is
    comparable across checkpoints.
    """
    validation = unit_all[
        (unit_all.phase == "dev") & unit_all.label_status.isin(["positive", "negative"])
    ].copy()
    if validation.empty:
        raise RuntimeError("validation set is empty")
    rng = np.random.default_rng(seed + 310_000)
    selected: list[pd.DataFrame] = []
    for label_status in ("positive", "negative"):
        group = validation[validation.label_status == label_status]
        if group.empty:
            continue
        n = min(len(group), max_per_class)
        indices = rng.choice(len(group), size=n, replace=False)
        selected.append(group.iloc[np.sort(indices)])
    if not selected or len(selected) < 2:
        raise RuntimeError("validation set lacks both positive and negative labels")
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(["recording", "context_end_s"], kind="stable").reset_index(drop=True)


def evaluate_validation_loss(
    model: str,
    encoder: nn.Module,
    head: nn.Module,
    rows: pd.DataFrame,
    cache: RecordingViewCache,
    device: torch.device,
    batch_size: int = 4,
) -> float:
    """Evaluate balanced validation BCE without touching the test set."""
    if rows.empty:
        return float("nan")
    encoder_training = encoder.training
    head_training = head.training
    encoder.eval(); head.eval()
    losses: list[tuple[float, int]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows.iloc[start : start + batch_size]
            windows, quality = batch_contexts(batch_rows, cache)
            logits = forward_context(model, encoder, head, windows, quality, device)
            labels = torch.tensor(batch_rows.label.to_numpy(dtype=np.float32), device=device)
            value = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(value):
                raise RuntimeError("non-finite validation loss")
            losses.append((float(value.detach().cpu()), len(batch_rows)))
    if encoder_training:
        encoder.train()
    else:
        encoder.eval()
    if head_training:
        head.train()
    else:
        head.eval()
    return float(sum(value * count for value, count in losses) / max(1, sum(count for _, count in losses)))


def build_schedule(unit: pd.DataFrame, seed: int, updates: int, batch_size: int = 64) -> np.ndarray:
    fit = unit[(unit.phase == "fit") & unit.label_status.isin(["positive", "negative"])].copy()
    positive = fit[fit.label_status == "positive"]
    negative = fit[fit.label_status == "negative"]
    if positive.empty or negative.empty:
        raise RuntimeError("fit unit lacks both positive and negative labels")
    events = sorted({event for value in positive.seizure_event_ids for event in str(value).split("|") if event})
    event_pools = {event: positive[positive.seizure_event_ids.map(lambda x: event in str(x).split("|"))].index.to_numpy() for event in events}
    if not event_pools:
        raise RuntimeError("positive fit rows have no seizure event IDs")
    # Sample a recording uniformly, then a 30-s background block uniformly,
    # then one eligible window within that block.  Sampling rows directly
    # would overweight long uninterrupted backgrounds.
    negative_blocks: dict[str, list[np.ndarray]] = {}
    for rec, group in negative.groupby("recording", sort=True):
        block_ids = np.floor(group.context_end_s.to_numpy(float) / 30.0).astype(int)
        blocks = [sub.index.to_numpy() for _, sub in group.assign(_block=block_ids).groupby("_block", sort=True)]
        if blocks:
            negative_blocks[str(rec)] = blocks
    if not negative_blocks:
        raise RuntimeError("negative fit rows have no recording IDs")
    rng = np.random.default_rng(seed)
    positions = unit.index.to_numpy()
    result = np.empty((updates, batch_size), dtype=np.int64)
    event_names = tuple(sorted(event_pools))
    record_names = tuple(sorted(negative_blocks))
    for update in range(updates):
        pos: list[int] = []
        neg: list[int] = []
        for _ in range(batch_size // 2):
            event = event_names[int(rng.integers(len(event_names)))]
            candidates = event_pools[event]
            pos.append(int(candidates[int(rng.integers(len(candidates)))]))
            recording = record_names[int(rng.integers(len(record_names)))]
            blocks = negative_blocks[recording]
            candidates = blocks[int(rng.integers(len(blocks)))]
            neg.append(int(candidates[int(rng.integers(len(candidates)))]))
        batch = np.asarray(pos + neg, dtype=np.int64)
        result[update] = batch[rng.permutation(batch_size)]
    # Index positions are local to unit after reset_index; convert the source
    # index labels selected above into local row positions.
    lookup = {int(value): i for i, value in enumerate(unit.index.to_numpy())}
    return np.vectorize(lookup.__getitem__, otypes=[np.int64])(result)


def model_and_head(model: str, method: str, checkpoint: Path, device: torch.device) -> tuple[nn.Module, SharedContextHead, dict[str, Any]]:
    """Build a detector, load the frozen CHB checkpoint and configure adaptation."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("update", -1)) != 5000:
        raise ValueError(f"checkpoint is not step 5000: {checkpoint}")
    # LoRA needs gradients through the frozen backbone computation so the
    # low-rank parametrizations receive updates; the underlying pretrained
    # tensors remain requires_grad=False.  Full tuning also enables gradients.
    train_backbone = method in {"full", "lora"}
    if model == "singlem":
        encoder = SingLEMAdapter(ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt", train_backbone=train_backbone)
        encoder.projection.load_state_dict({"weight": state["encoder"]["projection.weight"], "bias": state["encoder"]["projection.bias"]})
        if method == "lora":
            for parameter in encoder.encoder.parameters():
                parameter.requires_grad_(False)
            lora_targets = apply_lora(encoder.encoder, model)
        else:
            lora_targets = []
    elif model == "cbramod":
        encoder = CBraModAdapter(ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth", train_backbone=train_backbone)
        encoder.projection.load_state_dict({"weight": state["encoder"]["projection.weight"], "bias": state["encoder"]["projection.bias"]})
        if method == "lora":
            for parameter in encoder.backbone.parameters():
                parameter.requires_grad_(False)
            lora_targets = apply_lora(encoder.backbone, model)
        else:
            lora_targets = []
    elif model == "tcn_gat":
        encoder = MultiScaleTCNGAT()
        encoder.load_state_dict(state["encoder"], strict=True)
        # The native detector head is not part of the H7 context contract; it
        # is kept frozen for every adaptation method so the parameter report
        # reflects parameters that can actually affect the H7 output.
        encoder.native_head.requires_grad_(False)
        lora_targets: list[str] = []
        if method == "head_only":
            encoder.requires_grad_(False)
        elif method == "lora":
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            lora_targets = apply_lora(encoder, model)
        else:
            lora_targets = []
    else:
        raise ValueError(model)
    head = SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    if method == "head_only":
        # Foundation adapters expose a trainable projection; TCN's output
        # projection is the analogous trainable representation layer.
        if model == "tcn_gat":
            encoder.output_projection.requires_grad_(True)
        else:
            encoder.projection.requires_grad_(True)
    elif method == "lora":
        if model == "tcn_gat":
            encoder.output_projection.requires_grad_(True)
        else:
            encoder.projection.requires_grad_(True)
    # SharedContextHead is always the trainable context head.
    encoder = encoder.to(device)
    head = head.to(device)
    return encoder, head, {"checkpoint_update": int(state["update"]), "lora_targets": lora_targets}


def forward_context(model: str, encoder: nn.Module, head: nn.Module, windows: np.ndarray, quality: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(windows)).to(device=device, dtype=torch.float32)
    q = torch.from_numpy(np.ascontiguousarray(quality)).to(device=device, dtype=torch.float32)
    batch = x.shape[0]
    # Raw detector views are kept at each model's native sampling rate.  The
    # 31 windows are the original causal context used by H7.
    if model in {"singlem", "cbramod"}:
        x = x.reshape(batch * WINDOWS_PER_CONTEXT, 16, 10, RATE[model])
        embeddings = encoder.forward_window(x)
    else:
        x = x.reshape(batch * WINDOWS_PER_CONTEXT, 16, 10 * RATE[model])
        q = q.reshape(batch * WINDOWS_PER_CONTEXT, 16, 3)
        embeddings = encoder.forward_window(x, q)
    embeddings = embeddings.reshape(batch, WINDOWS_PER_CONTEXT, 16, 128)
    return head(embeddings)


def parameter_groups(model: str, encoder: nn.Module, head: nn.Module, method: str) -> list[dict[str, Any]]:
    if model in {"singlem", "cbramod"}:
        backbone = encoder.encoder if model == "singlem" else encoder.backbone
        if method == "full":
            return [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 3e-5}, {"params": list(encoder.projection.parameters()) + list(head.parameters()), "lr": 3e-4}]
        return [{"params": [p for p in encoder.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad], "lr": 3e-4}]
    if method == "full":
        return [{"params": [p for n, p in encoder.named_parameters() if p.requires_grad and not n.startswith("output_projection")], "lr": 1e-4}, {"params": list(encoder.output_projection.parameters()) + list(head.parameters()), "lr": 3e-4}]
    return [{"params": [p for p in encoder.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad], "lr": 3e-4}]


def run_one_train(model: str, method: str, partition: str, patient: str, seed: int, updates: int, device: torch.device, smoke: bool = False) -> dict[str, Any]:
    # Make LoRA initialization, dropout and any backend RNGs reproducible for
    # every patient-detector-method unit.  The sampling schedule has its own
    # derived seed below, so changing the schedule cannot silently alter
    # model initialization.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    unit_all = load_unit_table(load_label_table(), model, patient, partition)
    unit = unit_all[unit_all.phase == "fit"].reset_index(drop=True)
    if len(unit) < 2 or not {"positive", "negative"}.issubset(set(unit.label_status)):
        raise RuntimeError(f"invalid fit labels for {model}/{partition}/{patient}")
    run_dir = OUT / "runs" / f"{model}__{method}__{partition}__{patient}__seed{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    schedule = build_schedule(unit, seed=seed + 100_000, updates=updates)
    np.save(run_dir / "sampling_schedule.npy", schedule)
    checkpoint_source = CHECKPOINT_ROOT / model / "split0_seed17_main/checkpoints/step_05000.pt"
    encoder, head, model_meta = model_and_head(model, method, checkpoint_source, device)
    groups = parameter_groups(model, encoder, head, method)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    # Keep the full patient recording set resident for the duration of a
    # patient-local run.  The previous 128-entry LRU repeatedly re-read EDFs
    # for long patients (hundreds of recordings), turning training into an
    # I/O-bound loop without changing any scientific inputs.
    cache = RecordingViewCache(model, max_recordings=MAX_RECORDINGS)
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 200_000)
    validation_sample: pd.DataFrame | None = None
    if EARLY_STOP_ENABLED and not smoke:
        validation_sample = build_validation_sample(unit_all, seed)
        validation_sample.to_csv(run_dir / "validation_sample.csv", index=False)
    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    best_update = 0
    bad_validation_checks = 0
    actual_updates = 0
    early_stopped = False
    if method == "head_only":
        # A frozen backbone must also be in evaluation mode; otherwise the
        # TCN-GAT attention dropout (and any foundation dropout) would make
        # the supposedly fixed representation stochastic during head tuning.
        encoder.eval()
        head.train()
    else:
        encoder.train()
        head.train()
    started = time.monotonic()
    for update in range(updates):
        rows = unit.iloc[schedule[update]].copy()
        order = rng.permutation(len(rows))
        rows = rows.iloc[order].reset_index(drop=True)
        optimizer.zero_grad(set_to_none=True)
        # The effective batch remains 64, but the expensive foundation
        # backbones are evaluated in small context micro-batches.  Summing
        # the normalized losses gives the same gradient as one 64-context
        # batch while avoiding a multi-gigabyte autograd graph for 31 windows
        # per context.
        labels = torch.tensor(rows.label.to_numpy(dtype=np.float32), device=device)
        micro_losses: list[tuple[float, int]] = []
        for micro_start in range(0, len(rows), MICROBATCH_CONTEXTS):
            micro_rows = rows.iloc[micro_start : micro_start + MICROBATCH_CONTEXTS]
            windows, quality = batch_contexts(micro_rows, cache)
            logits = forward_context(model, encoder, head, windows, quality, device)
            micro_labels = labels[micro_start : micro_start + len(micro_rows)]
            raw_loss = F.binary_cross_entropy_with_logits(logits, micro_labels, pos_weight=torch.tensor(1.0, device=device))
            if not torch.isfinite(raw_loss):
                raise RuntimeError(f"non-finite loss at update {update + 1}")
            sample_fraction = len(micro_rows) / len(rows)
            (raw_loss * sample_fraction).backward()
            micro_losses.append((float(raw_loss.detach().cpu()), len(micro_rows)))
        loss_value = float(sum(value * count for value, count in micro_losses) / len(rows)) if micro_losses else float("nan")
        trainable = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at update {update + 1}")
        optimizer.step(); scheduler.step()
        history.append({"update": update + 1, "loss": loss_value, "grad_norm_pre_clip": grad_norm, "positive_n": int(labels.sum().detach().cpu()), "negative_n": int(len(labels) - labels.sum().detach().cpu()), "microbatch_contexts": MICROBATCH_CONTEXTS, "elapsed_s": time.monotonic() - started})
        current_update = update + 1
        actual_updates = current_update
        if validation_sample is not None and (current_update % EARLY_STOP_EVERY == 0 or current_update == updates):
            val_loss = evaluate_validation_loss(model, encoder, head, validation_sample, cache, device, batch_size=MICROBATCH_CONTEXTS)
            history[-1]["validation_loss"] = val_loss
            if val_loss + EARLY_STOP_MIN_DELTA < best_validation_loss:
                best_validation_loss = val_loss
                best_update = current_update
                bad_validation_checks = 0
                best_state = {
                    "encoder": {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()},
                    "head": {name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
                }
            else:
                bad_validation_checks += 1
            if bad_validation_checks >= EARLY_STOP_PATIENCE and current_update < updates:
                early_stopped = True
        if current_update % max(1, min(100, updates)) == 0 or update == 0 or early_stopped:
            atomic_json(run_dir / "progress.json", {"status": "running", "model": model, "method": method, "partition": partition, "patient_id": patient, "seed": seed, "update": current_update, "total_updates": updates, "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None, "best_update": best_update, "early_stop_bad_checks": bad_validation_checks, "updated_utc": now()})
        # The formal protocol retains fixed intermediate checkpoints.  A
        # short smoke naturally has no checkpoint before its final update.
        if current_update in CHECKPOINT_STEPS and current_update < updates:
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": current_update, "seed": seed, "model": model, "method": method, "partition": partition, "patient_id": patient, "source_checkpoint": str(checkpoint_source.relative_to(ROOT))}, run_dir / f"checkpoint_step_{current_update:04d}.pt")
        if early_stopped:
            break
    if not early_stopped:
        actual_updates = updates
    checkpoint_last = run_dir / "checkpoint_last.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": actual_updates, "seed": seed, "model": model, "method": method, "partition": partition, "patient_id": patient, "source_checkpoint": str(checkpoint_source.relative_to(ROOT)), "selected_by_validation": False}, checkpoint_last)
    selected_update = actual_updates
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"], strict=True)
        head.load_state_dict(best_state["head"], strict=True)
        selected_update = best_update
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "update": selected_update, "training_updates": actual_updates, "seed": seed, "model": model, "method": method, "partition": partition, "patient_id": patient, "source_checkpoint": str(checkpoint_source.relative_to(ROOT)), "selected_by_validation": best_state is not None, "validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None}, checkpoint)
    if not early_stopped and updates in CHECKPOINT_STEPS:
        shutil.copyfile(checkpoint, run_dir / f"checkpoint_step_{updates:04d}.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    manifest = {
        "release_id": "h7-patient-finetune-comparison-v1",
        "status": "training_complete",
        "model": model, "method": method, "partition": partition, "patient_id": patient, "seed": seed,
        "updates": actual_updates,
        "requested_updates": updates,
        "best_validation_update": best_update if best_state is not None else None,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
        "early_stopped": early_stopped,
        "early_stopping": {
            "enabled": bool(validation_sample is not None),
            "monitor": "validation_balanced_bce",
            "every_updates": EARLY_STOP_EVERY,
            "patience": EARLY_STOP_PATIENCE,
            "min_delta": EARLY_STOP_MIN_DELTA,
            "validation_max_per_class": VALIDATION_MAX_PER_CLASS,
        },
        "batch_size": 64,
        "fit_only": False,
        "dev_read": validation_sample is not None,
        "holdout_read": False,
        "training_set_read": True,
        "validation_set_read": validation_sample is not None,
        "test_set_read": False,
        "phase_names": {"fit": "training_set", "dev": "validation_set", "holdout": "test_set"},
        "microbatch_contexts": MICROBATCH_CONTEXTS,
        "label_manifest": str((LABEL_DIR / "manifest.json").relative_to(ROOT)),
        "label_manifest_sha256": sha256(LABEL_DIR / "manifest.json"),
        "checkpoint_source": str(checkpoint_source.relative_to(ROOT)),
        "checkpoint_source_sha256": sha256(checkpoint_source),
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_files": {
            str(path.relative_to(run_dir)): sha256(path)
            for path in sorted(run_dir.glob("checkpoint*.pt"))
        },
        "parameter_report": trainable_parameter_report(nn.ModuleList([encoder, head])),
        "model_meta": model_meta,
        "loss": "BCEWithLogitsLoss(pos_weight=1)",
        "sampler": "exact 32 positive + 32 negative; positive event-uniform; negative EDF-uniform",
        "history": str((run_dir / "history.json").relative_to(ROOT)),
        "created_utc": now(),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "progress.json", {"status": "complete", "update": actual_updates, "total_updates": updates, "best_update": best_update if best_state is not None else None, "early_stopped": early_stopped, "updated_utc": now()})
    print(json.dumps({"status": "complete", "model": model, "method": method, "partition": partition, "patient_id": patient, "seed": seed, "updates": actual_updates, "requested_updates": updates, "best_validation_update": best_update if best_state is not None else None, "early_stopped": early_stopped, "out": str(run_dir)}, indent=2))
    del encoder, head, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def smoke_units(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    selected: list[tuple[str, str, str]] = []
    source_records = load_source_records()
    for model in MODELS:
        units = []
        for (partition, patient), group in table[table.model == model].groupby(["partition", "patient_id"], sort=True):
            fit = group[group.phase == "fit"]
            duration_h = sum(float(row["duration_s"]) for recording, row in source_records.items() if recording.split("/", maxsplit=2)[1] == partition and str(row["patient_id"]) == patient) / 3600.0
            # Smoke is a tensor/gradient contract check, so avoid selecting a
            # multi-day patient merely because it has many labels.  The
            # formal queue still includes every eligible unit.
            if duration_h <= 10.0:
                units.append((int(fit[fit.label_status == "positive"].shape[0]), partition, patient))
        if len(units) < 3:
            # In case a future cohort has fewer short patients, fall back to
            # the complete eligible set rather than silently changing the
            # smoke's three-point design.
            for (partition, patient), group in table[table.model == model].groupby(["partition", "patient_id"], sort=True):
                fit = group[group.phase == "fit"]
                units.append((int(fit[fit.label_status == "positive"].shape[0]), partition, patient))
        units.sort()
        if len(units) < 3:
            raise RuntimeError(f"not enough smoke units for {model}")
        selected.extend((model, p, q) for _, p, q in (units[0], units[len(units) // 2], units[-1]))
    return selected


def run_smoke(args: argparse.Namespace) -> None:
    out = OUT / "smoke"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite smoke namespace: {out}")
    out.mkdir(parents=True)
    table = load_label_table()
    units = smoke_units(table)
    records: list[dict[str, Any]] = []
    for model, partition, patient in units:
        for method in METHODS:
            try:
                result = run_one_train(model, method, partition, patient, args.seed, args.updates, args.device, smoke=True)
                records.append({"model": model, "method": method, "partition": partition, "patient_id": patient, "status": "PASS", "updates": args.updates, "checkpoint_sha256": result["checkpoint_sha256"]})
            except Exception as exc:
                records.append({"model": model, "method": method, "partition": partition, "patient_id": patient, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
                raise
    pd.DataFrame(records).to_csv(out / "smoke_results.csv", index=False)
    atomic_json(out / "manifest.json", {"release_id": "h7-patient-finetune-comparison-v1", "stage": "25-update-smoke", "status": "PASS", "holdout_read": False, "dev_optimization": False, "units": units, "methods": METHODS, "updates": args.updates, "seed": args.seed, "sha256_results": sha256(out / "smoke_results.csv"), "created_utc": now()})
    print(json.dumps({"status": "PASS", "units": len(units), "methods": len(METHODS), "runs": len(records), "out": str(out)}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    table = load_label_table()
    eligible = table[table.phase == "fit"].groupby(["model", "partition", "patient_id"], sort=True).size().reset_index(name="n")
    units = [(str(row.model), str(row.partition), str(row.patient_id)) for row in eligible.itertuples(index=False)]
    runs: list[dict[str, Any]] = []
    for model, partition, patient in units:
        for method in METHODS:
            run_dir = OUT / "runs" / f"{model}__{method}__{partition}__{patient}__seed{args.seed}"
            if run_dir.exists():
                progress_path = run_dir / "progress.json"
                try:
                    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                except Exception:
                    progress = {}
                if progress.get("status") == "complete" and (run_dir / "checkpoint.pt").exists():
                    print(json.dumps({"status": "already_complete", "run": str(run_dir)}), flush=True)
                    continue
                raise RuntimeError(f"refusing to overwrite incomplete run directory: {run_dir}")
            result = run_one_train(model, method, partition, patient, args.seed, args.updates, args.device)
            runs.append(result)
    atomic_json(OUT / f"train_seed{args.seed}_summary.json", {"status": "complete", "seed": args.seed, "runs_completed": len(runs), "total_expected": len(units) * len(METHODS), "created_utc": now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--updates", type=int, default=25)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--device", default=None)
    train = sub.add_parser("train")
    train.add_argument("--updates", type=int, default=1000)
    train.add_argument("--seed", type=int, choices=SEED_VALUES, default=17)
    train.add_argument("--device", default=None)
    args = parser.parse_args()
    args.device = device_from_arg(args.device)
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "train":
        command_train(args)


if __name__ == "__main__":
    main()
