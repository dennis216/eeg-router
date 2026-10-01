from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional
from torch.utils.data import DataLoader

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events
from bfa.evaluation.metrics import ThresholdScore, select_threshold
from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.precomputed import PrecomputedProjectionEncoder
from bfa.models.shared_head import SharedContextHead
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.training.dataset import CachedContextDataset, CachedFeatureContextDataset
from bfa.training.sampler import PatientBalancedSampler


ROOT = Path(__file__).resolve().parent.parent
PILOT_CONFIG = ROOT / "configs/pilot/bsn_pilot_v1.yaml"
V3_CACHE = Path("/mnt/d/EEGData/bfa_cache_v3")
PILOT_CACHE = Path("/mnt/d/EEGData/bfa_cache_bsn_pilot_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _model(model_name: str) -> torch.nn.Module:
    if model_name == "singlem":
        adapter = SingLEMAdapter(
            ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
        )
        return PrecomputedProjectionEncoder(16, adapter.projection)
    if model_name == "cbramod":
        adapter = CBraModAdapter(
            ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
        )
        return PrecomputedProjectionEncoder(200, adapter.projection)
    if model_name == "tcn_gat":
        model = MultiScaleTCNGAT()
        model.native_head.requires_grad_(False)
        return model
    raise ValueError(f"unknown model: {model_name}")


def _cache_root(model_name: str) -> Path:
    return V3_CACHE if model_name == "tcn_gat" else PILOT_CACHE


def _save_checkpoint(
    path: Path,
    *,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: PatientBalancedSampler,
    update: int,
    consumed_samples: int,
    run_manifest: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(consumed_cursor=consumed_samples),
        "update": update,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "run_manifest": run_manifest,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(_sha256(path) + "\n")


def _restore_checkpoint(
    path: Path,
    *,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: PatientBalancedSampler,
) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = encoder.load_state_dict(state["encoder"], strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("encoder.") and not key.startswith("backbone.")
    ]
    if incompatible.missing_keys or unexpected:
        raise ValueError(
            f"checkpoint encoder mismatch missing={incompatible.missing_keys} "
            f"unexpected={unexpected}"
        )
    head.load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["update"])


@torch.inference_mode()
def _predict(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    index: pd.DataFrame,
    model_name: str,
    *,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    if model_name == "tcn_gat":
        dataset = CachedContextDataset(
            index,
            _cache_root(model_name),
            model_name,
            quality_root=V3_CACHE,
            max_open_recordings=32,
        )
    else:
        dataset = CachedFeatureContextDataset(
            index,
            Path("/mnt/d/EEGData/bfa_features_bsn_pilot_v1"),
            model_name,
            quality_root=V3_CACHE,
            max_open_recordings=32,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    encoder.eval()
    head.eval()
    probabilities: list[np.ndarray] = []
    for batch_number, batch in enumerate(loader, start=1):
        inputs = batch["x"].cuda(non_blocking=True)
        quality = batch["quality"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = encoder.forward_window_sequence(inputs, quality)
            probability = torch.sigmoid(head(embeddings))
        probabilities.append(probability.float().cpu().numpy())
        if batch_number % 1000 == 0:
            print(f"INFERENCE_PROGRESS batches={batch_number}/{len(loader)}", flush=True)
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)


def _truth_intervals(
    seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float
) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def _score_probabilities(
    table: pd.DataFrame,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    true_positives = 0
    false_alarms = 0
    truth_events = 0
    nonseizure_seconds = 0.0
    for recording_id, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        evaluation_start_s = float(ordered.end.iloc[0])
        duration_s = float(
            recordings.loc[recordings.recording_id == recording_id, "duration_s"].iloc[0]
        )
        truths = _truth_intervals(seizures, str(recording_id), evaluation_start_s)
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=threshold,
        )
        matched = match_events(predictions, truths)
        true_positives += len(matched.pairs)
        false_alarms += len(matched.unmatched_predictions)
        truth_events += len(truths)
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        nonseizure_seconds += max(0.0, duration_s - evaluation_start_s - seizure_seconds)
    sensitivity = true_positives / truth_events if truth_events else 1.0
    nonseizure_hours = nonseizure_seconds / 3600.0
    return {
        "threshold": threshold,
        "true_positive_events": true_positives,
        "false_alarm_events": false_alarms,
        "truth_events": truth_events,
        "event_sensitivity": sensitivity,
        "nonseizure_hours": nonseizure_hours,
        "fa_per_24h": false_alarms * 24.0 / nonseizure_hours,
    }


def _threshold_sweep(
    table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = [
        _score_probabilities(table, seizures, recordings, float(threshold))
        for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2)
    ]
    scores = [
        ThresholdScore(
            threshold=float(row["threshold"]),
            event_sensitivity=float(row["event_sensitivity"]),
            fa_per_24h=float(row["fa_per_24h"]),
        )
        for row in rows
    ]
    selected = select_threshold(scores, target_sensitivity=0.80)
    selected_row = next(row for row in rows if row["threshold"] == selected.threshold)
    return pd.DataFrame(rows), selected_row


def _probability_table(index: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if len(index) != len(probabilities):
        raise ValueError("probabilities do not align with the frozen index")
    output = index[
        ["patient", "recording", "start", "end", "label", "warmup", "relative_path"]
    ].copy()
    output["probability"] = probabilities
    return output


def _validate(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    validation_index: pd.DataFrame,
    model_name: str,
    run_dir: Path,
    update: int,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    workers: int,
) -> dict[str, Any]:
    probability_path = run_dir / "validation" / f"probabilities_step_{update:05d}.parquet"
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    if probability_path.exists():
        table = pd.read_parquet(probability_path)
    else:
        probabilities = _predict(
            encoder, head, validation_index, model_name, batch_size=16, workers=workers
        )
        table = _probability_table(validation_index, probabilities)
        table.to_parquet(probability_path, index=False)
    labeled = table.label.notna()
    labels = table.loc[labeled, "label"].astype(int)
    probabilities = table.loc[labeled, "probability"]
    sweep, selected = _threshold_sweep(table, seizures, recordings)
    sweep.to_parquet(
        run_dir / "validation" / f"threshold_sweep_step_{update:05d}.parquet", index=False
    )
    metrics = {
        "update": update,
        "window_auroc": float(roc_auc_score(labels, probabilities)),
        "window_average_precision": float(average_precision_score(labels, probabilities)),
        "selected_event_operating_point": selected,
        "probability_sha256": _sha256(probability_path),
    }
    (run_dir / "validation" / f"metrics_step_{update:05d}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def _test_once(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    test_index: pd.DataFrame,
    model_name: str,
    run_dir: Path,
    threshold: float,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    workers: int,
) -> dict[str, Any]:
    metrics_path = run_dir / "test" / "metrics.json"
    if metrics_path.exists():
        raise RuntimeError("frozen test has already been evaluated for this run")
    probability_path = run_dir / "test" / "probabilities.parquet"
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = _predict(
        encoder, head, test_index, model_name, batch_size=16, workers=workers
    )
    table = _probability_table(test_index, probabilities)
    table.to_parquet(probability_path, index=False)
    score = _score_probabilities(table, seizures, recordings, threshold)
    metrics = {
        "threshold_source": "final_step_5000_validation_only",
        "test_evaluation_count": 1,
        "score": score,
        "probability_sha256": _sha256(probability_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument(
        "--run-id",
        default="BSN-pilot-v1",
        help="immutable output namespace; use a new id for any post-pilot extension",
    )
    args = parser.parse_args()
    config = yaml.safe_load(PILOT_CONFIG.read_text())
    training = config["training"]
    total_updates = int(training["optimizer_updates"])
    validation_every = int(training["full_validation_every_updates"])
    if args.smoke_updates:
        total_updates = args.smoke_updates
        validation_every = total_updates
    seed = (
        int(args.model_seed)
        if args.model_seed is not None
        else int(config["cohort"]["model_seed"])
    )
    split_seed = (
        int(args.split_seed)
        if args.split_seed is not None
        else int(config["cohort"]["split_seed"])
    )
    if seed not in {17, 42, 3407}:
        raise ValueError(f"model seed is not pre-registered in v3: {seed}")
    if split_seed not in {17, 42, 3407, 2026, 777}:
        raise ValueError(f"split seed is not pre-registered in v3: {split_seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    split_file = ROOT / "manifests" / "splits" / f"split_{split_seed}.json"
    split = json.loads(split_file.read_text())
    windows = pd.read_parquet(ROOT / config["cohort"]["window_manifest"])
    recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet")
    seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    train_index = windows[
        windows.patient.isin(split["train"]) & windows.train_eligible
    ].reset_index(drop=True)
    validation_index = windows[
        windows.patient.isin(split["validation"]) & ~windows.warmup
    ].reset_index(drop=True)
    test_index = windows[
        windows.patient.isin(split["test"]) & ~windows.warmup
    ].reset_index(drop=True)
    if args.smoke_updates:
        # Keep the smoke validation small, but deterministic and class-complete.
        # Taking the first rows can select only background windows and makes
        # AUROC undefined, hiding a broken positive path.
        smoke_positive = validation_index[validation_index.label == 1].head(32)
        smoke_negative = validation_index[validation_index.label == 0].head(32)
        validation_index = pd.concat(
            [smoke_positive, smoke_negative], ignore_index=True
        )
        if validation_index.label.nunique() < 2:
            raise RuntimeError("smoke validation subset must contain both labels")
    micro_batch = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation"])
    total_microbatches = total_updates * accumulation
    sampler = PatientBalancedSampler(
        train_index,
        batch_size=micro_batch,
        positive_fraction=float(training["target_positive_fraction"]),
        seed=seed,
        epoch_size=total_microbatches * micro_batch,
    )
    if args.model == "tcn_gat":
        train_dataset = CachedContextDataset(
            train_index,
            _cache_root(args.model),
            args.model,
            quality_root=V3_CACHE,
            max_open_recordings=32,
        )
    else:
        train_dataset = CachedFeatureContextDataset(
            train_index,
            Path("/mnt/d/EEGData/bfa_features_bsn_pilot_v1"),
            args.model,
            quality_root=V3_CACHE,
            max_open_recordings=32,
        )
    workers = int(training["dataloader_workers"])
    loader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    encoder = _model(args.model).cuda()
    head = SharedContextHead().cuda()
    trainable = [
        parameter
        for module in (encoder, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(training["minimum_learning_rate"])
    )
    run_suffix = f"smoke_{args.smoke_updates}" if args.smoke_updates else "main"
    run_dir = (
        ROOT
        / "runs"
        / args.run_id
        / args.model
        / f"split{split_seed}_seed{seed}_{run_suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "pilot_id": args.run_id,
        "model": args.model,
        "split": split_seed,
        "seed": seed,
        "split_file": str(split_file.relative_to(ROOT)),
        "seed_source": "pre_registered_B_Project_Protocol_v3",
        "pilot_config_sha256": _sha256(PILOT_CONFIG),
        "code_commit": _git_commit(),
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "optimizer_updates": total_updates,
        "effective_batch_size": micro_batch * accumulation,
        "started_at_unix": time.time(),
        "test_accessed": False,
        "frozen_backbone_feature_cache": args.model != "tcn_gat",
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and args.resume is None:
        raise RuntimeError(f"run directory already exists; use --resume: {run_dir}")
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    positives = float((train_index.label == 1).sum())
    negatives = float((train_index.label == 0).sum())
    pos_weight = torch.tensor(negatives / positives, device="cuda")
    start_update = 0
    if args.resume is not None:
        start_update = _restore_checkpoint(
            args.resume,
            encoder=encoder,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
        )
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    history_path = run_dir / "training_history.jsonl"
    run_started = time.monotonic()
    for update in range(start_update + 1, total_updates + 1):
        update_loss = 0.0
        update_started = time.monotonic()
        for _ in range(accumulation):
            batch = next(iterator)
            inputs = batch["x"].cuda(non_blocking=True)
            quality = batch["quality"].cuda(non_blocking=True)
            labels = batch["y"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = encoder.forward_window_sequence(inputs, quality)
                logits = head(embeddings)
                raw_loss = functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
                loss = raw_loss / accumulation
            loss.backward()
            update_loss += float(raw_loss.detach())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        )
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        history = {
            "update": update,
            "loss": update_loss / accumulation,
            "gradient_norm": gradient_norm,
            "learning_rate": scheduler.get_last_lr()[0],
            "update_seconds": time.monotonic() - update_started,
            "elapsed_seconds": time.monotonic() - run_started,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with history_path.open("a") as stream:
            stream.write(json.dumps(history, sort_keys=True) + "\n")
        if update % 25 == 0:
            print(
                f"TRAIN_PROGRESS model={args.model} update={update}/{total_updates} "
                f"loss={history['loss']:.6f}",
                flush=True,
            )
        if update % validation_every == 0:
            checkpoint = run_dir / "checkpoints" / f"step_{update:05d}.pt"
            _save_checkpoint(
                checkpoint,
                encoder=encoder,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                update=update,
                consumed_samples=update * accumulation * micro_batch,
                run_manifest=run_manifest,
            )
            validation_metrics = _validate(
                encoder,
                head,
                validation_index,
                args.model,
                run_dir,
                update,
                seizures,
                recordings,
                workers,
            )
            print(
                "VALIDATION_COMPLETE " + json.dumps(validation_metrics, sort_keys=True),
                flush=True,
            )
            encoder.train()
            head.train()

    if args.smoke_updates:
        print(f"PILOT_SMOKE_OK model={args.model} updates={total_updates}", flush=True)
        return
    final_validation = json.loads(
        (run_dir / "validation" / "metrics_step_05000.json").read_text()
    )
    threshold = float(final_validation["selected_event_operating_point"]["threshold"])
    run_manifest["validation_threshold"] = threshold
    run_manifest["test_accessed"] = True
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    test_metrics = _test_once(
        encoder,
        head,
        test_index,
        args.model,
        run_dir,
        threshold,
        seizures,
        recordings,
        workers,
    )
    completed = {
        "model": args.model,
        "optimizer_updates": total_updates,
        "training_and_validation_wall_seconds": time.monotonic() - run_started,
        "final_validation": final_validation,
        "test": test_metrics,
    }
    (run_dir / "completed.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n"
    )
    print("PILOT_MODEL_COMPLETE " + json.dumps(completed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
