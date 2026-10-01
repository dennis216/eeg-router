from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
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
from bfa.training.matrix import content_addressed_run_id
from bfa.training.sampler import PatientBalancedSampler

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "experiment" / "v3_unified.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def configure_determinism(seed: int) -> None:
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


def build_encoder(model_name: str) -> torch.nn.Module:
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
        encoder = MultiScaleTCNGAT()
        encoder.native_head.requires_grad_(False)
        return encoder
    raise ValueError(f"unknown model {model_name}")


def indexes_for_split(
    windows: pd.DataFrame, split: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = windows[windows.patient.isin(split["train"]) & windows.train_eligible].reset_index(
        drop=True
    )
    validation = windows[windows.patient.isin(split["validation"]) & ~windows.warmup].reset_index(
        drop=True
    )
    test = windows[windows.patient.isin(split["test"]) & ~windows.warmup].reset_index(drop=True)
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("all split partitions must have non-empty indexes")
    if set(train.patient) & set(validation.patient):
        raise ValueError("patient leakage from train to validation")
    if set(train.patient) & set(test.patient):
        raise ValueError("patient leakage from train to test")
    if set(validation.patient) & set(test.patient):
        raise ValueError("patient leakage from validation to test")
    return train, validation, test


def balanced_smoke_validation(index: pd.DataFrame) -> pd.DataFrame:
    positive = index[index.label == 1].head(32)
    negative = index[index.label == 0].head(32)
    output = pd.concat([positive, negative], ignore_index=True)
    if output.label.nunique() != 2:
        raise RuntimeError("smoke validation subset must include both labels")
    return output


def make_dataset(
    index: pd.DataFrame, model_name: str, config: dict[str, Any]
) -> CachedContextDataset | CachedFeatureContextDataset:
    inputs = config["inputs"]
    quality_root = Path(inputs["quality_cache_root"])
    if model_name == "tcn_gat":
        return CachedContextDataset(
            index,
            Path(inputs["tcn_cache_root"]),
            model_name,
            quality_root=quality_root,
            max_open_recordings=32,
        )
    return CachedFeatureContextDataset(
        index,
        Path(inputs["foundation_feature_cache_root"]),
        model_name,
        quality_root=quality_root,
        max_open_recordings=32,
    )


def assert_cache_available(index: pd.DataFrame, model_name: str, config: dict[str, Any]) -> None:
    if not len(index):
        raise ValueError("cache check requires a non-empty index")
    relative_path = Path(str(index.relative_path.iloc[0])).with_suffix(".npy")
    inputs = config["inputs"]
    if model_name == "tcn_gat":
        required = [
            Path(inputs["tcn_cache_root"]) / model_name / relative_path,
            Path(inputs["quality_cache_root"]) / "quality" / relative_path,
        ]
    else:
        required = [
            Path(inputs["foundation_feature_cache_root"]) / model_name / relative_path,
            Path(inputs["quality_cache_root"]) / "quality" / relative_path,
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen cache is absent: " + ", ".join(missing))


@torch.inference_mode()
def predict(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    index: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    *,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    dataset = make_dataset(index, model_name, config)
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
    outputs: list[np.ndarray] = []
    for batch_number, batch in enumerate(loader, start=1):
        inputs = batch["x"].cuda(non_blocking=True)
        quality = batch["quality"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = encoder.forward_window_sequence(inputs, quality)
            probabilities = torch.sigmoid(head(embeddings))
        outputs.append(probabilities.float().cpu().numpy())
        if batch_number % 1000 == 0:
            print(f"V3_INFERENCE_PROGRESS batches={batch_number}/{len(loader)}", flush=True)
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def probability_table(index: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if len(index) != len(probabilities):
        raise ValueError("probabilities do not align with the frozen index")
    output = index[
        ["patient", "recording", "start", "end", "label", "warmup", "relative_path"]
    ].copy()
    output["probability"] = probabilities
    return output


def truth_intervals(
    seizures: pd.DataFrame, recording_id: str, evaluation_start_s: float
) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id == recording_id]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def score_probabilities(
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
        truths = truth_intervals(seizures, str(recording_id), evaluation_start_s)
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


def threshold_sweep(
    table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = [
        score_probabilities(table, seizures, recordings, float(threshold))
        for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2)
    ]
    selected = select_threshold(
        [
            ThresholdScore(
                threshold=float(row["threshold"]),
                event_sensitivity=float(row["event_sensitivity"]),
                fa_per_24h=float(row["fa_per_24h"]),
            )
            for row in rows
        ],
        target_sensitivity=0.80,
    )
    selected_row = next(row for row in rows if row["threshold"] == selected.threshold)
    return pd.DataFrame(rows), selected_row


def validate(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    index: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    run_dir: Path,
    epoch: int,
    seizures: pd.DataFrame,
    recordings: pd.DataFrame,
    workers: int,
) -> dict[str, Any]:
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    probability_path = validation_dir / f"probabilities_epoch_{epoch:02d}.parquet"
    if probability_path.exists():
        table = pd.read_parquet(probability_path)
    else:
        probabilities = predict(
            encoder, head, index, model_name, config, batch_size=16, workers=workers
        )
        table = probability_table(index, probabilities)
        table.to_parquet(probability_path, index=False)
    labeled = table.label.notna()
    labels = table.loc[labeled, "label"].astype(int)
    probabilities = table.loc[labeled, "probability"]
    if labels.nunique() != 2:
        raise RuntimeError("validation labels must include both classes")
    sweep, selected = threshold_sweep(table, seizures, recordings)
    sweep_path = validation_dir / f"threshold_sweep_epoch_{epoch:02d}.parquet"
    sweep.to_parquet(sweep_path, index=False)
    metrics = {
        "epoch": epoch,
        "window_auroc": float(roc_auc_score(labels, probabilities)),
        "window_average_precision": float(average_precision_score(labels, probabilities)),
        "selected_event_operating_point": selected,
        "probability_sha256": sha256_file(probability_path),
        "threshold_sweep_sha256": sha256_file(sweep_path),
    }
    (validation_dir / f"metrics_epoch_{epoch:02d}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int | None,
    best_metrics: dict[str, Any] | None,
    no_improvement_epochs: int,
    run_manifest: dict[str, Any],
    sampler: PatientBalancedSampler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "no_improvement_epochs": no_improvement_epochs,
        "sampler": sampler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "run_manifest": run_manifest,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n")


def load_checkpoint(
    path: Path,
    *,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["encoder"], strict=True)
    head.load_state_dict(state["head"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def is_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_auroc = float(candidate["window_auroc"])
    current_auroc = float(current["window_auroc"])
    if candidate_auroc != current_auroc:
        return candidate_auroc > current_auroc
    candidate_fa = float(candidate["selected_event_operating_point"]["fa_per_24h"])
    current_fa = float(current["selected_event_operating_point"]["fa_per_24h"])
    return candidate_fa < current_fa


def make_loader(
    index: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    *,
    model_seed: int,
    split_seed: int,
    epoch: int,
    updates: int,
) -> tuple[DataLoader[Any], PatientBalancedSampler]:
    training = config["training"]
    micro_batch = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation"])
    sampler = PatientBalancedSampler(
        index,
        batch_size=micro_batch,
        positive_fraction=float(training["target_positive_fraction"]),
        seed=int(model_seed * 1_000_000 + split_seed * 1_000 + epoch),
        epoch_size=updates * accumulation * micro_batch,
    )
    workers = int(training["dataloader_workers"])
    loader = DataLoader(
        make_dataset(index, model_name, config),
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    return loader, sampler


def train_epoch(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    updates: int,
    accumulation: int,
    pos_weight: torch.Tensor,
    gradient_clip_norm: float,
    epoch: int,
    total_epochs: int,
) -> tuple[float, PatientBalancedSampler]:
    encoder.train()
    head.train()
    iterator = iter(loader)
    trainable = [
        parameter
        for module in (encoder, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    losses: list[float] = []
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
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
                (raw_loss / accumulation).backward()
            update_loss += float(raw_loss.detach())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm))
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        losses.append(update_loss / accumulation)
        if update % 25 == 0 or update == updates:
            print(
                f"V3_TRAIN_PROGRESS epoch={epoch}/{total_epochs} update={update}/{updates} "
                f"loss={losses[-1]:.6f}",
                flush=True,
            )
    sampler = loader.sampler
    if not isinstance(sampler, PatientBalancedSampler):
        raise TypeError("v3 loader must use PatientBalancedSampler")
    return float(np.mean(losses)), sampler


def readonly_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen file {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-namespace",
        default=None,
        help="override the output namespace for a frozen execution subset",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    experiment = config["experiment"]
    if args.model not in experiment["models"]:
        raise ValueError("model is outside the frozen v3 matrix")
    if args.model_seed not in experiment["model_seeds"]:
        raise ValueError("model seed is outside the frozen v3 matrix")
    if args.split_seed not in experiment["split_seeds"]:
        raise ValueError("split seed is outside the frozen v3 matrix")
    if not torch.cuda.is_available():
        raise RuntimeError("full v3 runner requires CUDA")

    split_path = ROOT / "manifests" / "splits" / f"split_{args.split_seed}.json"
    split = json.loads(split_path.read_text())
    windows = pd.read_parquet(ROOT / config["inputs"]["window_manifest"])
    recordings = pd.read_parquet(ROOT / config["inputs"]["recordings_manifest"])
    seizures = pd.read_parquet(ROOT / config["inputs"]["seizures_manifest"])
    train_index, validation_index, test_index = indexes_for_split(windows, split)
    assert_cache_available(train_index, args.model, config)
    assert_cache_available(validation_index, args.model, config)
    assert_cache_available(test_index, args.model, config)

    training = config["training"]
    micro_batch = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation"])
    effective_batch = micro_batch * accumulation
    updates_per_epoch = math.ceil(len(train_index) / effective_batch)
    max_epochs = int(training["max_epochs"])
    if args.smoke_updates:
        max_epochs = 1
        updates_per_epoch = args.smoke_updates
        validation_index = balanced_smoke_validation(validation_index)

    config_sha = sha256_file(CONFIG_PATH)
    split_sha = sha256_file(split_path)
    commit = git_commit()
    run_id = args.run_id or content_addressed_run_id(
        configuration="unified",
        model=args.model,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        config_sha256=config_sha,
        split_sha256=split_sha,
        code_commit=commit,
    )
    namespace = (
        args.output_namespace
        if args.output_namespace is not None
        else ("v3-full-smoke" if args.smoke_updates else experiment["output_namespace"])
    )
    run_dir = ROOT / "runs" / namespace / "unified" / run_id
    dry_run = {
        "run_id": run_id,
        "namespace": namespace,
        "model": args.model,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "updates_per_epoch": updates_per_epoch,
        "max_epochs": max_epochs,
        "effective_batch_size": effective_batch,
        "config_sha256": config_sha,
        "split_sha256": split_sha,
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.dry_run:
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return
    if (run_dir / "completed.json").exists():
        print(f"V3_RUN_ALREADY_COMPLETE run_id={run_id}", flush=True)
        return
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"run directory exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_determinism(args.model_seed)
    encoder = build_encoder(args.model).cuda()
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
        optimizer,
        T_max=max_epochs * updates_per_epoch,
        eta_min=float(training["minimum_learning_rate"]),
    )
    manifest_path = run_dir / "run_manifest.json"
    run_manifest = {
        "protocol": config["protocol"],
        "run_id": run_id,
        "configuration": "unified",
        "model": args.model,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "split_manifest": str(split_path.relative_to(ROOT)),
        "split_sha256": split_sha,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": config_sha,
        "code_commit": commit,
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "updates_per_epoch": updates_per_epoch,
        "max_epochs": max_epochs,
        "effective_batch_size": effective_batch,
        "test_accessed": False,
        "started_at_unix": time.time(),
    }
    start_epoch = 1
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    if args.resume:
        last_checkpoint = run_dir / "checkpoints" / "last.pt"
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"cannot resume without {last_checkpoint}")
        state = load_checkpoint(
            last_checkpoint,
            encoder=encoder,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        start_epoch = int(state["epoch"]) + 1
        best_epoch = state.get("best_epoch")
        best_metrics = state.get("best_metrics")
        no_improvement = int(state.get("no_improvement_epochs", 0))
        run_manifest = state["run_manifest"]
        run_manifest["resumed_at_unix"] = time.time()
    else:
        manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    positives = float((train_index.label == 1).sum())
    negatives = float((train_index.label == 0).sum())
    pos_weight = torch.tensor(negatives / positives, device="cuda")
    if not args.resume:
        no_improvement = 0
    history_path = run_dir / "training_history.jsonl"
    started = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        loader, sampler = make_loader(
            train_index,
            args.model,
            config,
            model_seed=args.model_seed,
            split_seed=args.split_seed,
            epoch=epoch,
            updates=updates_per_epoch,
        )
        loss, sampler = train_epoch(
            encoder,
            head,
            loader,
            optimizer=optimizer,
            scheduler=scheduler,
            updates=updates_per_epoch,
            accumulation=accumulation,
            pos_weight=pos_weight,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            epoch=epoch,
            total_epochs=max_epochs,
        )
        metrics = validate(
            encoder,
            head,
            validation_index,
            args.model,
            config,
            run_dir,
            epoch,
            seizures,
            recordings,
            int(training["dataloader_workers"]),
        )
        improved = is_better(metrics, best_metrics)
        if improved:
            best_epoch = epoch
            best_metrics = metrics
            no_improvement = 0
        else:
            no_improvement += 1
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"
        save_checkpoint(
            checkpoint,
            encoder=encoder,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            no_improvement_epochs=no_improvement,
            run_manifest=run_manifest,
            sampler=sampler,
        )
        shutil.copy2(checkpoint, run_dir / "checkpoints" / "last.pt")
        (run_dir / "checkpoints" / "last.pt.sha256").write_text(
            sha256_file(run_dir / "checkpoints" / "last.pt") + "\n"
        )
        history = {
            "epoch": epoch,
            "loss": loss,
            "validation": metrics,
            "improved": improved,
            "no_improvement_epochs": no_improvement,
            "elapsed_seconds": time.monotonic() - started,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with history_path.open("a") as stream:
            stream.write(json.dumps(history, sort_keys=True) + "\n")
        print("V3_VALIDATION_COMPLETE " + json.dumps(history, sort_keys=True), flush=True)
        if no_improvement >= int(training["early_stopping_patience"]):
            print(f"V3_EARLY_STOP epoch={epoch}", flush=True)
            break

    if args.smoke_updates:
        print(f"V3_SMOKE_OK model={args.model} updates={args.smoke_updates}", flush=True)
        return
    if best_epoch is None or best_metrics is None:
        raise RuntimeError("full v3 run ended without a validation checkpoint")
    best_checkpoint = run_dir / "checkpoints" / f"epoch_{best_epoch:02d}.pt"
    threshold_lock = run_dir / "threshold_lock.json"
    threshold = float(best_metrics["selected_event_operating_point"]["threshold"])
    readonly_json(
        threshold_lock,
        {
            "source": "validation_only",
            "best_epoch": best_epoch,
            "threshold": threshold,
            "selected_event_operating_point": best_metrics["selected_event_operating_point"],
            "checkpoint": str(best_checkpoint.relative_to(run_dir)),
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "validation_probability_sha256": best_metrics["probability_sha256"],
        },
    )
    if (run_dir / "test" / "metrics.json").exists():
        raise RuntimeError("frozen v3 test was already evaluated for this run")
    load_checkpoint(best_checkpoint, encoder=encoder, head=head)
    run_manifest["test_accessed"] = True
    run_manifest["best_epoch"] = best_epoch
    run_manifest["validation_threshold"] = threshold
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    test_probabilities = predict(
        encoder,
        head,
        test_index,
        args.model,
        config,
        batch_size=16,
        workers=int(training["dataloader_workers"]),
    )
    test_table = probability_table(test_index, test_probabilities)
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_probability_path = test_dir / "probabilities.parquet"
    test_table.to_parquet(test_probability_path, index=False)
    test_metrics = {
        "test_evaluation_count": 1,
        "threshold_source": "validation_only_best_checkpoint",
        "score": score_probabilities(test_table, seizures, recordings, threshold),
        "probability_sha256": sha256_file(test_probability_path),
    }
    (test_dir / "metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True) + "\n"
    )
    completed = {
        "run_id": run_id,
        "model": args.model,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "best_epoch": best_epoch,
        "training_and_validation_wall_seconds": time.monotonic() - started,
        "final_validation": best_metrics,
        "test": test_metrics,
        "threshold_lock_sha256": sha256_file(threshold_lock),
        "checkpoint_sha256": sha256_file(best_checkpoint),
    }
    (run_dir / "completed.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
    print("V3_UNIFIED_COMPLETE " + json.dumps(completed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
