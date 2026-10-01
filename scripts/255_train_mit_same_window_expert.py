from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from bfa.models.shared_head import SharedContextHead
from bfa.same_window_moe import aggregate_components, eventize_common, score_events
from bfa.training.dataset import CachedContextDataset, CachedFeatureContextDataset
from bfa.training.sampler import PatientBalancedSampler


ROOT = Path("/root/b_false_alarm_atlas")
OUTPUT = ROOT / "outputs/reports/same-window-router-first-moe-v1/mit_experts"


def load_v3():
    path = ROOT / "scripts/28_run_v3_unified.py"
    spec = importlib.util.spec_from_file_location("same_window_v3_helpers", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v3 = load_v3()


def config() -> dict[str, object]:
    return {"inputs": {"tcn_cache_root": "/mnt/d/EEGData/bfa_cache_v3", "foundation_feature_cache_root": "/mnt/d/EEGData/bfa_features_v3_official_noclip", "quality_cache_root": "/mnt/d/EEGData/bfa_cache_v3"}, "training": {"micro_batch_size": 16, "gradient_accumulation": 4, "target_positive_fraction": 0.30, "dataloader_workers": 8}}


def index_parts(fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = json.loads((ROOT / f"manifests/groupkfold_cv_v1/fold_{fold}.json").read_text())
    windows = pd.read_parquet(ROOT / "manifests/windows.parquet")
    train = windows[windows.patient.isin(split["train"]) & windows.label.notna()].reset_index(drop=True)
    val = windows[windows.patient.isin(split["validation"]) & windows.label.notna()].reset_index(drop=True)
    test = windows[windows.patient.isin(split["test"]) & windows.label.notna()].reset_index(drop=True)
    if set(train.patient) & set(val.patient) or set(train.patient) & set(test.patient) or set(val.patient) & set(test.patient):
        raise RuntimeError("MIT patient leakage")
    return train, val, test


def loader(index: pd.DataFrame, model: str, *, training: bool, seed: int, epoch_size: int = 0, positive_fraction: float = 0.30) -> DataLoader:
    cfg = config()["inputs"]
    if model == "tcn_gat":
        dataset = CachedContextDataset(
            index, Path(cfg["tcn_cache_root"]), model,
            quality_root=Path(cfg["quality_cache_root"]), max_open_recordings=32,
            context_windows=1,
        )
    else:
        dataset = CachedFeatureContextDataset(
            index, Path(cfg["foundation_feature_cache_root"]), model,
            quality_root=Path(cfg["quality_cache_root"]), max_open_recordings=32,
            context_windows=1,
        )
    kwargs = dict(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    if training:
        sampler = PatientBalancedSampler(index, batch_size=64, positive_fraction=positive_fraction, seed=seed, epoch_size=epoch_size)
        return DataLoader(dataset, batch_size=64, sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=128, shuffle=False, **kwargs)


def current_embeddings(encoder, batch):
    x = batch["x"].cuda(non_blocking=True)[:, -1:]
    quality = batch["quality"].cuda(non_blocking=True)[:, -1:]
    return encoder.forward_window_sequence(x, quality)


@torch.inference_mode()
def predict(encoder, head, index: pd.DataFrame, model: str) -> np.ndarray:
    encoder.eval(); head.eval(); pieces = []
    for batch in loader(index, model, training=False, seed=3407):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pieces.append(torch.sigmoid(head(current_embeddings(encoder, batch))).float().cpu().numpy())
    return np.concatenate(pieces)


def raw_truths(seizures: pd.DataFrame, recording: str) -> list[tuple[float, float]]:
    return [(float(row.start_s), float(row.end_s)) for row in seizures[seizures.recording_id.astype(str).eq(str(recording))].itertuples()]


def score(index: pd.DataFrame, probability: np.ndarray, threshold: float, recordings: pd.DataFrame, seizures: pd.DataFrame) -> dict[str, float | int]:
    table = index[["recording", "end"]].copy(); table["probability"] = probability
    durations = recordings.set_index("recording_id").duration_s.to_dict(); parts = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end")
        events = eventize_common(
            ordered.end.to_numpy(float),
            ordered.probability.to_numpy(float),
            threshold=threshold,
        )
        parts.append(score_events(events, raw_truths(seizures, str(recording)), float(ordered.end.min()), float(durations[str(recording)])))
    return aggregate_components(parts)


def select(index, probability, recordings, seizures):
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        rows.append({"threshold": float(threshold), **score(index, probability, float(threshold), recordings, seizures)})
    curve = pd.DataFrame(rows)
    feasible = curve[curve.sensitivity >= 0.80]
    if len(feasible):
        pool = feasible.copy()
        pool["fa_rank"] = pool.fa_per_24h.rank(method="average", pct=True)
        pool["false_alarm_time_rank"] = pool.false_alarm_time_min_per_24h.rank(method="average", pct=True)
        pool["joint_false_alarm_score"] = pool.fa_rank + pool.false_alarm_time_rank
        chosen = pool.sort_values(
            ["joint_false_alarm_score", "false_alarm_time_min_per_24h", "fa_per_24h", "threshold"],
            ascending=[True, True, True, False],
        ).iloc[0]
    else:
        pool = curve[curve.sensitivity == curve.sensitivity.max()].copy()
        pool["fa_rank"] = pool.fa_per_24h.rank(method="average", pct=True)
        pool["false_alarm_time_rank"] = pool.false_alarm_time_min_per_24h.rank(method="average", pct=True)
        pool["joint_false_alarm_score"] = pool.fa_rank + pool.false_alarm_time_rank
        chosen = pool.sort_values(
            ["joint_false_alarm_score", "false_alarm_time_min_per_24h", "fa_per_24h", "threshold"],
            ascending=[True, True, True, False],
        ).iloc[0]
    return float(chosen.threshold), curve, float(chosen.joint_false_alarm_score)


def checkpoint_key(metric):
    if float(metric["sensitivity"]) >= 0.80:
        return (0, float(metric["joint_false_alarm_score"]), float(metric["false_alarm_time_min_per_24h"]), float(metric["fa_per_24h"]), -float(metric["sensitivity"]))
    return (1, -float(metric["sensitivity"]), float(metric["joint_false_alarm_score"]), float(metric["false_alarm_time_min_per_24h"]), float(metric["fa_per_24h"]))


def select_checkpoint(history: list[dict[str, object]]) -> pd.Series:
    table = pd.DataFrame(history)
    feasible = table[table.sensitivity >= 0.80]
    pool = feasible.copy() if len(feasible) else table[table.sensitivity == table.sensitivity.max()].copy()
    pool["fa_checkpoint_rank"] = pool.fa_per_24h.rank(method="average", pct=True)
    pool["false_alarm_time_checkpoint_rank"] = pool.false_alarm_time_min_per_24h.rank(method="average", pct=True)
    pool["checkpoint_joint_score"] = pool.fa_checkpoint_rank + pool.false_alarm_time_checkpoint_rank
    return pool.sort_values(
        ["checkpoint_joint_score", "false_alarm_time_min_per_24h", "fa_per_24h", "sensitivity", "update"],
        ascending=[True, True, True, False, True],
    ).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=("singlem", "cbramod", "tcn_gat"), required=True); parser.add_argument("--fold", type=int, default=0); parser.add_argument("--maximum-updates", type=int, default=5000); parser.add_argument("--positive-fraction", type=float, default=0.30); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if not 0 < args.positive_fraction < 1: raise ValueError("positive fraction must be between zero and one")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    maximum = min(args.maximum_updates, 20) if args.smoke else args.maximum_updates
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    train, val, test = index_parts(args.fold); recordings = pd.read_parquet(ROOT / "manifests/recordings.parquet"); seizures = pd.read_parquet(ROOT / "manifests/seizures.parquet")
    if args.smoke:
        train = pd.concat([train[train.label == 0].head(256), train[train.label == 1].head(256)]).reset_index(drop=True)
        val = pd.concat([val[val.label == 0].head(128), val[val.label == 1].head(128)]).reset_index(drop=True)
    suffix = "" if np.isclose(args.positive_fraction, 0.30) else f"_pos{int(round(args.positive_fraction * 100)):02d}"
    fold_suffix = "" if args.fold == 0 else f"_fold{args.fold}"
    run = OUTPUT / ("smoke" if args.smoke else "formal") / f"{args.model}{suffix}{fold_suffix}"; run.mkdir(parents=True, exist_ok=True)
    encoder = v3.build_encoder(args.model).cuda(); head = SharedContextHead().cuda()
    parameters = [p for module in (encoder, head) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum, eta_min=1e-5)
    iterator = iter(loader(train, args.model, training=True, seed=3407, epoch_size=maximum * 64, positive_fraction=args.positive_fraction))
    pos, neg = float((train.label == 1).sum()), float((train.label == 0).sum())
    history = []; best_key = None; stale = 0; update = 0; started = time.monotonic()
    check_every = 10 if args.smoke else 500; minimum = 10 if args.smoke else 1500
    while update < maximum:
        encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True); loss_value = 0.0
        for _ in range(1):
            batch = next(iterator); y = batch["y"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = functional.binary_cross_entropy_with_logits(head(current_embeddings(encoder, batch)), y)
            loss.backward(); loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); scheduler.step(); update += 1
        if update % check_every == 0 or update == maximum:
            val_probability = predict(encoder, head, val, args.model)
            threshold, curve, joint_score = select(val, val_probability, recordings, seizures)
            metric = score(val, val_probability, threshold, recordings, seizures)
            metric["joint_false_alarm_score"] = joint_score
            row = {"update": update, "train_loss": loss_value, "threshold": threshold, **metric, "elapsed_seconds": time.monotonic() - started}; history.append(row)
            candidate = {"model": args.model, "encoder": encoder.state_dict(), "head": head.state_dict(), "update": update, "threshold": threshold}
            torch.save(candidate, run / f"candidate_{update}.pt")
            curve.to_csv(run / f"candidate_{update}_val_threshold_curve.csv", index=False)
            key = checkpoint_key(metric)
            if best_key is None or key < best_key:
                best_key, stale = key, 0
                torch.save({"model": args.model, "encoder": encoder.state_dict(), "head": head.state_dict(), "update": update, "threshold": threshold}, run / "best.pt")
                curve.to_csv(run / "best_val_threshold_curve.csv", index=False)
            else: stale += 1
            (run / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n"); print(json.dumps(row, sort_keys=True), flush=True)
            if update >= minimum and stale >= 7: break
    selected = select_checkpoint(history)
    selected_update = int(selected["update"])
    state = torch.load(run / f"candidate_{selected_update}.pt", map_location="cpu", weights_only=False)
    torch.save(state, run / "best.pt")
    shutil.copyfile(run / f"candidate_{selected_update}_val_threshold_curve.csv", run / "best_val_threshold_curve.csv")
    for candidate_path in run.glob("candidate_*.pt"):
        candidate_path.unlink()
    encoder.load_state_dict(state["encoder"]); head.load_state_dict(state["head"]); threshold = float(state["threshold"])
    val_probability = predict(encoder, head, val, args.model); val_metric = score(val, val_probability, threshold, recordings, seizures)
    _, _, val_joint_score = select(val, val_probability, recordings, seizures)
    val_metric["joint_false_alarm_score"] = val_joint_score
    result = {"status": "SMOKE_PASS" if args.smoke else "PASS", "model": args.model, "fold": args.fold, "best_update": int(state["update"]), "threshold": threshold, "val": val_metric, "single_window_context": True, "positive_sampling_fraction": args.positive_fraction, "pos_weight": 1.0, "threshold_order": "raw_probability_then_causal_ema_then_threshold", "threshold_objective": "sensitivity_gate_then_sum_of_percentile_ranks_for_fa_count_and_duration", "checkpoint_objective": "global_sum_of_percentile_ranks_for_fa_count_and_duration", "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20}
    if not args.smoke:
        test_probability = predict(encoder, head, test, args.model); result["test"] = score(test, test_probability, threshold, recordings, seizures)
        pd.DataFrame({"patient": test.patient, "recording": test.recording, "window_start_s": test.start, "window_end_s": test.end, "label": test.label, "probability": test_probability}).to_parquet(run / "test_probabilities.parquet", index=False)
    (run / "completed.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
