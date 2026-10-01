"""Run the gated 25-update smoke for H6 causal policy router v2.

This is deliberately a smoke test, not a performance experiment.  It uses
only the corrected Train labels and the frozen 10-s raw-EEG cache.  Eval is
never read.  The batch is stratified into seizure, all-FP background,
high-disagreement background, and an ordinary-background stratum.  The
TUSZ expert thresholds leave no future-60-s all-safe rows in this frozen
label table, so the ordinary stratum is explicitly the least-disagreement
background (two or more correct experts) and this fact is recorded in the
audit rather than silently duplicating rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/root/b_false_alarm_atlas")
LABEL_NS = "h6-causal-policy-router-v2-corrected"
LABEL_DIR = ROOT / "outputs/reports" / LABEL_NS / "labels"
CACHE_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v2"
OUT = ROOT / "outputs/reports" / LABEL_NS / "training_smoke"
MODELS = ("singlem", "cbramod", "tcn_gat")
STRATA = ("seizure", "background_disagreement", "background_all_fp", "background_ordinary")
# Optional knobs for the independently named reweighted retry.  Defaults
# reproduce the original audited smoke/formal training exactly.
POLICY_SEIZURE_WEIGHT = 1.0
NORMALIZE_INPUT = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CausalPolicyNet(nn.Module):
    """The frozen v2 architecture: raw 16 x 1280 -> three independent heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.policy = nn.Linear(128, 3)
        self.reject = nn.Linear(128, 1)
        self.risk = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x).squeeze(-1)
        return self.policy(h), self.reject(h).squeeze(-1), self.risk(h).squeeze(-1)


def add_strata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create non-overlapping smoke strata without changing supervision targets."""
    out = df.copy()
    seizure = out["future_has_seizure"].astype(int).eq(1)
    all_fp = (~seizure) & out["reject_label"].astype(int).eq(1)
    # A background window with exactly one correct expert is the difficult
    # disagreement stratum.  Two or more correct experts is the least-
    # disagreement ordinary background available under frozen thresholds.
    disagreement = (~seizure) & (~all_fp) & out["correct_count"].astype(int).eq(1)
    ordinary = (~seizure) & (~all_fp) & out["correct_count"].astype(int).ge(2)
    masks = {"seizure": seizure, "background_all_fp": all_fp,
             "background_disagreement": disagreement, "background_ordinary": ordinary}
    out["smoke_stratum"] = ""
    assignment_count = np.zeros(len(out), dtype=int)
    for name, mask in masks.items():
        assignment_count += mask.to_numpy(dtype=int)
        out.loc[mask, "smoke_stratum"] = name
    assigned = out["smoke_stratum"].ne("")
    if not np.all(assignment_count <= 1):
        raise RuntimeError("stratum masks overlap")
    if int(assigned.sum()) != len(out):
        missing = out.loc[~assigned, "outcome_group"].value_counts().to_dict()
        raise RuntimeError(f"strata are not exhaustive: {missing}")
    counts = out["smoke_stratum"].value_counts().reindex(STRATA, fill_value=0).astype(int).to_dict()
    true_all_safe = int(((~seizure) & (~all_fp) & out["correct_count"].astype(int).eq(3)).sum())
    audit = {
        "counts": counts,
        "true_future_60s_all_safe_rows": true_all_safe,
        "ordinary_background_definition": "correct_count>=2 among non-seizure, non-all-FP rows",
        "ordinary_background_is_fallback": true_all_safe == 0,
        "disagreement_definition": "correct_count==1 among non-seizure, non-all-FP rows",
        "overlap_check": bool(np.any(assignment_count > 1)),
        "exhaustive_check": True,
    }
    return out, audit


def build_index(df: pd.DataFrame, seed: int, per_stratum: int) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    sizes: dict[str, int] = {}
    for stratum in STRATA:
        sub = df[df["smoke_stratum"].eq(stratum)]
        if len(sub) == 0:
            raise RuntimeError(f"missing required smoke stratum: {stratum}")
        take = min(per_stratum, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub.iloc[np.sort(idx)].copy()
        parts.append(chosen)
        sizes[stratum] = int(len(chosen))
    batch = pd.concat(parts, ignore_index=True)
    # Shuffle within each step later; retaining the stratum column here makes
    # the exact smoke batch auditable.
    return batch, sizes


@lru_cache(maxsize=256)
def load_cached_x(path_str: str, row_index: int) -> np.ndarray:
    path = Path(path_str)
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"][int(row_index)], dtype=np.float32)
    if x.shape != (16, 1280) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid raw cache sample {path}:{row_index} shape={x.shape}")
    return x


def batch_x(batch: pd.DataFrame) -> np.ndarray:
    xs: list[np.ndarray] = []
    for row in batch.itertuples(index=False):
        path = (CACHE_ROOT / str(row.raw_cache_path_v2)).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        xs.append(load_cached_x(str(path), int(row.decision_index_v1)))
    arr = np.stack(xs, axis=0)
    if NORMALIZE_INPUT:
        # Causal, per-window channel robust z-score.  It uses only the ten
        # seconds already available at the decision endpoint and preserves
        # the raw waveform input (no expert outputs or future data).
        med = np.median(arr, axis=2, keepdims=True)
        q25 = np.quantile(arr, 0.25, axis=2, keepdims=True)
        q75 = np.quantile(arr, 0.75, axis=2, keepdims=True)
        scale = np.maximum(q75 - q25, 1.0)
        arr = np.clip((arr - med) / scale, -12.0, 12.0).astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[1:] != (16, 1280) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad batch input shape={arr.shape}")
    return arr


def targets(batch: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy = torch.tensor(batch[[f"policy_target_{m}" for m in MODELS]].to_numpy(dtype=np.float32), device=device)
    eligible = torch.tensor(batch["policy_eligible"].to_numpy(dtype=np.float32), device=device)
    reject = torch.tensor(batch["reject_label"].to_numpy(dtype=np.float32), device=device)
    risk = torch.tensor(batch["risk_label"].to_numpy(dtype=np.float32), device=device)
    return policy, eligible, reject, risk


def run(args: argparse.Namespace) -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke output in {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_path = LABEL_DIR / "train_causal_policy.parquet"
    dev_path = LABEL_DIR / "dev_causal_policy.parquet"
    train_manifest = LABEL_DIR.parent / "label_manifest.json"
    cache_manifest = CACHE_ROOT / "raw_cache_10s_manifest.json"
    train, train_audit = add_strata(pd.read_parquet(train_path))
    dev, dev_audit = add_strata(pd.read_parquet(dev_path))
    train_batch, batch_sizes = build_index(train, args.seed, args.per_stratum)
    # Dev is used for a finite forward-only check, never for optimization or
    # threshold selection.
    dev_batch, dev_sizes = build_index(dev, args.seed + 1, min(args.per_stratum, 8))

    model = CausalPolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, object]] = []
    generator = np.random.default_rng(args.seed + 100)
    model.train()
    for update in range(1, args.updates + 1):
        # Fixed four-strata quota in every update, with deterministic within-
        # stratum resampling.  No Eval rows are ever touched.
        chunks: list[pd.DataFrame] = []
        for stratum in STRATA:
            sub = train[train["smoke_stratum"].eq(stratum)]
            idx = generator.choice(len(sub), size=args.per_stratum, replace=len(sub) < args.per_stratum)
            chunks.append(sub.iloc[idx])
        batch = pd.concat(chunks, ignore_index=True)
        order = generator.permutation(len(batch))
        batch = batch.iloc[order].reset_index(drop=True)
        x = torch.from_numpy(batch_x(batch)).to(device=device, dtype=torch.float32)
        policy_y, eligible, reject_y, risk_y = targets(batch, device)
        policy_row_weight = torch.where(
            torch.tensor(batch["future_has_seizure"].to_numpy(dtype=np.float32), device=device) > 0,
            torch.tensor(float(POLICY_SEIZURE_WEIGHT), device=device),
            torch.tensor(1.0, device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        policy_logits, reject_logits, risk_logits = model(x)
        logp = F.log_softmax(policy_logits, dim=1)
        per_row_policy = -(policy_y * logp).sum(dim=1)
        weighted_eligible = eligible * policy_row_weight
        policy_loss = (per_row_policy * weighted_eligible).sum() / weighted_eligible.sum().clamp_min(1.0)
        reject_loss = F.binary_cross_entropy_with_logits(reject_logits, reject_y)
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, risk_y)
        total = policy_loss + 0.2 * reject_loss + 0.5 * risk_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at update {update}")
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at update {update}")
        optimizer.step()
        history.append({
            "update": update,
            "loss": float(total.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "reject_loss": float(reject_loss.detach().cpu()),
            "risk_loss": float(risk_loss.detach().cpu()),
            "grad_norm_pre_clip": grad_norm,
            "policy_eligible_n": int(eligible.sum().detach().cpu()),
            "reject_positive_n": int(reject_y.sum().detach().cpu()),
            "risk_positive_n": int(risk_y.sum().detach().cpu()),
        })
        if getattr(args, "checkpoint_every", 0) and update % int(args.checkpoint_every) == 0:
            latest = OUT / "checkpoint_latest.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": update, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280"}, latest)
            (OUT / "progress.json").write_text(json.dumps({"status": "running", "update": update, "total_updates": args.updates, "updated_utc": utc_now(), "checkpoint": str(latest.relative_to(ROOT))}, indent=2) + "\n")

    # One deterministic Dev forward pass.  This validates tensor contracts;
    # it intentionally emits no performance metric and does not read Eval.
    model.eval()
    with torch.no_grad():
        dx = torch.from_numpy(batch_x(dev_batch)).to(device=device, dtype=torch.float32)
        dpolicy, dreject, drisk = model(dx)
        dev_policy_prob = torch.softmax(dpolicy, dim=1)
        dev_reject_prob = torch.sigmoid(dreject)
        dev_risk_prob = torch.sigmoid(drisk)
    finite_outputs = all(torch.isfinite(t).all().item() for t in (dpolicy, dreject, drisk, dev_policy_prob, dev_reject_prob, dev_risk_prob))
    argmax_share = float(torch.bincount(dev_policy_prob.argmax(dim=1), minlength=3).max().cpu() / len(dev_batch))
    checkpoint = OUT / "checkpoint.pt"
    history_path = OUT / "history.json"
    status_path = OUT / "smoke_status.json"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": args.updates, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280"}, checkpoint)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    status = {
        "release_id": LABEL_NS,
        "experiment_stage": "formal_training" if getattr(args, "formal", False) else "25_update_smoke",
        "status": "PASS" if finite_outputs and len(history) == args.updates else "FAIL",
        "created_utc": utc_now(),
        "updates": args.updates,
        "seed": args.seed,
        "device": str(device),
        "model_training": "router_only",
        "eval_read": False,
        "threshold_selection": False,
        "input_contract": {"input": "raw EEG only", "channels": 16, "samples": 1280, "sampling_hz": 128, "seconds": 10.0, "expert_outputs_in_x": False},
        "architecture": "Conv1d(16,32,k15,s4)+BN+ReLU -> Conv1d(32,64,k9,s4)+BN+ReLU -> Conv1d(64,128,k7,s4)+BN+ReLU -> GAP -> policy Linear(128,3), reject Linear(128,1), risk Linear(128,1)",
        "loss": "policy_soft_target_CE + 0.2*reject_BCE + 0.5*seizure_risk_BCE",
        "batch_quota": {"per_stratum": args.per_stratum, "strata": list(STRATA), "train_sizes": batch_sizes, "ordinary_background_fallback": bool(train_audit["ordinary_background_is_fallback"])},
        "policy_seizure_weight": float(POLICY_SEIZURE_WEIGHT),
        "input_normalization": "per-window channel median/IQR robust z-score clipped [-12,12]" if NORMALIZE_INPUT else "none",
        "train_strata_audit": train_audit,
        "dev_strata_audit": dev_audit,
        "dev_forward_only": {"rows": int(len(dev_batch)), "stratum_sizes": dev_sizes, "finite": bool(finite_outputs), "policy_argmax_max_share": argmax_share, "thresholds_selected": False},
        "source_label_manifest": str(train_manifest.relative_to(ROOT)),
        "source_label_manifest_sha256": sha256(train_manifest),
        "source_train_labels": str(train_path.relative_to(ROOT)),
        "source_train_labels_sha256": sha256(train_path),
        "source_dev_labels": str(dev_path.relative_to(ROOT)),
        "source_dev_labels_sha256": sha256(dev_path),
        "source_raw_cache_manifest": str(cache_manifest.relative_to(ROOT)),
        "source_raw_cache_manifest_sha256": sha256(cache_manifest),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "history": str(history_path.relative_to(ROOT)),
        "formal_training_allowed_by_smoke": bool(finite_outputs and len(history) == args.updates),
        "note": ("Formal Train optimization only; no Eval probabilities, test metrics, or threshold tuning." if getattr(args, "formal", False) else "Smoke only; no Eval probabilities, test metrics, threshold tuning, or performance conclusion."),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (OUT / "smoke_status.sha256").write_text(sha256(status_path) + "  smoke_status.json\n")
    (OUT / "progress.json").write_text(json.dumps({"status": "complete", "update": args.updates, "total_updates": args.updates, "updated_utc": utc_now(), "checkpoint": str(checkpoint.relative_to(ROOT))}, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "updates": args.updates, "device": str(device), "policy_argmax_max_share": argmax_share, "out": str(OUT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--per-stratum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--formal", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
