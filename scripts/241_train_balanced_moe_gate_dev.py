"""Development-only balanced gate audit for H6 complementarity.

This is an independent Train/Dev diagnostic.  It does not retrain the three
seizure experts and never reads TUSZ Eval.  The policy head is trained only on
decision-needed rows (rows where the frozen experts do not all have the same
outcome); clean all-safe background is excluded from expert-ranking loss.
Rare target classes are sampled patient-uniformly and the reject/risk heads
use separate balanced samplers.  The resulting checkpoint and Dev event
scores are diagnostic and are not a confirmatory test result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path("/root/b_false_alarm_atlas")
LABEL_NS = "h6-causal-policy-router-v2-corrected"
LABEL_DIR = ROOT / "outputs/reports" / LABEL_NS / "labels"
CACHE_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v2"
OUT = ROOT / "outputs/reports/h6-moe-complementarity-gate-balanced-v2"
MODELS = ("singlem", "cbramod", "tcn_gat")
TARGET_TO_ID = {name: i for i, name in enumerate(MODELS)}
GRID = (0.20, 0.35, 0.50, 0.65, 0.80, 0.90)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build non-overlapping policy, reject and risk source tables."""
    out = df.reset_index(drop=True).copy()
    is_seizure = out["future_has_seizure"].astype(int).eq(1)
    is_background = ~is_seizure
    is_all_fp = out["reject_label"].astype(int).eq(1)
    decision = (
        out["policy_eligible"].astype(int).eq(1)
        & out["policy_target_argmax"].isin(MODELS)
        & out["correct_count"].astype(int).lt(3)
    )
    policy = out.loc[decision].copy()
    if policy.empty:
        raise RuntimeError("no policy decision rows")
    policy["target_id"] = policy["policy_target_argmax"].map(TARGET_TO_ID).astype(int)
    sets = {
        "policy": policy,
        "policy_singlem": policy[policy.target_id.eq(0)].copy(),
        "policy_cbramod": policy[policy.target_id.eq(1)].copy(),
        "policy_tcn_gat": policy[policy.target_id.eq(2)].copy(),
        "reject_positive": out.loc[is_background & is_all_fp].copy(),
        "reject_negative": out.loc[is_background & ~is_all_fp].copy(),
        "risk_positive": out.loc[is_seizure].copy(),
        "risk_negative": out.loc[is_background].copy(),
    }
    for name, table in sets.items():
        if table.empty:
            raise RuntimeError(f"empty training stratum: {name}")
    return sets


class PatientSampler:
    """Uniformly choose a patient, then one row from that patient."""

    def __init__(self, df: pd.DataFrame, rng: np.random.Generator):
        self.df = df.reset_index(drop=True)
        self.rng = rng
        self.by_patient = [group.index.to_numpy(dtype=int) for _, group in self.df.groupby("patient_id", sort=True)]
        if not self.by_patient:
            raise RuntimeError("sampler has no patients")

    def draw(self, n: int) -> pd.DataFrame:
        rows = []
        for _ in range(int(n)):
            group = self.by_patient[int(self.rng.integers(0, len(self.by_patient)))]
            rows.append(int(group[int(self.rng.integers(0, len(group)))]))
        return self.df.iloc[rows].copy()


@lru_cache(maxsize=4096)
def load_array(path_str: str) -> np.ndarray:
    path = Path(path_str)
    with np.load(path, allow_pickle=False) as z:
        arr = np.asarray(z["x"], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (16, 1280) or not np.isfinite(arr).all():
        raise RuntimeError(f"invalid cache array {path}: {arr.shape}")
    return arr


def batch_x(batch: pd.DataFrame) -> np.ndarray:
    # Load each compressed cache file once per batch and scatter rows back to
    # the original order.  The previous row-wise implementation repeatedly
    # decompressed the same .npz files and left the GPU mostly idle.
    out = np.empty((len(batch), 16, 1280), dtype=np.float32)
    for cache_path, group in batch.groupby("raw_cache_path_v2", sort=False):
        arr = load_array(str((CACHE_ROOT / str(cache_path)).resolve()))
        positions = group.index.to_numpy(dtype=int)
        rows = group["decision_index_v1"].to_numpy(dtype=int)
        out[positions] = arr[rows]
    if out.shape[1:] != (16, 1280) or not np.isfinite(out).all():
        raise RuntimeError(f"invalid batch shape {out.shape}")
    return out


def model_class():
    mod = load_module(ROOT / "scripts/170_smoke_h6_causal_policy_router_v2.py", "balanced_gate_model")
    return mod.CausalPolicyNet


def make_batch(sets: dict[str, pd.DataFrame], samplers: dict[str, PatientSampler], per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = [
        samplers["policy_singlem"].draw(per_class),
        samplers["policy_cbramod"].draw(per_class),
        samplers["policy_tcn_gat"].draw(per_class),
        samplers["reject_positive"].draw(per_class),
        samplers["reject_negative"].draw(per_class),
        samplers["risk_positive"].draw(per_class),
        samplers["risk_negative"].draw(per_class),
    ]
    out = pd.concat(parts, ignore_index=True)
    order = rng.permutation(len(out))
    return out.iloc[order].reset_index(drop=True)


def losses(model, batch: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(batch_x(batch)).to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        policy_logits, reject_logits, risk_logits = model(x)
        policy_rows = batch["policy_target_argmax"].isin(MODELS).to_numpy(dtype=bool) & (batch["correct_count"].to_numpy(dtype=int) < 3)
        reject_rows = batch["future_has_seizure"].to_numpy(dtype=int) == 0
        risk_rows = np.ones(len(batch), dtype=bool)
        policy_idx = torch.from_numpy(policy_rows).to(device)
        reject_idx = torch.from_numpy(reject_rows).to(device)
        policy_y = torch.tensor(batch.loc[policy_rows, [f"policy_target_{m}" for m in MODELS]].to_numpy(dtype=np.float32), device=device)
        reject_y = torch.tensor(batch["reject_label"].to_numpy(dtype=np.float32), device=device)
        risk_y = torch.tensor(batch["risk_label"].to_numpy(dtype=np.float32), device=device)
        policy_logp = F.log_softmax(policy_logits[policy_idx], dim=1)
        policy_loss = -(policy_y * policy_logp).sum(dim=1).mean()
        reject_loss = F.binary_cross_entropy_with_logits(reject_logits[reject_idx], reject_y[reject_idx])
        risk_loss = F.binary_cross_entropy_with_logits(risk_logits, risk_y)
        total = policy_loss + 0.2 * reject_loss + 0.5 * risk_loss
    return total, {"loss": float(total.detach().float().cpu()), "policy_loss": float(policy_loss.detach().float().cpu()), "reject_loss": float(reject_loss.detach().float().cpu()), "risk_loss": float(risk_loss.detach().float().cpu())}


def predict(model, df: pd.DataFrame, device: torch.device, batch_size: int = 256) -> pd.DataFrame:
    model.eval()
    out = np.zeros((len(df), 5), dtype=np.float32)
    with torch.inference_mode():
        for cache_path, group in df.groupby("raw_cache_path_v2", sort=False):
            arr = load_array(str((CACHE_ROOT / str(cache_path)).resolve()))
            group = group.sort_values("decision_index_v1")
            indices = group.index.to_numpy(dtype=int)
            rows = group.decision_index_v1.to_numpy(dtype=int)
            for start in range(0, len(rows), batch_size):
                sel = rows[start : start + batch_size]
                x = torch.from_numpy(arr[sel]).to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    policy, reject, risk = model(x)
                values = np.concatenate([
                    torch.softmax(policy.float(), dim=1).cpu().numpy(),
                    torch.sigmoid(reject.float()).cpu().numpy()[:, None],
                    torch.sigmoid(risk.float()).cpu().numpy()[:, None],
                ], axis=1)
                out[indices[start : start + len(sel)]] = values
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite predictions")
    result = df.copy()
    for i, model_name in enumerate(MODELS):
        result[f"router_competence_{model_name}"] = out[:, i]
        result[f"target_weight_{model_name}"] = result[f"policy_target_{model_name}"]
    result["router_reject_probability"] = out[:, 3]
    result["router_seizure_risk"] = out[:, 4]
    return result


def classification_metrics(df: pd.DataFrame, label: str) -> dict[str, object]:
    decision = df["policy_target_argmax"].isin(MODELS) & (df["correct_count"].astype(int) < 3)
    sub = df.loc[decision].copy()
    y = sub["policy_target_argmax"].map(TARGET_TO_ID).to_numpy(dtype=int)
    p = sub[[f"router_competence_{m}" for m in MODELS]].to_numpy(dtype=float).argmax(axis=1)
    recalls = {}
    for i, name in enumerate(MODELS):
        mask = y == i
        recalls[name] = float((p[mask] == i).mean()) if mask.any() else float("nan")
    target_share = {name: float((y == i).mean()) for i, name in enumerate(MODELS)}
    selected_share = {name: float((p == i).mean()) for i, name in enumerate(MODELS)}
    return {
        "model": label,
        "rows": int(len(sub)),
        "accuracy": float((p == y).mean()),
        "balanced_accuracy": float(np.nanmean(list(recalls.values()))),
        "target_share": target_share,
        "selected_share": selected_share,
        "per_class_recall": recalls,
    }


def constant_metrics(df: pd.DataFrame, label: str, class_id: int) -> dict[str, object]:
    decision = df["policy_target_argmax"].isin(MODELS) & (df["correct_count"].astype(int) < 3)
    sub = df.loc[decision]
    y = sub["policy_target_argmax"].map(TARGET_TO_ID).to_numpy(dtype=int)
    p = np.full(len(y), class_id, dtype=int)
    recalls = {name: float((p[y == i] == i).mean()) if np.any(y == i) else float("nan") for i, name in enumerate(MODELS)}
    return {"model": label, "rows": int(len(y)), "accuracy": float((p == y).mean()), "balanced_accuracy": float(np.nanmean(list(recalls.values()))), "target_share": {name: float((y == i).mean()) for i, name in enumerate(MODELS)}, "selected_share": {MODELS[class_id]: 1.0}, "per_class_recall": recalls}


def action_scoring(labels: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    scorer = load_module(ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py", "balanced_gate_scorer")
    expert_root = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1/dev"
    manifest_path = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/dev_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    events, expert_thresholds, ensemble_cache = scorer._read_event_cache(manifest, expert_root)
    fixed_rows = {m: scorer._score_actions(labels, manifest, events, f"fixed_{m}", None, ensemble_cache=ensemble_cache) for m in MODELS}
    fixed_summaries = {m: scorer._summary(v) for m, v in fixed_rows.items()}
    fixed_frame = pd.DataFrame([{"method": f"fixed_{m}", **s} for m, s in fixed_summaries.items()])
    max_sens = float(fixed_frame.sensitivity.max())
    safe_fixed = fixed_frame[fixed_frame.sensitivity >= max_sens - 0.03]
    best_fixed = str(safe_fixed.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0].method).removeprefix("fixed_")
    baseline_sens = float(fixed_summaries[best_fixed]["sensitivity"])
    no_reject_rows = scorer._score_actions(labels, manifest, events, "router_no_reject", None, ensemble_cache=ensemble_cache)
    no_reject_summary = scorer._summary(no_reject_rows)
    candidates = []
    for rp in GRID:
        for ec in GRID:
            for sr in GRID:
                thresholds = {"reject_probability": rp, "expert_competence": ec, "seizure_risk": sr}
                rows = scorer._score_actions(labels, manifest, events, "router", thresholds, ensemble_cache=ensemble_cache)
                summary = scorer._summary(rows)
                candidates.append({**thresholds, **summary, "safe_sensitivity": bool(float(summary["sensitivity"]) >= baseline_sens - 0.03)})
    candidate_frame = pd.DataFrame(candidates)
    safe = candidate_frame[candidate_frame.safe_sensitivity]
    selected = (safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0] if not safe.empty else candidate_frame.sort_values(["sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0]).to_dict()
    selected.update({"best_fixed_model": best_fixed, "best_fixed_sensitivity": baseline_sens, "threshold_source": "TUSZ Dev only", "eval_used_for_selection": False, "expert_thresholds": expert_thresholds, "status": "safe_dev_point_selected" if not safe.empty else "no_safe_dev_point"})
    threshold = {k: float(selected[k]) for k in ("reject_probability", "expert_competence", "seizure_risk")}
    routed_rows = scorer._score_actions(labels, manifest, events, "router", threshold, ensemble_cache=ensemble_cache)
    methods_rows = {"best_fixed": fixed_rows[best_fixed], "router_no_reject": no_reject_rows, "balanced_router": routed_rows}
    summaries = [{"method": method, **scorer._summary(rows)} for method, rows in methods_rows.items()]
    bootstrap = pd.concat([scorer._bootstrap(rows, method, 2000, 20260901) for method, rows in methods_rows.items()], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=False)
    fixed_frame.to_csv(out_dir / "fixed_baselines.csv", index=False)
    candidate_frame.to_csv(out_dir / "threshold_candidates.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    bootstrap.to_csv(out_dir / "patient_bootstrap.csv", index=False)
    (out_dir / "threshold_selection.json").write_text(json.dumps(selected, indent=2, sort_keys=True, default=float) + "\n")
    return {"fixed_summaries": fixed_summaries, "best_fixed": best_fixed, "best_fixed_sensitivity": baseline_sens, "router_no_reject": no_reject_summary, "selected": selected, "summaries": summaries, "bootstrap_replicates": 2000}


def run(args: argparse.Namespace) -> None:
    global OUT
    if getattr(args, "output", None):
        OUT = Path(args.output).resolve()
    if OUT.exists():
        existing = {p.name for p in OUT.iterdir()}
        # The duplicate-overlap preflight creates the namespace directory
        # before execution.  It is safe to retain that single audit file;
        # every other pre-existing artifact is an overwrite refusal.
        if existing != {"no_duplicate_overlap_preflight.json"}:
            raise RuntimeError(f"refusing to overwrite {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the balanced gate audit")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_path = LABEL_DIR / "train_causal_policy.parquet"
    dev_path = LABEL_DIR / "dev_causal_policy.parquet"
    label_manifest = LABEL_DIR.parent / "label_manifest.json"
    audit_dir = ROOT / "outputs/reports/h6-moe-complementarity-gate-audit-v2"
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    train_sets = build_sets(train)
    dev_sets = build_sets(dev)
    rng = np.random.default_rng(args.seed)
    samplers = {name: PatientSampler(table, rng) for name, table in train_sets.items() if name != "policy"}
    Model = model_class()
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    history = []
    model.train()
    for update in range(1, args.updates + 1):
        batch = make_batch(train_sets, samplers, args.per_class, rng)
        optimizer.zero_grad(set_to_none=True)
        total, item = losses(model, batch, device)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at update {update}")
        total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        if not math.isfinite(grad):
            raise RuntimeError(f"non-finite gradient at update {update}")
        optimizer.step()
        item.update({"update": update, "grad_norm_pre_clip": grad})
        history.append(item)
        if update % args.checkpoint_every == 0:
            checkpoint = OUT / "checkpoint_latest.pt"
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": update, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280_balanced_policy"}, checkpoint)
            (OUT / "progress.json").write_text(json.dumps({"status": "running", "update": update, "total_updates": args.updates, "updated_utc": utc_now(), "checkpoint": str(checkpoint.relative_to(ROOT))}, indent=2) + "\n")
    checkpoint = OUT / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": args.updates, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280_balanced_policy"}, checkpoint)
    predictions_train = predict(model, train, device)
    predictions_dev = predict(model, dev, device)
    (OUT / "train_predictions.parquet").parent.mkdir(parents=True, exist_ok=True)
    predictions_train.to_parquet(OUT / "train_predictions.parquet", index=False)
    predictions_dev.to_parquet(OUT / "dev_predictions.parquet", index=False)
    classification = [classification_metrics(predictions_train, "balanced_gate_train"), classification_metrics(predictions_dev, "balanced_gate_dev"), constant_metrics(predictions_dev, "constant_singlem_dev", 0)]
    (OUT / "classification_metrics.json").write_text(json.dumps(classification, indent=2, sort_keys=True) + "\n")
    scoring = None if getattr(args, "skip_event_scoring", False) else action_scoring(predictions_dev, OUT / "dev_event_scoring")
    manifest = {
        "release_id": OUT.name,
        "stage": "development_only_balanced_gate",
        "status": "complete_train_dev_only",
        "created_utc": utc_now(),
        "eval_read": False,
        "test_read": False,
        "model_training": "router_only",
        "experts_retrained": False,
        "policy_sampling": "patient-uniform per target argmax class; clean all-safe and all-FP rows excluded from expert ranking policy loss",
        "reject_sampling": "patient-balanced all-FP vs non-all-FP background",
        "risk_sampling": "patient-balanced seizure vs background",
        "policy_classes": list(MODELS),
        "architecture": "same H6 causal policy CNN: raw 16x1280 -> Conv1d blocks -> GAP -> policy/reject/risk heads",
        "loss": "balanced soft-target policy CE + 0.2 reject BCE + 0.5 risk BCE",
        "updates": args.updates,
        "seed": args.seed,
        "device": str(device),
        "train_counts": {name: int(len(table)) for name, table in train_sets.items()},
        "dev_counts": {name: int(len(table)) for name, table in dev_sets.items()},
        "source_train_labels": str(train_path.relative_to(ROOT)),
        "source_train_labels_sha256": sha256(train_path),
        "source_dev_labels": str(dev_path.relative_to(ROOT)),
        "source_dev_labels_sha256": sha256(dev_path),
        "source_label_manifest": str(label_manifest.relative_to(ROOT)),
        "source_label_manifest_sha256": sha256(label_manifest),
        "source_complementarity_audit": str((audit_dir / "manifest.json").relative_to(ROOT)),
        "source_complementarity_audit_sha256": sha256(audit_dir / "manifest.json"),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "classification_metrics": classification,
        "dev_event_scoring": scoring,
        "threshold_source": "TUSZ Dev only",
        "bootstrap_replicates": 2000,
        "note": "Diagnostic development version only. No Eval probabilities, test metrics, threshold tuning, or detector retraining were performed.",
    }
    (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "  manifest.json\n")
    (OUT / "progress.json").write_text(json.dumps({"status": "complete", "update": args.updates, "total_updates": args.updates, "updated_utc": utc_now(), "checkpoint": str(checkpoint.relative_to(ROOT))}, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "classification": classification, "dev_event_scoring": scoring, "out": str(OUT)}, indent=2, sort_keys=True, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--output", default=None, help="optional independent output directory for a smoke/development retry")
    parser.add_argument("--skip-event-scoring", action="store_true", help="skip slow Dev event scoring for a code/throughput smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
