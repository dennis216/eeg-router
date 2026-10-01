"""Exact-FP-mask balanced development gate audit.

The preceding balanced gate equalized target-expert classes.  This independent
retry equalizes the six *correctness masks* that require an expert choice,
while keeping 000 (all experts FP) exclusively for reject and 111 (all experts
correct) out of expert-ranking loss.  Train/Dev only; TUSZ Eval is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/b_false_alarm_atlas")
BASE_PATH = ROOT / "scripts/241_train_balanced_moe_gate_dev.py"
spec = importlib.util.spec_from_file_location("balanced_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = ROOT / "outputs/reports/h6-moe-complementarity-gate-mask-balanced-v1"
MASKES = ("001", "010", "011", "100", "101", "110")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mask_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = df.reset_index(drop=True).copy()
    decision = (
        out["policy_eligible"].astype(int).eq(1)
        & out["policy_target_argmax"].isin(base.MODELS)
        & (out["correct_count"].astype(int) < 3)
    )
    policy = out.loc[decision].copy()
    sets: dict[str, pd.DataFrame] = {"policy": policy}
    for mask in MASKES:
        sets[f"mask_{mask}"] = policy[policy.correct_mask.astype(str).eq(mask)].copy()
        if sets[f"mask_{mask}"].empty:
            raise RuntimeError(f"empty policy mask {mask}")
    background = out["future_has_seizure"].astype(int).eq(0)
    all_fp = out["reject_label"].astype(int).eq(1)
    sets["reject_positive"] = out.loc[background & all_fp].copy()
    sets["reject_negative"] = out.loc[background & ~all_fp].copy()
    sets["risk_positive"] = out.loc[out["future_has_seizure"].astype(int).eq(1)].copy()
    sets["risk_negative"] = out.loc[background].copy()
    for name, table in sets.items():
        if table.empty:
            raise RuntimeError(f"empty set {name}")
    return sets


def make_mask_batch(samplers: dict[str, base.PatientSampler], per_mask: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = [samplers[f"mask_{m}"].draw(per_mask) for m in MASKES]
    parts += [samplers["reject_positive"].draw(per_mask), samplers["reject_negative"].draw(per_mask), samplers["risk_positive"].draw(per_mask), samplers["risk_negative"].draw(per_mask)]
    out = pd.concat(parts, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def classification_metrics(df: pd.DataFrame, label: str) -> dict[str, object]:
    return base.classification_metrics(df, label)


def run(args: argparse.Namespace) -> None:
    global OUT
    if args.output:
        OUT = Path(args.output).resolve()
    if OUT.exists():
        existing = {p.name for p in OUT.iterdir()}
        if existing != {"no_duplicate_overlap_preflight.json"}:
            raise RuntimeError(f"refusing to overwrite {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_path = base.LABEL_DIR / "train_causal_policy.parquet"
    dev_path = base.LABEL_DIR / "dev_causal_policy.parquet"
    label_manifest = base.LABEL_DIR.parent / "label_manifest.json"
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    train_sets, dev_sets = mask_sets(train), mask_sets(dev)
    rng = np.random.default_rng(args.seed)
    samplers = {name: base.PatientSampler(table, rng) for name, table in train_sets.items() if name != "policy"}
    Model = base.model_class()
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    history = []
    model.train()
    for update in range(1, args.updates + 1):
        batch = make_mask_batch(samplers, args.per_mask, rng)
        optimizer.zero_grad(set_to_none=True)
        total, item = base.losses(model, batch, device)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at {update}")
        total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        if not np.isfinite(grad):
            raise RuntimeError(f"non-finite gradient at {update}")
        optimizer.step()
        item.update({"update": update, "grad_norm_pre_clip": grad})
        history.append(item)
        if update % args.checkpoint_every == 0:
            OUT.mkdir(parents=True, exist_ok=True)
            checkpoint = OUT / "checkpoint_latest.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": update, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280_mask_balanced"}, checkpoint)
            (OUT / "progress.json").write_text(json.dumps({"status": "running", "update": update, "total_updates": args.updates, "updated_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint.relative_to(ROOT))}, indent=2) + "\n")
    checkpoint = OUT / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": args.updates, "seed": args.seed, "architecture": "causal_policy_cnn_16x1280_mask_balanced"}, checkpoint)
    pred_train = base.predict(model, train, device)
    pred_dev = base.predict(model, dev, device)
    pred_train.to_parquet(OUT / "train_predictions.parquet", index=False)
    pred_dev.to_parquet(OUT / "dev_predictions.parquet", index=False)
    classification = [classification_metrics(pred_train, "mask_balanced_gate_train"), classification_metrics(pred_dev, "mask_balanced_gate_dev"), base.constant_metrics(pred_dev, "constant_singlem_dev", 0)]
    (OUT / "classification_metrics.json").write_text(json.dumps(classification, indent=2, sort_keys=True) + "\n")
    scoring = None if args.skip_event_scoring else base.action_scoring(pred_dev, OUT / "dev_event_scoring")
    manifest = {
        "release_id": OUT.name,
        "stage": "development_only_exact_mask_balanced_gate",
        "status": "complete_train_dev_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_read": False,
        "test_read": False,
        "model_training": "router_only",
        "experts_retrained": False,
        "policy_sampling": "patient-uniform, equal quota for correctness masks 001/010/011/100/101/110",
        "mask_policy_exclusion": {"000": "reject-only; excluded from expert ranking", "111": "all-correct; excluded from expert ranking"},
        "reject_sampling": "patient-balanced all-FP vs non-all-FP background",
        "risk_sampling": "patient-balanced seizure vs background",
        "updates": args.updates,
        "per_mask": args.per_mask,
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
        "source_complementarity_audit": "outputs/reports/h6-moe-complementarity-gate-audit-v2/manifest.json",
        "source_complementarity_audit_sha256": sha256(ROOT / "outputs/reports/h6-moe-complementarity-gate-audit-v2/manifest.json"),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "classification_metrics": classification,
        "dev_event_scoring": scoring,
        "threshold_source": "TUSZ Dev only" if scoring is not None else None,
        "bootstrap_replicates": 2000 if scoring is not None else None,
        "note": "Development diagnostic only; no Eval probabilities, test metrics, threshold tuning, or detector retraining.",
    }
    (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "  manifest.json\n")
    (OUT / "progress.json").write_text(json.dumps({"status": "complete", "update": args.updates, "total_updates": args.updates, "updated_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint.relative_to(ROOT))}, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "classification": classification, "dev_event_scoring": scoring, "out": str(OUT)}, indent=2, sort_keys=True, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--per-mask", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--skip-event-scoring", action="store_true")
    parser.add_argument("--output", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
