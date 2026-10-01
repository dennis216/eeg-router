"""Audit per-detector label prevalence and gradient contributions.

This is a post-hoc audit of the frozen h6-raw-phenotype-moe-v4 checkpoint.
The original training history did not persist per-head gradients, so this
script recomputes the gradients of the *same weighted BCE objectives* on the
frozen train-fit/dev tables. It never reads Siena or TUSZ eval and refuses to
overwrite an existing audit namespace.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/root/b_false_alarm_atlas")
REPORTS = ROOT / "outputs/reports"
SRC = REPORTS / "h6-raw-phenotype-moe-v4"
FEATURES_PATH = REPORTS / "h6-mask-phenotype-holdout-v4/tusz_fit_dev_phenotype_mask.parquet"
OUT_NAME = "h6-raw-phenotype-moe-v4-gradient-audit-v1"
OUT = REPORTS / OUT_NAME
MODELS = ("singlem", "cbramod", "tcn_gat")
BITS = (1, 2, 4)
ALL_LABELS = (*MODELS, "seizure_risk")
FEATURES = [
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "high_low_ratio",
]


class RawPhenotypeGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_targets(df: pd.DataFrame) -> np.ndarray:
    mask = df["mask"].astype(int).to_numpy()
    y = np.zeros((len(df), 4), dtype=np.float32)
    for j, bit in enumerate(BITS):
        y[:, j] = ((mask & bit) != 0).astype(np.float32)
    y[:, 3] = df["seizure"].astype(float).to_numpy(dtype=np.float32)
    return y


def shared_and_total_grad(model: nn.Module, loss: torch.Tensor, head: int) -> tuple[float, float, float, dict[str, torch.Tensor]]:
    """Return total, shared, and output-row norms plus detached shared gradient."""
    params = dict(model.named_parameters())
    # Keep the final linear row separate; all earlier layers are shared.
    shared_names = [n for n in params if not n.startswith("net.6.")]
    grads = torch.autograd.grad(loss, list(params.values()), retain_graph=True, allow_unused=True)
    by_name = {n: (g.detach().float() if g is not None else torch.zeros_like(params[n])) for n, g in zip(params, grads)}
    total_sq = sum(float(g.pow(2).sum()) for g in by_name.values())
    shared_sq = sum(float(by_name[n].pow(2).sum()) for n in shared_names)
    row_parts = [by_name["net.6.weight"][head], by_name["net.6.bias"][head].reshape(1)]
    row_sq = sum(float(g.pow(2).sum()) for g in row_parts)
    return math.sqrt(total_sq), math.sqrt(shared_sq), math.sqrt(row_sq), {n: by_name[n] for n in shared_names}


def cosine(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    av = torch.cat([x.reshape(-1) for x in a.values()])
    bv = torch.cat([x.reshape(-1) for x in b.values()])
    den = float(torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv))
    return float(torch.dot(av, bv) / den) if den > 0 else 0.0


def prevalence(df: pd.DataFrame, split: str, phase: str) -> dict[str, object]:
    d = df[df["partition"].eq(split) & df["phase"].eq(phase)].copy()
    n = len(d)
    y = make_targets(d)
    labels = {}
    for j, name in enumerate(ALL_LABELS):
        labels[name] = {"positive": int(y[:, j].sum()), "negative": int(n - y[:, j].sum()), "positive_pct": float(y[:, j].mean() * 100) if n else 0.0}
    masks = d["mask_bits"].value_counts().sort_index()
    return {
        "partition": split, "phase": phase, "n": n,
        "labels": labels,
        "mask_counts": {str(k): int(v) for k, v in masks.items()},
        "mask_percent": {str(k): float(v / n * 100) for k, v in masks.items()} if n else {},
    }


def gradient_audit(df: pd.DataFrame, split: str, phase: str, checkpoint: dict[str, object], device: torch.device) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    d = df[df["partition"].eq(split) & df["phase"].eq(phase)].copy()
    X = d[FEATURES].to_numpy(dtype=np.float32)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    X = (X - mean) / np.maximum(scale, 1e-12)
    y = make_targets(d)
    x = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    model = RawPhenotypeGate().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # deterministic post-hoc audit; no dropout noise
    logits = model(x)
    pw = torch.tensor(checkpoint["pos_weight"], dtype=torch.float32, device=device)
    rows: list[dict[str, object]] = []
    shared_grads: dict[str, dict[str, torch.Tensor]] = {}
    for j, name in enumerate(ALL_LABELS):
        loss = F.binary_cross_entropy_with_logits(logits[:, j], yt[:, j], pos_weight=pw[j], reduction="mean")
        total, shared, output, sg = shared_and_total_grad(model, loss, j)
        rows.append({
            "partition": split, "phase": phase, "head": name, "n": len(d),
            "weighted_bce": float(loss.detach()), "raw_positive_rate": float(y[:, j].mean()) if len(y) else 0.0,
            "total_grad_norm": total, "shared_grad_norm": shared, "output_row_grad_norm": output,
        })
        shared_grads[name] = sg
    # Pairwise direction of shared gradients; useful for detecting conflict.
    for i, a in enumerate(ALL_LABELS):
        for b in ALL_LABELS[i + 1:]:
            rows2 = {"partition": split, "phase": phase, "head_a": a, "head_b": b, "shared_gradient_cosine": cosine(shared_grads[a], shared_grads[b])}
            # Store below in a separate table.
            if "_pair_rows" not in locals():
                _pair_rows: list[dict[str, object]] = []
            _pair_rows.append(rows2)
    return rows, locals().get("_pair_rows", [])


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    scaler = json.loads((SRC / "scaler.json").read_text())
    ck = torch.load(SRC / "gate_checkpoint.pt", map_location="cpu")
    checkpoint = {"state_dict": ck["state_dict"], "scaler_mean": scaler["mean"], "scaler_scale": scaler["scale"], "pos_weight": [11.579999923706055, 6.00835657119751, 2.4092140197753906, 6.813664436340332]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev = [prevalence(df, "train", "fit"), prevalence(df, "dev", "dev")]
    all_grad: list[dict[str, object]] = []
    all_pairs: list[dict[str, object]] = []
    for split, phase in (("train", "fit"), ("dev", "dev")):
        g, p = gradient_audit(df, split, phase, checkpoint, device)
        all_grad.extend(g); all_pairs.extend(p)
    # Normalize gradient shares among detector heads only, preserving the raw values.
    gd = pd.DataFrame(all_grad)
    for (split, phase), idx in gd.groupby(["partition", "phase"]).groups.items():
        s = float(gd.loc[idx, "shared_grad_norm"].iloc[:3].sum())
        t = float(gd.loc[idx, "total_grad_norm"].iloc[:3].sum())
        gd.loc[idx[:3], "shared_gradient_share_detector_heads"] = gd.loc[idx[:3], "shared_grad_norm"] / max(s, 1e-12)
        gd.loc[idx[:3], "total_gradient_share_detector_heads"] = gd.loc[idx[:3], "total_grad_norm"] / max(t, 1e-12)
    gd.to_csv(OUT / "per_head_gradient_audit.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(OUT / "shared_gradient_cosines.csv", index=False)
    (OUT / "label_prevalence.json").write_text(json.dumps(prev, indent=2, sort_keys=True) + "\n")
    manifest = {
        "release_id": OUT_NAME, "status": "complete", "task": "post-hoc per-detector label prevalence and gradient contribution audit",
        "interpretation": "Gradients are recomputed at the frozen final checkpoint because the original training history did not store per-head traces; no Siena or TUSZ eval rows are read.",
        "source_checkpoint": str(SRC / "gate_checkpoint.pt"), "source_checkpoint_sha256": sha256(SRC / "gate_checkpoint.pt"),
        "source_features": str(FEATURES_PATH), "source_features_sha256": sha256(FEATURES_PATH),
        "partitions": ["TUSZ train/fit", "TUSZ dev/dev"], "device": str(device),
        "label_prevalence": prev,
        "gradient_csv": str(OUT / "per_head_gradient_audit.csv"), "cosine_csv": str(OUT / "shared_gradient_cosines.csv"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
