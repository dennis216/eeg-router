"""Verify that the surviving official no-clip cache reproduces frozen H4 outputs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.training.dataset import CachedContextDataset, CachedFeatureContextDataset


ROOT = Path("/root/b_false_alarm_atlas")
RAW = Path("/mnt/d/EEGData/bfa_cache_v3_official_noclip")
FEATURES = ROOT / "cache" / "h4_reconstructed_frozen_features"
OUT = ROOT / "outputs" / "reports" / "h4-router-revision-v1" / "cache_parity"
RECORDING = "chb02_01.edf"
RELATIVE = Path("chb02/chb02_01.edf")
FOLD = 0
SEED = 3407
MODELS = ("singlem", "cbramod", "tcn_gat")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_feature(model: str) -> Path:
    target = FEATURES / model / RELATIVE.with_suffix(".npy")
    if target.exists():
        return target
    rate = 128 if model == "singlem" else 200
    feature_dim = 16 if model == "singlem" else 200
    batch_size = 256 if model == "singlem" else 32
    if model == "singlem":
        adapter = SingLEMAdapter(
            ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt"
        )
    else:
        adapter = CBraModAdapter(
            ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"
        )
    adapter = adapter.cuda().eval()
    adapter.projection.cpu()
    signal = np.load(RAW / model / RELATIVE.with_suffix(".npy"), mmap_mode="r")
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=10 * rate, axis=-1
    )[:, :: 2 * rate]
    count = windows.shape[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        target, mode="w+", dtype=np.float32, shape=(count, 16, feature_dim)
    )
    with torch.inference_mode():
        for offset in range(0, count, batch_size):
            batch = np.ascontiguousarray(
                windows[:, offset : offset + batch_size].transpose(1, 0, 2)
            ).reshape(-1, 16, 10, rate)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature = adapter.backbone_features(torch.from_numpy(batch).cuda())
            output[offset : offset + len(batch)] = feature.float().cpu().numpy()
    output.flush()
    del output, adapter
    torch.cuda.empty_cache()
    return target


def build_quality() -> Path:
    target = FEATURES / "quality" / RELATIVE.with_suffix(".npy")
    if target.exists():
        return target
    cache_module = load_module(ROOT / "scripts" / "06_cache_views.py", "h4_quality_builder")
    raw_uv, sampling_hz = cache_module.load_canonical_raw_uv(
        Path("/mnt/d/EEGData/chbmit-1.0.0") / RELATIVE
    )
    quality = cache_module._quality_features(raw_uv, sampling_hz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, quality, allow_pickle=False)
    return target


@torch.inference_mode()
def predict(runner, model: str, index: pd.DataFrame) -> np.ndarray:
    build_quality()
    if model == "tcn_gat":
        dataset = CachedContextDataset(
            index, RAW, model, quality_root=FEATURES, max_open_recordings=2
        )
    else:
        build_feature(model)
        dataset = CachedFeatureContextDataset(
            index, FEATURES, model, quality_root=FEATURES, max_open_recordings=2
        )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    checkpoint = (
        ROOT
        / "runs"
        / "v3-groupkfold-confirmatory-v1"
        / model
        / f"split{FOLD}_seed{SEED}_main"
        / "checkpoints"
        / "step_05000.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = runner._model(model).cuda()
    head = runner.SharedContextHead().cuda()
    encoder.load_state_dict(state["encoder"], strict=False)
    head.load_state_dict(state["head"], strict=True)
    encoder.eval()
    head.eval()
    values: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["x"].cuda(non_blocking=True)
        quality = batch["quality"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values.append(torch.sigmoid(head(encoder.forward_window_sequence(inputs, quality))).float().cpu().numpy())
    del encoder, head, state
    torch.cuda.empty_cache()
    return np.concatenate(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BFA_ROOT", str(ROOT))
    os.environ.setdefault(
        "BFA_CONFIG", str(ROOT / "configs" / "standard" / "v3_groupkfold_confirmatory_v1.yaml")
    )
    runner = load_module(
        Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/11_run_bsn_pilot.py"),
        "h4_cache_parity_runner",
    )
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    index = windows[(windows.recording.astype(str) == RECORDING) & ~windows.warmup].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for model in MODELS:
        actual = predict(runner, model, index)
        frozen = pd.read_parquet(
            ROOT / "runs" / "v3-groupkfold-confirmatory-v1" / model / f"split{FOLD}_seed{SEED}_main" / "test" / "probabilities.parquet"
        )
        expected = frozen[frozen.recording.astype(str) == RECORDING].sort_values("end").probability.to_numpy(float)
        if len(actual) != len(expected):
            raise RuntimeError(f"row mismatch model={model}: {len(actual)} != {len(expected)}")
        error = np.abs(actual.astype(float) - expected)
        rows.append(
            {
                "model": model,
                "rows": len(actual),
                "max_abs_probability_error": float(error.max()),
                "mean_abs_probability_error": float(error.mean()),
                "exact_fraction": float(np.mean(error == 0.0)),
                "passed_1e_6": bool(float(error.max()) <= 1e-6),
            }
        )
        print(f"CACHE_PARITY model={model} max_error={error.max():.9g}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "probability_parity.csv", index=False)
    manifest = {
        "status": "PASS" if frame.passed_1e_6.all() else "FAIL",
        "recording": RECORDING,
        "fold": FOLD,
        "seed": SEED,
        "raw_cache": str(RAW),
        "feature_cache": str(FEATURES),
        "results": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(frame.to_string(index=False))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
