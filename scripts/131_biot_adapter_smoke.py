from __future__ import annotations

import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/BIOT"))

from bfa.preprocessing.signal import load_canonical_raw_uv  # noqa: E402
from model.biot import BIOTEncoder  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "outputs/reports/chb-mit-biot-router-minimal-v1/adapter_smoke.json"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    source = ROOT / "manifests/recordings.parquet"
    import pandas as pd

    row = pd.read_parquet(source).sort_values(["patient_id", "recording_id"], kind="stable").iloc[0]
    edf = Path("/mnt/d/EEGData/chbmit-1.0.0") / str(row.relative_path)
    raw_uv, source_hz = load_canonical_raw_uv(edf)
    ratio = Fraction(200.0 / float(source_hz)).limit_denominator(10000)
    signal_200 = resample_poly(raw_uv.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
    window = signal_200[:, :2000]
    scale = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
    normalized = window / (scale + 1e-8)
    if normalized.shape != (16, 2000) or not np.isfinite(normalized).all():
        raise RuntimeError(f"invalid BIOT input {normalized.shape}")
    checkpoint = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
    model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    batch = torch.from_numpy(np.stack([normalized, normalized], axis=0))
    started = time.monotonic()
    with torch.inference_mode():
        embedding = model(batch)
    payload = {
        "protocol_id": "chb-mit-biot-router-minimal-v1",
        "status": "PASS",
        "source_recording": str(row.recording_id),
        "source_relative_path": str(row.relative_path),
        "source_edf_sha256": str(row.sha256),
        "source_sampling_hz": float(source_hz),
        "input_shape": list(normalized.shape),
        "input_sampling_hz": 200,
        "input_seconds": 10,
        "channel_count": 16,
        "normalization": "per_channel_abs_95th_quantile",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "embedding_shape": list(embedding.shape),
        "embedding_finite": bool(torch.isfinite(embedding).all()),
        "embedding_std": float(embedding.std()),
        "elapsed_seconds": time.monotonic() - started,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
