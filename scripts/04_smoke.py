from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bfa.preprocessing.cache import cache_key
from bfa.preprocessing.signal import MODEL_RATES, load_canonical_uv, model_view
from bfa.provenance import sha256_file


def preprocess_smoke(root: Path, recording: str) -> None:
    matches = sorted(root.rglob(recording))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {recording}, found {len(matches)}")
    path = matches[0]
    data_uv, sampling_hz = load_canonical_uv(path)
    if data_uv.shape[0] != 16 or not np.isfinite(data_uv).all():
        raise AssertionError("canonical input must contain 16 finite channels")
    edf_hash = sha256_file(path)
    cache_root = root / "cache" / "model_input"
    cache_root.mkdir(parents=True, exist_ok=True)
    for model, target_hz in MODEL_RATES.items():
        config = {
            "branch": "model_input",
            "model": model,
            "sampling_hz": target_hz,
            "bandpass_hz": [0.5, 45.0],
            "notch_hz": 60.0,
        }
        key = cache_key(edf_hash, config)
        output = model_view(data_uv, sampling_hz, model)
        if output.shape[0] != 16 or not np.isfinite(output).all():
            raise AssertionError(f"invalid {model} cache")
        destination = cache_root / f"{recording}.{model}.{key}.npy"
        np.save(destination, output, allow_pickle=False)
        print(f"{model}: shape={output.shape} cache={destination} key={key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preprocess"], required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--root", type=Path, default=Path("/mnt/d/EEGData/chbmit-1.0.0"))
    args = parser.parse_args()
    preprocess_smoke(args.root, args.recording)


if __name__ == "__main__":
    main()
