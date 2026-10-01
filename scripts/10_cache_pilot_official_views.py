from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from bfa.preprocessing.signal import load_canonical_raw_uv


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _cache_recording(
    row: dict[str, object],
    data_root: str,
    output_root: str,
    profile: str,
    config_sha256: str | None,
) -> dict[str, object]:
    started = time.monotonic()
    relative = Path(str(row["relative_path"]))
    root = Path(output_root)
    singlem_target = root / "singlem" / relative.with_suffix(".npy")
    cbramod_target = root / "cbramod" / relative.with_suffix(".npy")
    metadata_target = root / "metadata" / relative.with_suffix(".json")
    if singlem_target.exists() and cbramod_target.exists() and metadata_target.exists():
        metadata = json.loads(metadata_target.read_text())
        if (
            metadata.get("profile") == profile
            and metadata.get("preprocess_config_sha256") == config_sha256
        ):
            return {"recording_id": row["recording_id"], "status": "cached", "seconds": 0.0}
        raise RuntimeError(f"cache provenance mismatch for {relative}; use a separate cache root")
    raw_uv, sampling_hz = load_canonical_raw_uv(Path(data_root) / relative)
    notched = mne.filter.notch_filter(
        raw_uv.astype(np.float64), sampling_hz, freqs=[60.0], verbose="ERROR"
    )
    singlem_filtered = mne.filter.filter_data(
        notched.copy(), sampling_hz, 0.5, 50.0, verbose="ERROR"
    )
    cbramod_filtered = mne.filter.filter_data(notched, sampling_hz, 0.3, 75.0, verbose="ERROR")
    singlem = (resample_poly(singlem_filtered, 1, 2, axis=-1) * 0.01).astype(np.float32)
    cbramod = (resample_poly(cbramod_filtered, 25, 32, axis=-1) * 0.01).astype(np.float32)
    _atomic_npy(singlem_target, singlem)
    _atomic_npy(cbramod_target, cbramod)
    metadata = {
        "profile": profile,
        "preprocess_config_sha256": config_sha256,
        "recording_id": row["recording_id"],
        "relative_path": str(relative),
        "edf_sha256": row["sha256"],
        "source_unit": "uV",
        "powerline_notch_hz": 60.0,
        "singlem": {
            "bandpass_hz": [0.5, 50.0],
            "sampling_hz": 128.0,
            "scale": 0.01,
            "hard_clipping": False,
        },
        "cbramod": {
            "bandpass_hz": [0.3, 75.0],
            "sampling_hz": 200.0,
            "scale": 0.01,
            "hard_clipping": False,
        },
    }
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "recording_id": row["recording_id"],
        "status": "created",
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/recordings.parquet"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/EEGData/chbmit-1.0.0"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/mnt/d/EEGData/bfa_cache_bsn_pilot_v1")
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--profile", default="BSN-pilot-v1")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    rows = pd.read_parquet(args.manifest).sort_values(["patient_id", "recording_id"], kind="stable")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = None
    if args.config is not None:
        config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _cache_recording,
                row,
                str(args.data_root),
                str(args.output_root),
                args.profile,
                config_sha256,
            )
            for row in rows.to_dict("records")
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(
                json.dumps({"completed": completed, "total": len(futures), **result}),
                flush=True,
            )


if __name__ == "__main__":
    main()
