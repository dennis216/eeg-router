"""Create the immutable 10-second raw-EEG view for H6 v2.

The v1 cache already contains the same canonical 128-Hz, 16-channel,
60-second context ending at each decision time.  This script takes its final
1,280 samples, so it does not reread EDF files or alter any source data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
SOURCE = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports/h6-causal-policy-router-v2"
PARTITIONS = ("train", "dev")
SAMPLES_10S = 1280

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", default="train,dev")
    args = ap.parse_args()
    parts = tuple(x.strip() for x in args.partitions.split(",") if x.strip())
    if any(x not in PARTITIONS for x in parts):
        raise ValueError("Eval is intentionally excluded")
    manifest_out = OUT / "raw_cache_10s_manifest.json"
    if manifest_out.exists():
        raise RuntimeError(f"refusing overwrite {manifest_out}")
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"release_id": "h6-causal-policy-router-v2", "source_release_id": "h6-tusz-raw-pre-router-v1", "partitions": {}, "eval_read": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    for part in parts:
        labels_path = OUT / "labels" / f"{part}_causal_policy.parquet"
        labels = pd.read_parquet(labels_path)
        source_rel = sorted(labels.source_cache_path_v1.unique().tolist())
        written = 0
        for rel in source_rel:
            source_path = SOURCE / rel
            target_rel = Path("raw_cache_10s") / part / Path(str(labels.loc[labels.source_cache_path_v1 == rel, "recording"].iloc[0])).with_suffix(".npz")
            target = OUT / target_rel
            if target.exists():
                raise RuntimeError(f"refusing overwrite {target}")
            with np.load(source_path, allow_pickle=False) as archive:
                if "x" not in archive.files:
                    raise RuntimeError(f"missing x array: {source_path}")
                x = archive["x"]
                if x.ndim != 3 or x.shape[1] != 16 or x.shape[2] < SAMPLES_10S or not np.isfinite(x).all():
                    raise RuntimeError(f"bad source cache shape/content: {source_path} {x.shape}")
                expected = int((labels.source_cache_path_v1 == rel).sum())
                if x.shape[0] != expected:
                    raise RuntimeError(f"row mismatch for {source_path}: cache={x.shape[0]} labels={expected}")
                y = np.asarray(x[:, :, -SAMPLES_10S:], dtype=np.float16)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".tmp.npz")
            np.savez_compressed(temp, x=y)
            temp.replace(target)
            written += 1
        files = sorted((OUT / "raw_cache_10s" / part).rglob("*.npz"))
        result["partitions"][part] = {"source_cache_files": len(source_rel), "target_cache_files": len(files), "samples": int(len(labels)), "channels": 16, "sampling_hz": 128, "context_seconds": 10, "samples_per_context": SAMPLES_10S, "files_written": written, "source_label_sha256": sha256(labels_path), "cache_sha256": {str(f.relative_to(OUT)): sha256(f) for f in files}}
    result["status"] = "complete"
    manifest_out.write_text(json.dumps(result, indent=2, sort_keys=True) + chr(10))
    manifest_out.with_suffix(".sha256").write_text(sha256(manifest_out) + chr(10))
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
