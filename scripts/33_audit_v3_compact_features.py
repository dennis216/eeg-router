"""Audit the frozen raw/QC feature sources for H2/H3 without touching test outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = Path("/mnt/d/EEGData/bfa_cache_v3/tcn_gat")
QUALITY_ROOT = Path("/mnt/d/EEGData/bfa_cache_v3/quality")
META_ROOT = Path("/mnt/d/EEGData/bfa_cache_v3/metadata")
OUT = ROOT / "outputs" / "reports" / "v3-compact-extension-v1" / "feature_source_audit.json"


def main() -> None:
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    recordings = sorted(windows.recording.unique().tolist())
    rows: list[dict[str, object]] = []
    for recording in recordings:
        relative = str(windows.loc[windows.recording == recording, "relative_path"].iloc[0])
        rel = Path(relative).with_suffix(".npy")
        raw = RAW_ROOT / rel
        quality = QUALITY_ROOT / rel
        metadata = META_ROOT / Path(relative).with_suffix(".json")
        expected_windows = int((windows.recording == recording).sum())
        item: dict[str, object] = {
            "recording": recording,
            "relative_path": relative,
            "expected_windows": expected_windows,
            "raw_exists": raw.is_file(),
            "quality_exists": quality.is_file(),
            "metadata_exists": metadata.is_file(),
            "raw_shape": None,
            "quality_shape": None,
            "quality_columns": None,
            "window_count_match": False,
        }
        if raw.is_file():
            item["raw_shape"] = list(np.load(raw, mmap_mode="r").shape)
        if quality.is_file():
            item["quality_shape"] = list(np.load(quality, mmap_mode="r").shape)
        if metadata.is_file():
            metadata_json = json.loads(metadata.read_text())
            item["quality_columns"] = metadata_json.get("quality_columns")
        shape = item["quality_shape"]
        item["window_count_match"] = bool(shape and int(shape[0]) == expected_windows)
        rows.append(item)
    frame = pd.DataFrame(rows)
    summary = {
        "recordings": len(frame),
        "raw_available": int(frame.raw_exists.sum()),
        "quality_available": int(frame.quality_exists.sum()),
        "metadata_available": int(frame.metadata_exists.sum()),
        "window_count_match": int(frame.window_count_match.sum()),
        "raw_shape_channels": sorted({tuple(shape) for shape in frame.raw_shape.dropna()}),
        "quality_columns": sorted({column for values in frame.quality_columns.dropna() for column in values}),
        "h2_h3_feature_sources": [
            "2-second-window amplitude/RMS and spatial dispersion from frozen tcn_gat signal cache",
            "relative delta/theta/alpha/beta/gamma band power from frozen signal cache",
            "line_noise_ratio, clipping_fraction, flat_fraction from frozen quality cache",
            "event duration, peak probability, and event counts from frozen event tables",
        ],
        "all_required_sources_available": bool(
            frame.raw_exists.all() and frame.quality_exists.all() and frame.metadata_exists.all() and frame.window_count_match.all()
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "recordings": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
