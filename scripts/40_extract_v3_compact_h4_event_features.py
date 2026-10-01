"""Extract deterministic signal/QC features for H4 train/validation/test events."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EVENT_DIR = ROOT / "outputs" / "events"
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"


def _load_feature_module():
    path = ROOT / "scripts" / "35_extract_v3_compact_fp_features.py"
    spec = importlib.util.spec_from_file_location("h4_feature_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = _load_feature_module()
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    relative_by_recording = windows.drop_duplicates("recording").set_index("recording")["relative_path"].to_dict()
    sources = (
        (EVENT_DIR / "v3-compact-extension-v1_train_events.parquet", "train_fit_subset"),
        (EVENT_DIR / "v3-compact-extension-v1_validation_events.parquet", "validation"),
        (EVENT_DIR / "v3-compact-extension-v1_test_events.parquet", "test"),
    )
    frames: list[pd.DataFrame] = []
    for path, partition in sources:
        frame = pd.read_parquet(path)
        frame["partition"] = partition
        frame = frame[frame.event_type.isin(["fp", "tp"])].copy()
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    feature_frames: list[pd.DataFrame] = []
    for recording, group in events.groupby("recording", sort=True):
        feature_frames.append(base._features_for_recording(group, str(relative_by_recording[recording])))
    features = pd.concat(feature_frames, ignore_index=True)
    output_path = EVENT_DIR / "v3-compact-extension-v1_h4_event_features.parquet"
    features.to_parquet(output_path, index=False)
    manifest = {
        "release_id": "v3-compact-extension-v1",
        "path": str(output_path),
        "rows": int(len(features)),
        "partitions": {str(k): int(v) for k, v in features.partition.value_counts().to_dict().items()},
        "event_types": {str(k): int(v) for k, v in features.event_type.value_counts().to_dict().items()},
        "fit_or_selection_performed": False,
        "source": "frozen tcn_gat signal cache and frozen quality cache",
    }
    (REPORT_DIR / "h4_event_feature_manifest.json").write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(__import__("json").dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
