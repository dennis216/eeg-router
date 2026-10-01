"""Extract deterministic TP/FP signal and QC features for standard H4.

The source event table is already frozen.  This script performs no fitting,
threshold selection, clustering, or classifier training; it only maps each
event to the cached signal/QC arrays.  All three partitions are retained so
that the downstream router can fit on outer-train discovery, choose a
workpoint on validation, and evaluate test once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
EVENT_PATH = REPORT_DIR / "h2h3_events.parquet"
OUTPUT_PATH = REPORT_DIR / "h4_groupkfold_event_features.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_helper():
    path = ROOT / "scripts" / "35_extract_v3_compact_fp_features.py"
    spec = importlib.util.spec_from_file_location("groupkfold_h4_feature_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import feature helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    events = pd.read_parquet(EVENT_PATH)
    events = events[events.event_type.isin(["tp", "fp"])].copy()
    # The frozen event table names the outer fold ``fold``; the shared feature
    # helper uses the historical ``split`` field.  This is only a rename and
    # does not alter assignments.
    events["split"] = events["fold"].astype(int)
    windows = pd.read_parquet(ROOT / "manifests" / "windows.parquet")
    relative_by_recording = (
        windows.drop_duplicates("recording").set_index("recording")["relative_path"].astype(str).to_dict()
    )
    helper = _load_helper()
    frames: list[pd.DataFrame] = []
    recordings = sorted(events.recording.astype(str).unique())
    for index, recording in enumerate(recordings, start=1):
        group = events[events.recording.astype(str) == recording].copy()
        frames.append(helper._features_for_recording(group, relative_by_recording[recording]))
        if index % 25 == 0:
            print(f"H4_FEATURES_PROGRESS recordings={index}/{len(recordings)}", flush=True)
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    features.to_parquet(OUTPUT_PATH, index=False)
    manifest = {
        "release_id": RUN_ID,
        "rows": int(len(features)),
        "recordings": int(features.recording.nunique()) if not features.empty else 0,
        "partitions": {
            str(key): int(value) for key, value in features.partition.value_counts().to_dict().items()
        } if not features.empty else {},
        "event_types": {
            str(key): int(value) for key, value in features.event_type.value_counts().to_dict().items()
        } if not features.empty else {},
        "folds": sorted(int(value) for value in features.split.unique()) if not features.empty else [],
        "seeds": sorted(int(value) for value in features.seed.unique()) if not features.empty else [],
        "feature_names": [
            "event_duration_s",
            "peak_probability",
            "rms_median_uv",
            "rms_iqr_uv",
            "spatial_rms_dispersion",
            "spatial_rms_max_min_log",
            "line_noise_ratio",
            "clipping_fraction",
            "flat_fraction",
            "relative_delta_power",
            "relative_theta_power",
            "relative_alpha_power",
            "relative_beta_power",
            "relative_gamma_power",
            "high_low_ratio",
        ],
        "fit_or_selection_performed": False,
        "source_event_table": str(EVENT_PATH.relative_to(ROOT)),
        "source_event_table_sha256": _sha256(EVENT_PATH),
    }
    manifest_path = REPORT_DIR / "h4_groupkfold_event_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
