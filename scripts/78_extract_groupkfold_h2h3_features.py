"""Extract frozen signal/QC features for standard H2/H3 event tables."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
EVENT_INPUT_NAME = os.environ.get("BFA_H2H3_EVENT_INPUT", "h2h3_events.parquet")
FEATURE_OUTPUT_NAME = os.environ.get("BFA_H2H3_FEATURE_OUTPUT", "h2h3_fp_features.parquet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_feature_helper():
    path = ROOT / "scripts" / "35_extract_v3_compact_fp_features.py"
    spec = importlib.util.spec_from_file_location("feature_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    event_path = ROOT / "outputs" / "reports" / RUN_ID / EVENT_INPUT_NAME
    events = pd.read_parquet(event_path)
    events = events[events.event_type == "fp"].copy()
    events["split"] = events["fold"].astype(int)
    helper = _load_feature_helper()
    rows: list[pd.DataFrame] = []
    for recording, group in events.groupby("recording", sort=True):
        relative_path = str(group.relative_path.iloc[0])
        rows.append(helper._features_for_recording(group, relative_path))
        if len(rows) % 25 == 0:
            print(f"H2H3_FEATURES_PROGRESS recordings={len(rows)}", flush=True)
    features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_path = ROOT / "outputs" / "reports" / RUN_ID / FEATURE_OUTPUT_NAME
    features.to_parquet(output_path, index=False)
    manifest = {
        "release_id": RUN_ID,
        "rows": int(len(features)),
        "recordings": int(features.recording.nunique()) if not features.empty else 0,
        "partitions": {str(key): int(value) for key, value in features.partition.value_counts().to_dict().items()} if not features.empty else {},
        "models": sorted(features.model.unique().tolist()) if not features.empty else [],
        "folds": sorted(features.split.unique().tolist()) if not features.empty else [],
        "seeds": sorted(features.seed.unique().tolist()) if not features.empty else [],
        "source_events": str(event_path.relative_to(ROOT)),
        "source_events_sha256": _sha256(event_path),
        "fit_or_selection_performed": False,
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
    }
    manifest_name = output_path.with_name(output_path.stem + "_manifest.json")
    (manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
