"""Audit a built H3 FP-union dataset without training or touching raw EEG.

The audit is intentionally independent of the trainer.  It verifies that the
strict-overlap union, multi-label event identity, complete 10-second windows,
and manifest hashes match the frozen protocol before any GPU job is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL_COLUMNS = tuple(f"label_{name}" for name in MODELS)
WINDOW_SECONDS = 10.0
EPS = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def audit(dataset_manifest_path: Path, *, output_path: Path | None) -> dict[str, Any]:
    if not dataset_manifest_path.exists():
        fail(f"missing dataset manifest: {dataset_manifest_path}")
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != "h3-fp-union-10s-balanced-v1":
        fail("dataset manifest analysis_id is not the frozen H3 protocol")
    if manifest.get("status") != "data_built_only":
        fail("dataset manifest is not marked data_built_only")
    dataset = str(manifest.get("dataset", ""))
    if dataset not in {"chbmit", "tusz"}:
        fail(f"unexpected dataset: {dataset}")
    base = dataset_manifest_path.parent
    union_path = base / f"{dataset}_fp_union_events.parquet"
    windows_path = base / f"{dataset}_fp_union_10s_windows.parquet"
    source_path = base / f"{dataset}_canonical_fp_source_events.parquet"
    for path in (union_path, windows_path, source_path):
        if not path.exists():
            fail(f"manifest-listed file is missing: {path}")

    recorded_hashes = manifest.get("output_sha256", {})
    actual_hashes = {
        "canonical_source_events": sha256(source_path),
        "union_events": sha256(union_path),
        "windows": sha256(windows_path),
    }
    for key, actual in actual_hashes.items():
        if recorded_hashes.get(key) != actual:
            fail(f"hash mismatch for {key}: manifest={recorded_hashes.get(key)} actual={actual}")

    events = pd.read_parquet(union_path).copy()
    windows = pd.read_parquet(windows_path).copy()
    required_event = {
        "event_id", "patient_id", "relative_path", "event_start_s", "event_end_s",
        "event_duration_s", *LABEL_COLUMNS,
    }
    required_window = {
        "window_id", "window_index", "event_id", "patient_id", "relative_path",
        "window_start_s", "window_end_s", "window_index_in_event",
        "windows_per_event", *LABEL_COLUMNS,
    }
    missing_event = sorted(required_event - set(events.columns))
    missing_window = sorted(required_window - set(windows.columns))
    if missing_event:
        fail(f"union event table missing columns: {missing_event}")
    if missing_window:
        fail(f"window table missing columns: {missing_window}")
    if events.empty or windows.empty:
        fail("union event/window table is empty")
    if events["event_id"].duplicated().any() or windows["window_id"].duplicated().any():
        fail("event_id or window_id is duplicated")
    if windows["window_index"].duplicated().any():
        fail("window_index is duplicated")
    if set(windows["event_id"].astype(str)) - set(events["event_id"].astype(str)):
        fail("window table contains an event absent from union table")

    for column in LABEL_COLUMNS:
        values = pd.to_numeric(windows[column], errors="coerce")
        if values.isna().any() or not values.isin((0, 1)).all():
            fail(f"window label column is not binary: {column}")
        event_values = pd.to_numeric(events[column], errors="coerce")
        if event_values.isna().any() or not event_values.isin((0, 1)).all():
            fail(f"event label column is not binary: {column}")

    # Each event must carry one immutable target vector to every window.
    joined = windows.merge(
        events[["event_id", "patient_id", "relative_path", *LABEL_COLUMNS]],
        on="event_id",
        suffixes=("_window", "_event"),
        how="left",
        validate="many_to_one",
    )
    for column in LABEL_COLUMNS:
        if not (joined[f"{column}_window"].astype(int) == joined[f"{column}_event"].astype(int)).all():
            fail(f"event label vector changed across windows: {column}")
    if not (joined["patient_id_window"].astype(str) == joined["patient_id_event"].astype(str)).all():
        fail("event patient_id changed across windows")
    if not (joined["relative_path_window"].astype(str) == joined["relative_path_event"].astype(str)).all():
        fail("event relative_path changed across windows")

    # Check floor(duration/10), no incomplete tail, and non-overlap within an
    # event.  Endpoints touching exactly are allowed; positive overlap is not.
    event_index = events.set_index("event_id")
    bad_count = 0
    positive_overlap = 0
    for event_id, group in windows.groupby("event_id", sort=False):
        event = event_index.loc[event_id]
        duration = float(event["event_duration_s"])
        expected = int(math.floor((duration + EPS) / WINDOW_SECONDS))
        if len(group) != expected:
            bad_count += 1
            continue
        ordered = group.sort_values("window_start_s")
        starts = ordered["window_start_s"].to_numpy(dtype=float)
        ends = ordered["window_end_s"].to_numpy(dtype=float)
        if np.any((ends - starts) < WINDOW_SECONDS - EPS):
            fail(f"short/incomplete 10-second window in event {event_id}")
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1] - EPS):
            positive_overlap += 1
        if not np.allclose(starts, float(event["event_start_s"]) + np.arange(len(starts)) * WINDOW_SECONDS, atol=EPS):
            fail(f"window starts are not event-relative 10-second offsets: {event_id}")
        declared = ordered["windows_per_event"].astype(int).unique()
        if len(declared) != 1 or int(declared[0]) != expected:
            fail(f"windows_per_event declaration mismatch: {event_id}")
    if bad_count:
        fail(f"{bad_count} events have an incorrect number of complete windows")
    if positive_overlap:
        fail(f"{positive_overlap} events contain overlapping 10-second windows")

    summary = manifest.get("window_summary", {})
    expected_counts = {
        "strict_union_events": int(len(events)),
        "strict_union_events_retained": int(windows["event_id"].nunique()),
        "complete_10s_windows": int(len(windows)),
        "patients": int(windows["patient_id"].nunique()),
        "recordings": int(windows["relative_path"].nunique()),
    }
    manifest_checks = {}
    for key, actual in expected_counts.items():
        if key in summary and int(summary[key]) != actual:
            fail(f"manifest summary mismatch for {key}")
        manifest_checks[key] = actual

    result: dict[str, Any] = {
        "analysis_id": "h3-fp-union-10s-balanced-v1",
        "dataset": dataset,
        "status": "audit_passed",
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "output_sha256": actual_hashes,
        "checks": {
            "strict_overlap_union": True,
            "multi_label_target_vector_immutable": True,
            "all_complete_10s_windows": True,
            "event_relative_nonoverlap": True,
            "no_positive_window_overlap": True,
            "manifest_counts": manifest_checks,
        },
        "counts": {
            "union_events": int(len(events)),
            "windows": int(len(windows)),
            "patients": int(windows["patient_id"].nunique()),
            "recordings": int(windows["relative_path"].nunique()),
            "multi_label_events": int((events[list(LABEL_COLUMNS)].sum(axis=1) > 1).sum()),
            "multi_label_windows": int((windows[list(LABEL_COLUMNS)].sum(axis=1) > 1).sum()),
        },
        "training_executed": False,
        "test_evaluation_executed": False,
    }
    if output_path is not None:
        if output_path.exists():
            fail(f"refusing to overwrite audit output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_manifest, output_path=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
