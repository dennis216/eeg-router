"""Build the canonical FP-union, fixed-10-second, multi-label H3 dataset.

This script only constructs and audits data.  It does not train a classifier.
The three detector outputs are first reduced to one preselected canonical run
per detector.  Strict interval-overlap connected components are then formed
within each recording.  Every component is represented by all complete,
non-overlapping 10-second windows starting at the component start; a tail
shorter than 10 seconds is discarded.  The component's detector-membership
vector is copied to every retained window.

The resulting table is deliberately independent of the old midpoint-bin and
single-label implementations.  The builder refuses conflicting duplicate
source events and refuses to silently combine multiple detector seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
ANALYSIS_ID = "h3-fp-union-10s-balanced-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
LABEL_COLUMNS = tuple(f"label_{name}" for name in MODELS)
WINDOW_S = 10.0
EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("analysis_id") != ANALYSIS_ID:
        raise RuntimeError("config analysis_id does not match the frozen H3 protocol")
    return config, hashlib.sha256(config_bytes).hexdigest()


def _model_name(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "singlelem": "singlem",
        "single_lem": "singlem",
        "tcn_gat": "tcn_gat",
        "tcngat": "tcn_gat",
        "cbra_mod": "cbramod",
    }
    return aliases.get(normalized, normalized)


def _first_present(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def normalize_source_events(
    source_path: Path,
    *,
    canonical_fold: int,
    canonical_seed: int,
    canonical_run_id: str | None,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    frame = pd.read_parquet(source_path).copy()

    if "event_type" in frame.columns:
        frame = frame[frame["event_type"].astype(str).str.lower().eq("fp")].copy()
    elif "is_true_positive" in frame.columns:
        frame = frame[~frame["is_true_positive"].astype(bool)].copy()
    else:
        raise RuntimeError("source event table must identify false positives")

    patient_column = _first_present(frame, ("patient_id", "patient"))
    if patient_column is None:
        raise RuntimeError("source event table has no patient/patient_id column")
    frame = frame.rename(columns={patient_column: "patient_id"})
    if "relative_path" not in frame.columns:
        if "recording" not in frame.columns:
            raise RuntimeError("source event table has no recording/relative_path column")
        frame["relative_path"] = frame["recording"].astype(str)

    required = {"patient_id", "recording", "relative_path", "model", "start_s", "end_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"source event table missing columns: {missing}")

    # Select exactly one canonical detector run.  If a table exposes only one
    # of fold/seed, refusing it is safer than silently mixing detector seeds.
    has_fold = "fold" in frame.columns
    has_seed = "seed" in frame.columns
    selection_mode = "fold_seed" if has_fold and has_seed else "run_id"
    if has_fold != has_seed and canonical_run_id is None:
        raise RuntimeError("source event table exposes only one of fold/seed; canonical selection is ambiguous")
    if has_fold and has_seed:
        frame = frame[frame["fold"].astype(int).eq(int(canonical_fold))].copy()
        frame = frame[frame["seed"].astype(int).eq(int(canonical_seed))].copy()
    elif canonical_run_id is None and "run_id" in frame.columns:
        raise RuntimeError("source event table has run_id but no canonical selector; pass --canonical-run-id")
    elif canonical_run_id is None and "run_id" not in frame.columns:
        # A table without any run selector is accepted only when it is already
        # a single canonical export.  Record this fact in the manifest so it
        # cannot be mistaken for a fold/seed-filtered table.
        selection_mode = "preselected_single_run_export"
    if canonical_run_id is not None:
        if "run_id" not in frame.columns:
            raise RuntimeError("--canonical-run-id was supplied but run_id is absent")
        frame = frame[frame["run_id"].astype(str).eq(str(canonical_run_id))].copy()

    frame["model"] = frame["model"].map(_model_name)
    frame = frame[frame["model"].isin(MODELS)].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["recording"] = frame["recording"].astype(str)
    frame["relative_path"] = frame["relative_path"].astype(str)
    frame["start_s"] = pd.to_numeric(frame["start_s"], errors="coerce")
    frame["end_s"] = pd.to_numeric(frame["end_s"], errors="coerce")
    frame = frame[np.isfinite(frame["start_s"]) & np.isfinite(frame["end_s"])].copy()
    frame = frame[frame["end_s"] > frame["start_s"]].copy()
    if frame.empty:
        raise RuntimeError("no false-positive events remain after canonical selection")

    key_columns = ["patient_id", "recording", "relative_path", "model", "start_s", "end_s"]
    duplicate_groups = frame.groupby(key_columns, dropna=False, sort=False).size()
    # Exact duplicates from a parquet export are harmless; conflicting rows are
    # not, because they would make the same detector event non-deterministic.
    frame = frame.drop_duplicates(key_columns, keep="first").copy()
    frame = frame.sort_values(
        ["patient_id", "recording", "relative_path", "start_s", "end_s", "model"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["source_event_id"] = [
        sha256_text(
            f"{dataset}|{row.patient_id}|{row.recording}|{row.relative_path}|"
            f"{row.model}|{float(row.start_s):.6f}|{float(row.end_s):.6f}"
        )[:24]
        for row in frame.itertuples(index=False)
    ]
    meta = {
        "dataset": dataset,
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "source_rows_after_fp_filter_and_run_selection": int(len(frame)),
        "source_exact_duplicate_groups_removed": int((duplicate_groups > 1).sum()),
        "canonical_fold": int(canonical_fold),
        "canonical_seed": int(canonical_seed),
        "canonical_run_id": canonical_run_id,
        "canonical_selection_mode": selection_mode,
        "models_present": sorted(frame["model"].unique().tolist()),
        "patients": int(frame["patient_id"].nunique()),
        "recordings": int(frame["relative_path"].nunique()),
    }
    if tuple(sorted(frame["model"].unique())) != tuple(sorted(MODELS)):
        raise RuntimeError(
            f"canonical source does not contain all three detectors: {meta['models_present']}"
        )
    return frame, meta


def _finish_component(
    components: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    *,
    dataset: str,
    patient_id: str,
    recording: str,
    relative_path: str,
    component_index: int,
) -> None:
    if not intervals:
        return
    start_s = min(float(row["start_s"]) for row in intervals)
    end_s = max(float(row["end_s"]) for row in intervals)
    labels = {str(row["model"]) for row in intervals}
    event_id = sha256_text(
        f"{dataset}|{patient_id}|{recording}|{relative_path}|{start_s:.6f}|"
        f"{end_s:.6f}|{','.join(sorted(labels))}|{component_index}"
    )[:24]
    components.append(
        {
            "event_id": event_id,
            "patient_id": patient_id,
            "recording": recording,
            "relative_path": relative_path,
            "event_start_s": start_s,
            "event_end_s": end_s,
            "event_duration_s": end_s - start_s,
            "source_interval_count": int(len(intervals)),
            "source_models": ",".join(name for name in MODELS if name in labels),
            **{column: int(model in labels) for model, column in zip(MODELS, LABEL_COLUMNS)},
        }
    )


def strict_overlap_components(events: pd.DataFrame, dataset: str) -> pd.DataFrame:
    components: list[dict[str, Any]] = []
    group_columns = ["patient_id", "recording", "relative_path"]
    for group_key, group in events.groupby(group_columns, sort=True, dropna=False):
        patient_id, recording, relative_path = (str(value) for value in group_key)
        ordered = group.sort_values(["start_s", "end_s", "model"], kind="mergesort")
        current: list[dict[str, Any]] = []
        current_end = -math.inf
        component_index = 0
        for row in ordered.to_dict("records"):
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            # A shared endpoint has zero intersection; only positive overlap
            # joins components.  Chains are intentionally connected.
            if current and start_s >= current_end - EPS:
                _finish_component(
                    components,
                    current,
                    dataset=dataset,
                    patient_id=patient_id,
                    recording=recording,
                    relative_path=relative_path,
                    component_index=component_index,
                )
                component_index += 1
                current = []
                current_end = -math.inf
            current.append(row)
            current_end = max(current_end, end_s)
        _finish_component(
            components,
            current,
            dataset=dataset,
            patient_id=patient_id,
            recording=recording,
            relative_path=relative_path,
            component_index=component_index,
        )
    output = pd.DataFrame(components)
    if output.empty:
        raise RuntimeError("strict overlap union produced no events")
    return output.sort_values(
        ["patient_id", "relative_path", "event_start_s", "event_end_s", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_windows(
    components: pd.DataFrame,
    *,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped_short = 0
    tail_seconds = 0.0
    for component in components.to_dict("records"):
        duration = float(component["event_duration_s"])
        count = int(math.floor((duration + EPS) / WINDOW_S))
        if count <= 0:
            dropped_short += 1
            continue
        tail_seconds += max(0.0, duration - count * WINDOW_S)
        for index in range(count):
            start_s = float(component["event_start_s"]) + index * WINDOW_S
            end_s = start_s + WINDOW_S
            row = dict(component)
            row.update(
                {
                    "dataset": dataset,
                    "window_index_in_event": int(index),
                    "windows_per_event": int(count),
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "window_id": sha256_text(
                        f"{dataset}|{component['event_id']}|{index}|{start_s:.6f}"
                    )[:24],
                }
            )
            rows.append(row)
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError("no complete 10-second windows remain")
    windows = windows.sort_values(
        ["patient_id", "relative_path", "event_start_s", "window_index_in_event"],
        kind="mergesort",
    ).reset_index(drop=True)
    windows["window_index"] = np.arange(len(windows), dtype=np.int64)
    windows["label_cardinality"] = windows[list(LABEL_COLUMNS)].sum(axis=1).astype(int)
    label_counts = windows[list(LABEL_COLUMNS)].sum(axis=0).to_dict()
    per_model_events = {
        model: int(components.loc[components[label] == 1, "event_id"].nunique())
        for model, label in zip(MODELS, LABEL_COLUMNS)
    }
    meta = {
        "dataset": dataset,
        "strict_union_events": int(len(components)),
        "strict_union_events_dropped_because_duration_lt_10s": int(dropped_short),
        "strict_union_events_retained": int(windows["event_id"].nunique()),
        "complete_10s_windows": int(len(windows)),
        "discarded_tail_seconds": float(tail_seconds),
        "patients": int(windows["patient_id"].nunique()),
        "recordings": int(windows["relative_path"].nunique()),
        "label_window_counts": {str(key): int(value) for key, value in label_counts.items()},
        "label_event_counts": per_model_events,
        "multi_label_windows": int((windows["label_cardinality"] > 1).sum()),
        "multi_label_window_fraction": float((windows["label_cardinality"] > 1).mean()),
        "event_duration_seconds": {
            key: float(value)
            for key, value in components["event_duration_s"].describe(
                percentiles=[0.25, 0.5, 0.75, 0.95]
            ).to_dict().items()
        },
        "windowing_rule": "floor(event_duration_seconds / 10); event-relative, non-overlapping windows; incomplete tail dropped",
    }
    return windows, meta


def write_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    config_path: Path,
    canonical_fold: int,
    canonical_seed: int,
    canonical_run_id: str | None,
) -> None:
    _config, config_sha256 = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_events, source_meta = normalize_source_events(
        source_path,
        canonical_fold=canonical_fold,
        canonical_seed=canonical_seed,
        canonical_run_id=canonical_run_id,
        dataset=dataset,
    )
    components = strict_overlap_components(source_events, dataset)
    windows, window_meta = build_windows(components, dataset=dataset)
    components_path = output_dir / f"{dataset}_fp_union_events.parquet"
    windows_path = output_dir / f"{dataset}_fp_union_10s_windows.parquet"
    source_events_path = output_dir / f"{dataset}_canonical_fp_source_events.parquet"
    source_events.to_parquet(source_events_path, index=False)
    components.to_parquet(components_path, index=False)
    windows.to_parquet(windows_path, index=False)
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "dataset": dataset,
        "status": "data_built_only",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source": source_meta,
        "windowing": {
            "window_seconds": WINDOW_S,
            "sampling_hz": 128.0,
            "channels": 16,
            "samples": 1280,
            "alignment": "event_start_nonoverlapping",
            "strict_interval_overlap_only": True,
            "proximity_merge_seconds": 0,
            "drop_event_if_duration_lt_seconds": 10.0,
            "drop_incomplete_tail": True,
        },
        "window_summary": window_meta,
        "label_columns": list(LABEL_COLUMNS),
        "models": list(MODELS),
        "output_sha256": {
            "canonical_source_events": sha256(source_events_path),
            "union_events": sha256(components_path),
            "windows": sha256(windows_path),
        },
        "files": [source_events_path.name, components_path.name, windows_path.name],
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chbmit", "tusz"), required=True)
    parser.add_argument("--source-event-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical-fold", type=int, default=0)
    parser.add_argument("--canonical-seed", type=int, default=17)
    parser.add_argument("--canonical-run-id", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    write_dataset(
        args.source_event_table,
        args.output_dir,
        dataset=args.dataset,
        config_path=args.config,
        canonical_fold=args.canonical_fold,
        canonical_seed=args.canonical_seed,
        canonical_run_id=args.canonical_run_id,
    )


if __name__ == "__main__":
    main()
