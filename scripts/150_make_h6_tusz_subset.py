"""Create a small, immutable H6 smoke subset from a frozen TUSZ manifest.

The subset is used only for adapter/label/training smoke tests.  It never
selects patients using expert output or performance and is explicitly marked
``smoke_same_official_partition`` so it cannot be counted as a formal result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import resample_poly

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
RATE = 128
CONTEXT_SECONDS = 60.0
FUTURE_SECONDS = 60.0
DECISION_START = 70.0
DECISION_STRIDE = 60.0
CONTEXT_SAMPLES = int(RATE * CONTEXT_SECONDS)


def canonical_raw_128(raw_uv: np.ndarray, source_hz: float) -> np.ndarray:
    ratio = Fraction(float(RATE) / float(source_hz)).limit_denominator(10_000)
    output = resample_poly(raw_uv.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1)
    if not np.isfinite(output).all():
        raise ValueError("non-finite canonical raw resample")
    return output.astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--patients", type=int, default=1)
    parser.add_argument("--name", default="smoke")
    args = parser.parse_args()
    if args.patients < 1:
        raise ValueError("--patients must be positive")
    source = OUT / "manifests" / f"{args.partition}_manifest.json"
    if not source.exists():
        raise FileNotFoundError(source)
    source_payload = json.loads(source.read_text())
    chosen = set(source_payload["patients"][: args.patients])
    records = [record for record in source_payload["records"] if record["patient_id"] in chosen]
    if not records:
        raise RuntimeError("subset has no records")
    subset_id = f"{args.partition}_{args.name}"
    manifest_path = OUT / "subsets" / f"{subset_id}_manifest.json"
    index_path = OUT / "subsets" / f"{subset_id}_index.parquet"
    cache_root = OUT / "raw_cache" / subset_id
    if manifest_path.exists() or index_path.exists():
        raise RuntimeError(f"refusing to overwrite smoke subset: {manifest_path}")
    subset = {
        "release_id": NAMESPACE,
        "external_cohort_id": f"{NAMESPACE}-{subset_id}",
        "partition": args.partition,
        "selection_type": "smoke_same_official_partition",
        "selection_rule": "first canonical patients in frozen official manifest; no model-output selection",
        "source_manifest": str(source.relative_to(ROOT)),
        "source_manifest_sha256": sha256(source),
        "dataset_root": source_payload["dataset_root"],
        "decision_rule": {
            "sampling_hz": RATE,
            "channels": 16,
            "context_seconds": CONTEXT_SECONDS,
            "future_horizon_seconds": FUTURE_SECONDS,
            "decision_start_s": DECISION_START,
            "decision_stride_s": DECISION_STRIDE,
            "context_shape": [16, CONTEXT_SAMPLES],
        },
        "patients": sorted(chosen),
        "records": records,
        "patient_count": len(chosen),
        "record_count": len(records),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    rows: list[dict[str, object]] = []
    for record in records:
        relative = str(record["relative_edf"])
        count = int(record.get("decision_count", 0))
        cache_path = cache_root / Path(relative).with_suffix(".npz")
        for k in range(count):
            context_end = DECISION_START + k * DECISION_STRIDE
            future_start, future_end = context_end, context_end + FUTURE_SECONDS
            truths = [
                event for event in record["seizure_events"]
                if float(event["end_s"]) > future_start and float(event["start_s"]) < future_end
            ]
            rows.append({
                "sample_id": f"{subset_id}|{record['patient_id']}|{relative}|{k}",
                "partition": args.partition,
                "patient_id": str(record["patient_id"]),
                "recording": relative,
                "edf_sha256": str(record["edf_sha256"]),
                "context_start_s": context_end - CONTEXT_SECONDS,
                "context_end_s": context_end,
                "future_start_s": future_start,
                "future_end_s": future_end,
                "future_has_seizure": bool(truths),
                "cache_path": str(cache_path.relative_to(OUT)),
                "cache_row": k,
            })
    index = pd.DataFrame(rows)
    if index.empty:
        raise RuntimeError("subset produced no decision windows")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")
    manifest_path.with_suffix(".sha256").write_text(sha256(manifest_path) + "\n")
    index.to_parquet(index_path, index=False)
    index_path.with_suffix(".sha256").write_text(sha256(index_path) + "\n")

    for number, record in enumerate(records, start=1):
        relative = str(record["relative_edf"])
        target = cache_root / Path(relative).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        record_rows = index[index.recording == relative].sort_values("cache_row")
        path = Path(source_payload["dataset_root"]) / relative
        raw_uv, source_hz, channels = load_tuh_canonical_raw_uv(path)
        view = canonical_raw_128(raw_uv, source_hz)
        contexts = []
        for row in record_rows.itertuples(index=False):
            start = int(round(float(row.context_start_s) * RATE))
            end = start + CONTEXT_SAMPLES
            if start < 0 or end > view.shape[-1]:
                raise RuntimeError(f"context out of bounds for {relative}: {start}:{end}/{view.shape[-1]}")
            contexts.append(np.asarray(view[:, start:end], dtype=np.float16))
        array = np.stack(contexts).astype(np.float16, copy=False)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, x=array)
        os.replace(temporary, target)
        print(json.dumps({"partition": args.partition, "record": number, "records": len(records), "relative_edf": relative, "samples": len(array), "source_hz": source_hz, "channels": channels}, sort_keys=True), flush=True)
    print(json.dumps({"manifest": str(manifest_path), "index": str(index_path), "cache_root": str(cache_root), "records": len(records), "samples": len(index), "smoke_only": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
