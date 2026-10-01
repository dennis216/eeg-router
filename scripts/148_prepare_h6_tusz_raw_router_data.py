"""Prepare official TUSZ manifests and causal 60-s raw-EEG contexts.

The resulting samples are used by the *pre-router*: raw EEG from the past
60 seconds is paired with labels computed from the frozen experts' outputs in
the following 60 seconds.  This script never opens expert probabilities and
never creates labels; label construction is a separate audit step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv, tuh_model_view
from fractions import Fraction
from scipy.signal import resample_poly


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
DATASET_INVENTORY = ROOT / "outputs/reports/v3-compact-extension-v1/tuh_tusz_v2_0_6_dataset_manifest.json"
PARTITIONS = ("train", "dev", "eval")
CONTEXT_SECONDS = 60.0
FUTURE_SECONDS = 60.0
DECISION_START = 70.0  # frozen expert warm-up is 70 s in the standard scorer
DECISION_STRIDE = 60.0
RATE = 128
CONTEXT_SAMPLES = int(CONTEXT_SECONDS * RATE)


def canonical_raw_128(raw_uv: np.ndarray, source_hz: float) -> np.ndarray:
    """Return canonical bipolar EEG in microvolts, resampled only to 128 Hz.

    No detector-specific band-pass, scaling, clipping, or expert output is
    applied.  This is the causal router's raw-EEG input view.
    """
    ratio = Fraction(float(RATE) / float(source_hz)).limit_denominator(10_000)
    output = resample_poly(raw_uv.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1)
    if not np.isfinite(output).all():
        raise ValueError("non-finite canonical raw resample")
    return output.astype(np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_file_valid(path: Path, expected_rows: int) -> bool:
    """Validate an existing raw-cache file before honoring --resume.

    A prior interrupted write can leave a zero-length/truncated NPZ even when
    the progress marker was written.  Resume must regenerate such files rather
    than treating mere existence as completion.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "x" not in archive.files:
                return False
            array = archive["x"]
            return bool(
                array.ndim == 3
                and array.shape == (expected_rows, 16, CONTEXT_SAMPLES)
                and array.dtype.kind in "fc"
                and np.isfinite(array).all()
            )
    except Exception:
        return False


def read_annotations(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def duration_from_annotation(path: Path) -> float:
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if line.lower().startswith("# duration") and "=" in line:
            return float(line.split("=", 1)[1].split()[0])
    raise RuntimeError(f"duration header missing in {path}")


def stat_fingerprint(path: Path) -> str:
    """Fast provenance fingerprint for files on the Windows-mounted dataset.

    Reading every EDF to compute a content SHA256 is prohibitively slow over
    the WSL 9p mount (and is unnecessary for this data-preparation index).
    The frozen dataset tree inventory hash plus this size/mtime fingerprint is
    retained in the manifest; raw content hashes can be audited later on a
    local filesystem without changing the sample definitions.
    """
    stat = path.stat()
    return "stat-v1:" + hashlib.sha256(
        f"{path.as_posix()}\\0{stat.st_size}\\0{stat.st_mtime_ns}".encode()
    ).hexdigest()


def record_inventory(partition: str, limit_patients: int | None = None, *, fast: bool = False) -> list[dict[str, object]]:
    root = DATA_ROOT / "edf" / partition
    patients = sorted(p for p in root.iterdir() if p.is_dir())
    if limit_patients is not None:
        patients = patients[:limit_patients]
    records: list[dict[str, object]] = []
    for patient_dir in patients:
        for edf in sorted(patient_dir.rglob("*.edf")):
            bi = Path(str(edf.with_suffix(".csv")) + "_bi")
            if not bi.is_file():
                raise FileNotFoundError(bi)
            annotations = read_annotations(bi)
            seizures = [
                {"start_s": float(row["start_time"]), "end_s": float(row["stop_time"]), "label": str(row["label"])}
                for row in annotations
                if str(row.get("label", "")).lower() == "seiz"
            ]
            relative = edf.relative_to(DATA_ROOT).as_posix()
            duration = duration_from_annotation(bi)
            n_samples = max(0, int(np.floor((duration - DECISION_START - FUTURE_SECONDS) / DECISION_STRIDE)) + 1)
            records.append(
                {
                    "patient_id": patient_dir.name,
                    "relative_edf": relative,
                    "edf_sha256": stat_fingerprint(edf) if fast else sha256(edf),
                    "edf_hash_type": "stat_fingerprint" if fast else "sha256",
                    "edf_bytes": int(edf.stat().st_size),
                    "csv_bi_sha256": sha256(bi),
                    "duration_s": duration,
                    "seizure_events": seizures,
                    "decision_count": n_samples,
                }
            )
    return records


def build_manifest(partition: str, records: list[dict[str, object]]) -> Path:
    path = OUT / "manifests" / f"{partition}_manifest.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen manifest: {path}")
    patients = sorted({str(r["patient_id"]) for r in records})
    payload = {
        "protocol_id": NAMESPACE,
        "dataset": "TUSZ_v2.0.6",
        "partition": partition,
        "selection_rule": "official TUSZ partition; all EDFs for a patient remain together",
        "external_cohort_id": f"h6-tusz-{partition}-v1",
        "dataset_root": str(DATA_ROOT),
        "patient_count": len(patients),
        "record_count": len(records),
        "decision_rule": {
            "context_end_start_s": DECISION_START,
            "stride_s": DECISION_STRIDE,
            "context_seconds": CONTEXT_SECONDS,
            "future_horizon_seconds": FUTURE_SECONDS,
            "context_samples": CONTEXT_SAMPLES,
            "sampling_hz": RATE,
            "preprocessing": "canonical bipolar microvolts; polyphase resample only; no detector filter, scale, clipping, or expert output",
        },
        "patients": patients,
        "records": records,
        "source_tree_inventory_sha256": (
            json.loads(DATASET_INVENTORY.read_text()).get("tree_inventory_sha256")
            if DATASET_INVENTORY.exists() else None
        ),
        "hash_policy": "dataset tree inventory + EDF size/mtime fingerprint; annotation content SHA256",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".sha256").write_text(sha256(path) + "\n")
    return path


def build_index(partition: str, manifest: dict[str, object]) -> Path:
    path = OUT / "indices" / f"{partition}_index.parquet"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen index: {path}")
    rows: list[dict[str, object]] = []
    cache_root = OUT / "raw_cache" / partition
    for record in manifest["records"]:
        relative = str(record["relative_edf"])
        patient = str(record["patient_id"])
        cache_path = cache_root / Path(relative).with_suffix(".npz")
        count = int(record["decision_count"])
        for k in range(count):
            context_end = DECISION_START + k * DECISION_STRIDE
            future_start = context_end
            future_end = context_end + FUTURE_SECONDS
            truths = [
                event
                for event in record["seizure_events"]
                if float(event["end_s"]) > future_start and float(event["start_s"]) < future_end
            ]
            rows.append(
                {
                    "sample_id": f"{partition}|{patient}|{relative}|{k}",
                    "partition": partition,
                    "patient_id": patient,
                    "recording": relative,
                    "edf_sha256": str(record["edf_sha256"]),
                    "context_start_s": context_end - CONTEXT_SECONDS,
                    "context_end_s": context_end,
                    "future_start_s": future_start,
                    "future_end_s": future_end,
                    "future_has_seizure": bool(truths),
                    "cache_path": str(cache_path.relative_to(OUT)),
                    "cache_row": k,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no raw-router samples for {partition}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def extract_raw(partition: str, manifest: dict[str, object], index: pd.DataFrame, resume: bool) -> None:
    cache_root = OUT / "raw_cache" / partition
    progress_path = OUT / "raw_cache" / f"{partition}_progress.json"
    completed_records = 0
    failed_records = 0
    for number, record in enumerate(manifest["records"], start=1):
        relative = str(record["relative_edf"])
        target = cache_root / Path(relative).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = index[index.recording == relative].sort_values("cache_row")
        if target.exists():
            if not resume:
                raise RuntimeError(f"raw cache exists; use --resume: {target}")
            if len(rows) == 0 or cache_file_valid(target, len(rows)):
                completed_records += 1
                if len(rows) == 0:
                    print(f"H6_RAW_CACHE_PROGRESS partition={partition} records={number}/{len(manifest['records'])} samples=0 skipped_no_decision_windows", flush=True)
                continue
            # Preserve the damaged artifact for forensic review, then rebuild
            # it from the original EDF instead of trusting file existence.
            # Keep quarantine on the same filesystem as the cache.  The formal
            # cache may be redirected to the D: drive; using OUT here would
            # make os.replace cross devices and could fail with EXDEV.
            quarantine = cache_root.parent / "raw_cache_quarantine_v1" / partition / Path(relative).with_suffix(".npz")
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if quarantine.exists():
                quarantine = quarantine.with_name(quarantine.stem + f".record{number}" + quarantine.suffix)
            os.replace(target, quarantine)
            print(f"H6_RAW_CACHE_REGENERATE partition={partition} record={number} reason=invalid_npz quarantine={quarantine}", flush=True)
        # Official TUSZ manifests can contain short records with zero valid
        # decision windows. They are valid cohort records but have no router
        # samples; mark the record handled without attempting np.stack([]).
        if len(rows) == 0:
            completed_records += 1
            progress_path.write_text(json.dumps({"partition": partition, "records_total": len(manifest["records"]), "records_completed": completed_records, "records_failed": failed_records, "status": "running", "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
            continue
        path = DATA_ROOT / relative
        raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(path)
        view = canonical_raw_128(raw_uv, source_hz)
        contexts: list[np.ndarray] = []
        for row in rows.itertuples(index=False):
            start = int(round(float(row.context_start_s) * RATE))
            end = start + CONTEXT_SAMPLES
            if start < 0 or end > view.shape[-1]:
                continue
            contexts.append(np.asarray(view[:, start:end], dtype=np.float16))
        if len(contexts) != len(rows):
            raise RuntimeError(f"raw context extraction mismatch {relative}: {len(contexts)} vs {len(rows)}")
        array = np.stack(contexts).astype(np.float16, copy=False)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, x=array)
        os.replace(temporary, target)
        completed_records += 1
        progress_path.write_text(json.dumps({"partition": partition, "records_total": len(manifest["records"]), "records_completed": completed_records, "records_failed": failed_records, "status": "running", "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
        print(f"H6_RAW_CACHE_PROGRESS partition={partition} records={number}/{len(manifest['records'])} samples={len(rows)}", flush=True)
    progress_path.write_text(json.dumps({"partition": partition, "records_total": len(manifest["records"]), "records_completed": completed_records, "records_failed": failed_records, "status": "complete", "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--limit-patients", type=int)
    parser.add_argument("--fast-manifest", action="store_true", help="Use stat fingerprints instead of reading all EDF bytes")
    parser.add_argument("--extract-raw", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifests" / f"{args.partition}_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        records = record_inventory(args.partition, args.limit_patients, fast=args.fast_manifest)
        manifest_path = build_manifest(args.partition, records)
        manifest = json.loads(manifest_path.read_text())
    index_path = OUT / "indices" / f"{args.partition}_index.parquet"
    if index_path.exists():
        index = pd.read_parquet(index_path)
    else:
        index_path = build_index(args.partition, manifest)
        index = pd.read_parquet(index_path)
    if args.extract_raw:
        extract_raw(args.partition, manifest, index, args.resume)
    print(json.dumps({"partition": args.partition, "manifest": str(manifest_path), "index": str(index_path), "records": len(manifest["records"]), "samples": len(index), "raw_extracted": args.extract_raw}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
