"""Freeze a deterministic TUSZ inventory and run a real EDF/annotation smoke test.

This script performs format validation only. It does not train a model, select a
threshold, or compute external performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np


RELEASE_ID = "v3-compact-extension-v1"
CANONICAL_BIPOLAR = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)
LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def normalize_electrode(name: str) -> str:
    value = name.upper().strip()
    if value.startswith("EEG "):
        value = value[4:]
    for suffix in ("-REF", "-LE", "-AR", "-AVG"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = value.replace(" ", "")
    return LEGACY_TO_MODERN.get(value, value)


def inventory(root: Path) -> tuple[list[Path], dict[str, object]]:
    edf_root = root / "edf"
    files = sorted(edf_root.rglob("*.edf"))
    if not files:
        raise FileNotFoundError(f"no EDF files found under {edf_root}")

    by_partition = Counter()
    by_montage = Counter()
    patients = defaultdict(set)
    total_bytes = 0
    inventory_hash = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root)
        parts = rel.parts
        partition = parts[1] if len(parts) > 1 else "unknown"
        patient = parts[2] if len(parts) > 2 else "unknown"
        montage = parts[-2] if len(parts) > 1 else "unknown"
        size = path.stat().st_size
        total_bytes += size
        by_partition[partition] += 1
        by_montage[montage] += 1
        patients[partition].add(patient)
        inventory_hash.update(f"{rel.as_posix()}\t{size}\n".encode("utf-8"))

    drive = root.drive.rstrip(":").lower()
    root_tail = root.as_posix().split(":", 1)[-1].lstrip("/")
    root_wsl = f"/mnt/{drive}/{root_tail}" if drive else None
    manifest = {
        "release_id": RELEASE_ID,
        "dataset": "TUSZ_v2.0.6",
        "dataset_root": str(root),
        "dataset_root_wsl": root_wsl,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "edf_files": len(files),
        "csv_files": sum(path.with_suffix(".csv").exists() for path in files),
        "csv_bi_files": sum(Path(str(path.with_suffix(".csv")) + "_bi").exists() for path in files),
        "edf_missing_csv": sum(not path.with_suffix(".csv").exists() for path in files),
        "edf_missing_csv_bi": sum(not Path(str(path.with_suffix(".csv")) + "_bi").exists() for path in files),
        "edf_bytes": total_bytes,
        "edf_gib": total_bytes / (1024 ** 3),
        "edf_files_by_partition": dict(sorted(by_partition.items())),
        "patients_by_partition": {k: len(v) for k, v in sorted(patients.items())},
        "edf_files_by_montage": dict(sorted(by_montage.items())),
        "tree_inventory_sha256": inventory_hash.hexdigest(),
        "inventory_hash_definition": "SHA256 of sorted relative EDF path, tab, byte size, newline; not a content hash of all EDF payloads",
        "readme_sha256": sha256_file(root / "AAREADME.txt") if (root / "AAREADME.txt").exists() else None,
        "download_completeness_claimed": False,
        "download_note": "Local inventory is frozen. Upstream completeness requires comparison with the licensed distribution manifest.",
    }
    return files, manifest


def choose_smoke_files(files: list[Path], root: Path, patients_per_partition: int) -> list[Path]:
    by_partition_patient: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        rel = path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        partition, patient = rel.parts[1], rel.parts[2]
        by_partition_patient[partition][patient].append(path)

    chosen = []
    for partition in ("dev", "train", "eval"):
        patient_map = by_partition_patient.get(partition, {})
        ranked_patients = sorted(
            patient_map,
            key=lambda p: hashlib.sha256(f"{RELEASE_ID}|{partition}|{p}".encode()).hexdigest(),
        )[:patients_per_partition]
        for patient in ranked_patients:
            candidates = patient_map[patient]
            candidates = [p for p in candidates if p.with_suffix(".csv").exists() and Path(str(p) + "_bi").exists()]
            if not candidates:
                candidates = patient_map[patient]
            selected = min(
                candidates,
                key=lambda p: hashlib.sha256(f"{RELEASE_ID}|{p.relative_to(root).as_posix()}".encode()).hexdigest(),
            )
            chosen.append(selected)
    return chosen


def parse_annotation(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    metadata = {}
    header_line = None
    data_lines = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                key, value = body.split("=", 1)
                metadata[key.strip()] = value.strip()
            continue
        if header_line is None:
            header_line = line
        else:
            data_lines.append(line)
    if header_line is None:
        raise ValueError(f"missing CSV header: {path}")
    rows = list(csv.DictReader([header_line] + data_lines))
    return rows, metadata


def smoke_record(edf_path: Path, root: Path) -> dict[str, object]:
    rel = edf_path.relative_to(root)
    partition, patient = rel.parts[1], rel.parts[2]
    csv_path = edf_path.with_suffix(".csv")
    csv_bi_path = Path(str(csv_path) + "_bi")
    row: dict[str, object] = {
        "partition": partition,
        "patient_id": patient,
        "relative_edf": rel.as_posix(),
        "edf_bytes": edf_path.stat().st_size,
        "edf_sha256": sha256_file(edf_path),
        "csv_exists": csv_path.exists(),
        "csv_bi_exists": csv_bi_path.exists(),
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = row["errors"]  # type: ignore[assignment]
    warnings: list[str] = row["warnings"]  # type: ignore[assignment]
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        duration = float(raw.n_times / sfreq)
        row.update({
            "sfreq_hz": sfreq,
            "duration_s": duration,
            "n_channels": len(raw.ch_names),
            "mne_original_units": dict(sorted((raw._orig_units or {}).items())),
        })

        electrode_to_channel = {}
        for name in raw.ch_names:
            normalized = normalize_electrode(name)
            if normalized not in electrode_to_channel and normalized not in {"PHOTICPH", "EKG"}:
                electrode_to_channel[normalized] = name

        missing_pairs = []
        for pair in CANONICAL_BIPOLAR:
            left, right = pair.split("-")
            if left not in electrode_to_channel or right not in electrode_to_channel:
                missing_pairs.append(pair)
        row["canonical_bipolar_available"] = len(CANONICAL_BIPOLAR) - len(missing_pairs)
        row["missing_canonical_bipolar"] = missing_pairs
        row["legacy_temporal_mapping_used"] = any(normalize_electrode(x) != x.upper().replace("EEG ", "").split("-")[0] for x in raw.ch_names if any(t in x.upper() for t in ("T3", "T4", "T5", "T6")))
        if missing_pairs:
            errors.append("missing_canonical_bipolar")

        required_electrodes = sorted({x for pair in CANONICAL_BIPOLAR for x in pair.split("-")})
        picks = [electrode_to_channel[x] for x in required_electrodes if x in electrode_to_channel]
        sample_chunks = []
        window_s = min(10.0, duration)
        starts = sorted(set([0.0, max(0.0, duration / 2 - window_s / 2), max(0.0, duration - window_s)]))
        for start_s in starts:
            start = int(start_s * sfreq)
            stop = min(raw.n_times, start + int(window_s * sfreq))
            if stop > start and picks:
                sample_chunks.append(raw.get_data(picks=picks, start=start, stop=stop))
        samples = np.concatenate(sample_chunks, axis=1) if sample_chunks else np.empty((0, 0))
        row["sample_values"] = int(samples.size)
        row["sample_finite_fraction"] = float(np.isfinite(samples).mean()) if samples.size else 0.0
        row["sample_abs_p99_uv"] = float(np.nanpercentile(np.abs(samples), 99) * 1e6) if samples.size else None
        row["sample_abs_max_uv"] = float(np.nanmax(np.abs(samples)) * 1e6) if samples.size else None
        if not samples.size or not np.isfinite(samples).all():
            errors.append("nonfinite_or_empty_sample")
        elif float(row["sample_abs_p99_uv"]) > 1000.0:
            warnings.append("extreme_amplitude_p99_gt_1000_uv")
        elif float(row["sample_abs_max_uv"]) > 1000.0:
            warnings.append("extreme_amplitude_max_gt_1000_uv")

        annotation_summary = {}
        for kind, path in (("csv", csv_path), ("csv_bi", csv_bi_path)):
            if not path.exists():
                errors.append(f"missing_{kind}")
                continue
            ann_rows, meta = parse_annotation(path)
            bad_bounds = 0
            labels = Counter()
            for ann in ann_rows:
                try:
                    start_t = float(ann["start_time"])
                    stop_t = float(ann["stop_time"])
                    if start_t < 0 or stop_t <= start_t or stop_t > duration + 1.0:
                        bad_bounds += 1
                    labels[ann["label"]] += 1
                except Exception:
                    bad_bounds += 1
            meta_duration = None
            if "duration" in meta:
                try:
                    meta_duration = float(meta["duration"].split()[0])
                except Exception:
                    pass
            duration_delta = abs(meta_duration - duration) if meta_duration is not None else None
            annotation_summary[kind] = {
                "rows": len(ann_rows),
                "labels": dict(sorted(labels.items())),
                "bad_event_bounds": bad_bounds,
                "metadata_duration_s": meta_duration,
                "metadata_edf_duration_abs_delta_s": duration_delta,
                "sha256": sha256_file(path),
            }
            if bad_bounds:
                errors.append(f"{kind}_bad_event_bounds")
            if duration_delta is None or duration_delta > max(1.0 / sfreq, 0.02):
                errors.append(f"{kind}_duration_mismatch")
        row["annotations"] = annotation_summary
    except Exception as exc:
        errors.append(f"edf_read_exception:{type(exc).__name__}:{exc}")
    row["pass"] = not errors
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patients-per-partition", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files, manifest = inventory(root)
    selected = choose_smoke_files(files, root, args.patients_per_partition)
    results = [smoke_record(path, root) for path in selected]

    manifest["smoke_selection"] = {
        "patients_per_partition": args.patients_per_partition,
        "selection": "lowest SHA256(release_id|partition|patient_id), then deterministic annotated EDF per patient",
        "selected_records": [r["relative_edf"] for r in results],
    }
    summary = {
        "release_id": RELEASE_ID,
        "dataset": "TUSZ_v2.0.6",
        "dataset_root": str(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "real_format_smoke_only",
        "model_training": False,
        "model_inference": False,
        "threshold_selection": False,
        "selected_records": len(results),
        "selected_patients": len({(r["partition"], r["patient_id"]) for r in results}),
        "records_passed": sum(bool(r["pass"]) for r in results),
        "records_failed": sum(not bool(r["pass"]) for r in results),
        "records_with_warnings": sum(bool(r["warnings"]) for r in results),
        "all_pass": all(bool(r["pass"]) for r in results),
        "canonical_bipolar_target": list(CANONICAL_BIPOLAR),
        "legacy_mapping": LEGACY_TO_MODERN,
        "records": results,
        "next_gate": "freeze real-data adapter and run 1-2-record three-model inference smoke" if all(bool(r["pass"]) for r in results) else "fix format/channel/annotation failures before model inference",
    }

    manifest_path = output_dir / "tuh_tusz_v2_0_6_dataset_manifest.json"
    smoke_path = output_dir / "tuh_real_format_smoke.json"
    csv_out = output_dir / "tuh_real_format_smoke_records.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    smoke_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "partition", "patient_id", "relative_edf", "edf_bytes", "edf_sha256", "sfreq_hz", "duration_s",
            "n_channels", "canonical_bipolar_available", "sample_finite_fraction", "sample_abs_p99_uv",
            "sample_abs_max_uv", "csv_exists", "csv_bi_exists", "pass", "errors",
            "warnings",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: json.dumps(r[k]) if k in {"errors", "warnings"} else r.get(k) for k in fields})

    print(json.dumps({
        "manifest": str(manifest_path),
        "smoke": str(smoke_path),
        "records_csv": str(csv_out),
        "edf_files": manifest["edf_files"],
        "patients_by_partition": manifest["patients_by_partition"],
        "selected_records": summary["selected_records"],
        "passed": summary["records_passed"],
        "failed": summary["records_failed"],
        "warnings": summary["records_with_warnings"],
        "all_pass": summary["all_pass"],
        "next_gate": summary["next_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
