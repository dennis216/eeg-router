"""Resumable frozen inference for one standard GroupKFold TUSZ run.

This script performs no training and no threshold selection.  It loads exactly
one frozen standard checkpoint (model, outer fold, seed), uses the threshold
stored in that run's validation-only ``completed.json``, and writes per-EDF
probabilities plus provenance metadata.  Event-level scoring is a separate
post-inference step.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.precomputed import PrecomputedProjectionEncoder
from bfa.models.shared_head import SharedContextHead
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.preprocessing.tuh import MODEL_VIEW_CONFIG, load_tuh_canonical_raw_uv, tuh_model_view, tuh_quality_features


ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
REPORT_DIR = ROOT / "outputs" / "reports" / RUN_ID
DEFAULT_MANIFEST = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "tuh_external" / "tusz_groupkfold_external_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _model_objects(model: str, fold: int, seed: int):
    run = ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main"
    checkpoint = run / "checkpoints" / "step_05000.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["update"]) != 5000:
        raise ValueError(f"unexpected update in {checkpoint}")
    adapter = None
    if model == "singlem":
        adapter = SingLEMAdapter(ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt")
        encoder = PrecomputedProjectionEncoder(16, adapter.projection)
    elif model == "cbramod":
        adapter = CBraModAdapter(ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
        encoder = PrecomputedProjectionEncoder(200, adapter.projection)
    elif model == "tcn_gat":
        encoder = MultiScaleTCNGAT()
        encoder.native_head.requires_grad_(False)
    else:
        raise ValueError(model)
    encoder.load_state_dict(state["encoder"], strict=True)
    head = SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    completed = json.loads((run / "completed.json").read_text())
    threshold = float(completed["final_validation"]["selected_event_operating_point"]["threshold"])
    return adapter, encoder, head, checkpoint, threshold


def _windows(signal: np.ndarray, rate: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(signal, 10 * rate, axis=-1)[:, :: 2 * rate]


@torch.inference_mode()
def _encode_all(model: str, adapter, encoder, view: np.ndarray, quality: np.ndarray) -> np.ndarray:
    rate = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
    windows = _windows(view, rate)
    if abs(windows.shape[1] - quality.shape[0]) > 1:
        raise ValueError(f"window/quality mismatch {windows.shape[1]} vs {quality.shape[0]}")
    limit = min(windows.shape[1], quality.shape[0])
    windows, quality = windows[:, :limit], quality[:limit]
    batch_size = {"singlem": 128, "cbramod": 8, "tcn_gat": 32}[model]
    chunks: list[np.ndarray] = []
    for offset in range(0, limit, batch_size):
        end = min(limit, offset + batch_size)
        batch = np.ascontiguousarray(windows[:, offset:end].transpose(1, 0, 2))
        x = torch.from_numpy(batch).cuda(non_blocking=True)
        q = torch.from_numpy(np.ascontiguousarray(quality[offset:end])).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if model in {"singlem", "cbramod"}:
                x = x.reshape(len(batch), 16, 10, rate)
                frozen = adapter.backbone_features(x)
                embedding = encoder.forward_window(frozen)
            else:
                embedding = encoder.forward_window(x, q)
        chunks.append(embedding.float().cpu().numpy())
    result = np.concatenate(chunks) if chunks else np.empty((0, 16, 128), dtype=np.float32)
    if result.shape != (limit, 16, 128) or not np.isfinite(result).all():
        raise ValueError(f"bad embedding output {result.shape}")
    return result


@torch.inference_mode()
def _predict_contexts(head, embeddings: np.ndarray, batch_size: int = 128) -> np.ndarray:
    if len(embeddings) < 31:
        return np.empty(0, dtype=np.float32)
    probabilities: list[np.ndarray] = []
    for first_target in range(30, len(embeddings), batch_size):
        targets = range(first_target, min(len(embeddings), first_target + batch_size))
        contexts = np.stack([embeddings[index - 30 : index + 1] for index in targets])
        x = torch.from_numpy(np.ascontiguousarray(contexts)).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probability = torch.sigmoid(head(x))
        probabilities.append(probability.float().cpu().numpy())
    output = np.concatenate(probabilities)
    if not np.isfinite(output).all() or not ((output >= 0) & (output <= 1)).all():
        raise ValueError("invalid probability output")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 3407], required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standard TUH frozen inference")
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(manifest_path.read_text())
    cohort_hash = _sha256(manifest_path)
    model = args.model
    fold = int(args.fold)
    seed = int(args.seed)
    run_id = f"fold{fold}_seed{seed}"
    model_root = output_root / model / run_id
    probability_root = model_root / "probabilities"
    metadata_root = model_root / "metadata"
    progress_path = model_root / "progress.json"
    probability_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    adapter, encoder, head, checkpoint, threshold = _model_objects(model, fold, seed)
    checkpoint_hash = _sha256(checkpoint)
    if adapter is not None:
        adapter = adapter.cuda().eval()
    encoder = encoder.cuda().eval()
    head = head.cuda().eval()
    records = manifest["records"]
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    total_windows = 0
    started_all = time.monotonic()
    for number, record in enumerate(records, start=1):
        relative = Path(record["relative_edf"])
        target = probability_root / relative.with_suffix(".parquet")
        metadata_target = metadata_root / relative.with_suffix(".json")
        if target.exists() and metadata_target.exists():
            metadata = json.loads(metadata_target.read_text())
            if metadata.get("cohort_manifest_sha256") != cohort_hash or metadata.get("checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError(f"provenance mismatch for {relative}")
            completed_count += 1
            skipped_count += 1
            total_windows += int(metadata.get("prediction_windows", 0))
            continue
        # A small number of official TUSZ EDFs are shorter than the frozen
        # 70-s warm-up horizon. They contain no valid decision interval for
        # this experiment. Record an immutable empty output instead of
        # treating the absence of a 10-s model window as an inference failure.
        if float(record.get("duration_s", 0.0)) <= 70.0:
            target.parent.mkdir(parents=True, exist_ok=True)
            empty = pd.DataFrame({"patient_id": pd.Series(dtype="string"), "recording": pd.Series(dtype="string"), "window_end_s": pd.Series(dtype="float64"), "probability": pd.Series(dtype="float32")})
            temporary = target.with_suffix(".tmp.parquet")
            empty.to_parquet(temporary, index=False)
            os.replace(temporary, target)
            metadata = {
                "release_id": RUN_ID,
                "external_cohort_id": manifest["external_cohort_id"],
                "cohort_manifest_sha256": cohort_hash,
                "model": model,
                "fold": fold,
                "seed": seed,
                "patient_id": record["patient_id"],
                "relative_edf": relative.as_posix(),
                "edf_sha256": record["edf_sha256"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_update": 5000,
                "threshold": threshold,
                "threshold_source": "standard CHB validation-only final step 5000",
                "prediction_windows": 0,
                "embedding_windows": 0,
                "short_record_no_valid_context": True,
                "probability_sha256": _sha256(target),
                "probability_file": str(target.relative_to(ROOT)),
                "model_training": False,
                "threshold_retuning": False,
            }
            _atomic_json(metadata_target, metadata)
            completed_count += 1
            print(json.dumps({"model": model, "fold": fold, "seed": seed, "record": number, "total": len(records), "status": "skipped_short_record", "completed": completed_count}), flush=True)
            _atomic_json(progress_path, {"release_id": RUN_ID, "external_cohort_id": manifest["external_cohort_id"], "cohort_manifest_sha256": cohort_hash, "model": model, "fold": fold, "seed": seed, "records_total": len(records), "records_completed": completed_count, "records_failed": failed_count, "records_skipped_on_resume": skipped_count, "prediction_windows": total_windows, "checkpoint_sha256": checkpoint_hash, "threshold": threshold, "model_training": False, "threshold_retuning": False, "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started_all, "status": "running_or_incomplete"})
            continue
        record_started = time.monotonic()
        try:
            path = Path(manifest["dataset_root"]) / relative
            raw_uv, source_hz, source_channels = load_tuh_canonical_raw_uv(path)
            quality = tuh_quality_features(raw_uv, source_hz)
            view = tuh_model_view(raw_uv, source_hz, model)
            outside = float(np.mean(np.abs(view) > 1.5)) if model == "singlem" else None
            embeddings = _encode_all(model, adapter, encoder, view, quality)
            probabilities = _predict_contexts(head, embeddings)
            ends = np.arange(30, len(embeddings), dtype=float) * 2.0 + 10.0
            if len(ends) != len(probabilities):
                raise ValueError(f"time alignment failure {len(ends)} vs {len(probabilities)}")
            table = pd.DataFrame({"patient_id": record["patient_id"], "recording": relative.as_posix(), "window_end_s": ends, "probability": probabilities})
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            table.to_parquet(temporary, index=False)
            os.replace(temporary, target)
            metadata = {
                "release_id": RUN_ID,
                "external_cohort_id": manifest["external_cohort_id"],
                "cohort_manifest_sha256": cohort_hash,
                "model": model,
                "fold": fold,
                "seed": seed,
                "patient_id": record["patient_id"],
                "relative_edf": relative.as_posix(),
                "edf_sha256": record["edf_sha256"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_update": 5000,
                "threshold": threshold,
                "threshold_source": "standard CHB validation-only final step 5000",
                "source_sampling_hz": source_hz,
                "source_channels": source_channels,
                "raw_shape": list(raw_uv.shape),
                "model_view_shape": list(view.shape),
                "quality_shape": list(quality.shape),
                "embedding_windows": len(embeddings),
                "prediction_windows": len(probabilities),
                "probability_min": float(probabilities.min()) if len(probabilities) else None,
                "probability_max": float(probabilities.max()) if len(probabilities) else None,
                "probability_mean": float(probabilities.mean()) if len(probabilities) else None,
                "probability_exact_zero_count": int(np.sum(probabilities == 0)),
                "probability_exact_one_count": int(np.sum(probabilities == 1)),
                "singlem_fraction_outside_minus1_5_plus1_5": outside,
                "hard_clipping": False,
                "runtime_s": time.monotonic() - record_started,
                "probability_file": str(target.relative_to(ROOT)),
                "probability_sha256": _sha256(target),
                "model_training": False,
                "threshold_retuning": False,
            }
            _atomic_json(metadata_target, metadata)
            completed_count += 1
            total_windows += len(probabilities)
            print(json.dumps({"model": model, "fold": fold, "seed": seed, "record": number, "total": len(records), "patient": record["patient_id"], "relative_edf": relative.as_posix(), "windows": len(probabilities), "seconds": metadata["runtime_s"], "completed": completed_count, "skipped": skipped_count}), flush=True)
            del raw_uv, quality, view, embeddings, probabilities, table
            gc.collect()
        except Exception as exc:
            failed_count += 1
            _atomic_json(metadata_target.with_suffix(".failed.json"), {"release_id": RUN_ID, "model": model, "fold": fold, "seed": seed, "relative_edf": relative.as_posix(), "error_type": type(exc).__name__, "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"model": model, "fold": fold, "seed": seed, "record": number, "total": len(records), "status": "failed", "error": repr(exc)}), flush=True)
        _atomic_json(progress_path, {"release_id": RUN_ID, "external_cohort_id": manifest["external_cohort_id"], "cohort_manifest_sha256": cohort_hash, "model": model, "fold": fold, "seed": seed, "records_total": len(records), "records_completed": completed_count, "records_failed": failed_count, "records_skipped_on_resume": skipped_count, "prediction_windows": total_windows, "checkpoint_sha256": checkpoint_hash, "threshold": threshold, "model_training": False, "threshold_retuning": False, "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started_all, "status": "complete" if completed_count == len(records) and failed_count == 0 else "running_or_incomplete"})
    if completed_count != len(records) or failed_count:
        raise RuntimeError(f"{model} fold={fold} seed={seed} incomplete completed={completed_count}/{len(records)} failed={failed_count}")
    print(f"TUSZ_GROUPKFOLD_EXTERNAL_INFERENCE_COMPLETE model={model} fold={fold} seed={seed} records={completed_count} windows={total_windows}", flush=True)


if __name__ == "__main__":
    main()
