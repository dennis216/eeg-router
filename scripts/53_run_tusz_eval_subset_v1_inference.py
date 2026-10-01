"""Resumable continuous inference for the frozen TUSZ eval subset v1."""

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


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
MANIFEST_PATH = REPORT_DIR / "tusz_eval_subset_v1_manifest.json"
OUTPUT_ROOT = ROOT / "outputs" / "tuh_external" / "tusz_eval_subset_v1"
RUN_ROOT = ROOT / "runs" / "BSN-pilot-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def model_objects(model: str):
    run = RUN_ROOT / model / "split17_seed17_main"
    checkpoint = run / "checkpoints" / "step_05000.pt"
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


def windows(signal: np.ndarray, rate: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(signal, 10 * rate, axis=-1)[:, :: 2 * rate]


@torch.inference_mode()
def encode_all(model: str, adapter, encoder, view: np.ndarray, quality: np.ndarray) -> np.ndarray:
    rate = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
    win = windows(view, rate)
    limit = min(win.shape[1], quality.shape[0])
    if abs(win.shape[1] - quality.shape[0]) > 1:
        raise ValueError(f"window/quality mismatch {win.shape[1]} vs {quality.shape[0]}")
    win, quality = win[:, :limit], quality[:limit]
    batch_size = {"singlem": 128, "cbramod": 8, "tcn_gat": 32}[model]
    outputs = []
    for offset in range(0, limit, batch_size):
        end = min(limit, offset + batch_size)
        batch = np.ascontiguousarray(win[:, offset:end].transpose(1, 0, 2))
        x = torch.from_numpy(batch).cuda(non_blocking=True)
        q = torch.from_numpy(np.ascontiguousarray(quality[offset:end])).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if model in {"singlem", "cbramod"}:
                x = x.reshape(len(batch), 16, 10, rate)
                frozen = adapter.backbone_features(x)
                embedding = encoder.forward_window(frozen)
            else:
                embedding = encoder.forward_window(x, q)
        outputs.append(embedding.float().cpu().numpy())
    result = np.concatenate(outputs) if outputs else np.empty((0, 16, 128), dtype=np.float32)
    if result.shape != (limit, 16, 128) or not np.isfinite(result).all():
        raise ValueError(f"bad embedding output {result.shape}")
    return result


@torch.inference_mode()
def predict_contexts(head, embeddings: np.ndarray, batch_size: int = 128) -> np.ndarray:
    if len(embeddings) < 31:
        return np.empty(0, dtype=np.float32)
    probabilities = []
    for first_target in range(30, len(embeddings), batch_size):
        targets = range(first_target, min(len(embeddings), first_target + batch_size))
        contexts = np.stack([embeddings[i - 30 : i + 1] for i in targets])
        x = torch.from_numpy(np.ascontiguousarray(contexts)).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probability = torch.sigmoid(head(x))
        probabilities.append(probability.float().cpu().numpy())
    output = np.concatenate(probabilities)
    if not np.isfinite(output).all() or not ((output >= 0) & (output <= 1)).all():
        raise ValueError("invalid probabilities")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["singlem", "cbramod", "tcn_gat"], required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    model = args.model
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(manifest_path.read_text())
    frozen_hash = sha256(manifest_path)
    model_root = output_root / model
    probability_root = model_root / "probabilities"
    metadata_root = model_root / "metadata"
    progress_path = model_root / "progress.json"
    probability_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    adapter, encoder, head, checkpoint, threshold = model_objects(model)
    if adapter is not None:
        adapter = adapter.cuda().eval()
    encoder = encoder.cuda().eval()
    head = head.cuda().eval()
    records = manifest["records"]
    started_all = time.monotonic()
    completed = 0
    skipped = 0
    failed = 0
    total_windows = 0
    for number, record in enumerate(records, start=1):
        relative = Path(record["relative_edf"])
        target = probability_root / relative.with_suffix(".parquet")
        metadata_target = metadata_root / relative.with_suffix(".json")
        if target.exists() and metadata_target.exists():
            metadata = json.loads(metadata_target.read_text())
            if metadata.get("cohort_manifest_sha256") != frozen_hash or metadata.get("checkpoint_sha256") != sha256(checkpoint):
                raise RuntimeError(f"provenance mismatch for {relative}")
            completed += 1
            skipped += 1
            total_windows += int(metadata["prediction_windows"])
            continue
        record_started = time.monotonic()
        try:
            raw_uv, source_hz, source_channels = load_tuh_canonical_raw_uv(Path(manifest["dataset_root"]) / relative)
            quality = tuh_quality_features(raw_uv, source_hz)
            view = tuh_model_view(raw_uv, source_hz, model)
            singlem_outside = float(np.mean(np.abs(view) > 1.5)) if model == "singlem" else None
            embeddings = encode_all(model, adapter, encoder, view, quality)
            probabilities = predict_contexts(head, embeddings)
            ends = np.arange(30, len(embeddings), dtype=float) * 2.0 + 10.0
            if len(ends) != len(probabilities):
                raise ValueError("probability time alignment failure")
            table = pd.DataFrame({
                "patient_id": record["patient_id"],
                "recording": relative.as_posix(),
                "window_end_s": ends,
                "probability": probabilities,
            })
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            table.to_parquet(temporary, index=False)
            os.replace(temporary, target)
            metadata = {
                "release_id": "v3-compact-extension-v1",
                "cohort_id": manifest["cohort_id"],
                "cohort_manifest_sha256": frozen_hash,
                "model": model,
                "patient_id": record["patient_id"],
                "relative_edf": relative.as_posix(),
                "edf_sha256": record["edf_sha256"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_update": 5000,
                "threshold": threshold,
                "threshold_source": "CHB validation-only split17 seed17 final step 5000",
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
                "singlem_fraction_outside_minus1_5_plus1_5": singlem_outside,
                "hard_clipping": False,
                "runtime_s": time.monotonic() - record_started,
                "probability_file": str(target.relative_to(ROOT)),
                "probability_sha256": sha256(target),
                "model_training": False,
                "threshold_retuning": False,
            }
            atomic_json(metadata_target, metadata)
            completed += 1
            total_windows += len(probabilities)
            print(json.dumps({
                "model": model, "record": number, "total": len(records), "patient": record["patient_id"],
                "relative_edf": relative.as_posix(), "windows": len(probabilities),
                "seconds": metadata["runtime_s"], "completed": completed, "skipped": skipped,
            }), flush=True)
            del raw_uv, quality, view, embeddings, probabilities, table
            gc.collect()
        except Exception as exc:
            failed += 1
            atomic_json(metadata_target.with_suffix(".failed.json"), {
                "model": model, "relative_edf": relative.as_posix(), "error_type": type(exc).__name__,
                "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(json.dumps({"model": model, "record": number, "total": len(records), "status": "failed", "error": repr(exc)}), flush=True)
        atomic_json(progress_path, {
            "release_id": "v3-compact-extension-v1", "cohort_id": manifest["cohort_id"],
            "cohort_manifest_sha256": frozen_hash, "model": model, "records_total": len(records),
            "records_completed": completed, "records_failed": failed, "records_skipped_on_resume": skipped,
            "prediction_windows": total_windows, "checkpoint_sha256": sha256(checkpoint),
            "threshold": threshold, "model_training": False, "threshold_retuning": False,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started_all,
            "status": "complete" if completed == len(records) and failed == 0 else "running_or_incomplete",
        })
    if completed != len(records) or failed:
        raise RuntimeError(f"{model} incomplete completed={completed}/{len(records)} failed={failed}")
    print(f"TUSZ_SUBSET_INFERENCE_COMPLETE model={model} records={completed} windows={total_windows}", flush=True)


if __name__ == "__main__":
    main()
