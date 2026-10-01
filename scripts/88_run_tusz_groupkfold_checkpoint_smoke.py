"""Run a two-record frozen-checkpoint smoke for the standard GroupKFold models.

The smoke uses the prespecified fold0/seed17/step5000 checkpoints and the
validation-only thresholds stored in their ``completed.json`` files.  It does
not tune, score, or alter any external result.
"""

from __future__ import annotations

import gc
import hashlib
import json
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
MANIFEST_PATH = REPORT_DIR / "tusz_groupkfold_external_manifest.json"
OUTPUT = REPORT_DIR / "tusz_groupkfold_checkpoint_smoke.json"
PROBABILITY_OUTPUT = REPORT_DIR / "tusz_groupkfold_checkpoint_smoke_probabilities.csv"
MODELS = ("singlem", "cbramod", "tcn_gat")
FOLD = 0
SEED = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_objects(model: str):
    run = ROOT / "runs" / RUN_ID / model / f"split{FOLD}_seed{SEED}_main"
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


def _window_view(signal: np.ndarray, rate: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(signal, 10 * rate, axis=-1)[:, :: 2 * rate]


@torch.inference_mode()
def _infer_record(model: str, adapter, encoder, head, raw_uv: np.ndarray, source_hz: float, quality: np.ndarray, target_indices: list[int]):
    view = tuh_model_view(raw_uv, source_hz, model)
    rate = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
    windows = _window_view(view, rate)
    if abs(windows.shape[1] - quality.shape[0]) > 1:
        raise ValueError(f"{model} window/quality mismatch {windows.shape[1]} vs {quality.shape[0]}")
    limit = min(windows.shape[1], quality.shape[0])
    windows, quality = windows[:, :limit], quality[:limit]
    if max(target_indices) >= windows.shape[1] or min(target_indices) < 30:
        raise ValueError("target context index out of bounds")
    target_times = np.asarray(target_indices, dtype=float) * 2.0 + 10.0
    if model in {"singlem", "cbramod"}:
        needed = list(range(min(target_indices) - 30, max(target_indices) + 1))
        batch_size = 32 if model == "singlem" else 8
        chunks = []
        for offset in range(0, len(needed), batch_size):
            ids = needed[offset : offset + batch_size]
            batch = np.ascontiguousarray(windows[:, ids].transpose(1, 0, 2)).reshape(len(ids), 16, 10, rate)
            inputs = torch.from_numpy(batch).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunks.append(adapter.backbone_features(inputs).float().cpu().numpy())
        features = np.concatenate(chunks)
        first_needed = needed[0]
        contexts = np.stack([features[i - 30 - first_needed : i + 1 - first_needed] for i in target_indices])
        input_tensor = torch.from_numpy(np.ascontiguousarray(contexts)).cuda()
        quality_tensor = torch.from_numpy(np.stack([quality[i - 30 : i + 1] for i in target_indices])).cuda()
    else:
        contexts = np.stack([windows[:, i - 30 : i + 1].transpose(1, 0, 2) for i in target_indices])
        input_tensor = torch.from_numpy(np.ascontiguousarray(contexts)).cuda()
        quality_tensor = torch.from_numpy(np.stack([quality[i - 30 : i + 1] for i in target_indices])).cuda()
    probabilities = []
    for offset in range(0, len(target_indices), 2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = encoder.forward_window_sequence(input_tensor[offset : offset + 2], quality_tensor[offset : offset + 2])
            probabilities.append(torch.sigmoid(head(embedding)).float().cpu().numpy())
    output = np.concatenate(probabilities)
    return target_times, output, {
        "raw_shape": list(raw_uv.shape),
        "view_shape": list(view.shape),
        "quality_shape": list(quality.shape),
        "singlem_fraction_outside_minus1_5_plus1_5": float(np.mean(np.abs(view) > 1.5)) if model == "singlem" else None,
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    # A 31-window context needs at least 40 two-second windows.  This smoke
    # selection rule is fixed before model loading and only excludes records
    # that are too short to exercise the frozen temporal head.
    eligible = [record for record in manifest["records"] if float(record["duration_s"]) >= 100.0]
    positive = [record for record in eligible if record["seizure_events"]]
    negative = [record for record in eligible if not record["seizure_events"]]
    rank = lambda record: hashlib.sha256(f"{RUN_ID}|checkpoint-smoke|{record['relative_edf']}".encode()).hexdigest()
    selected = [min(positive, key=rank), min(negative, key=rank)]
    record_data = {}
    for record in selected:
        path = Path(manifest["dataset_root"]) / record["relative_edf"]
        raw_uv, source_hz, channels = load_tuh_canonical_raw_uv(path)
        quality = tuh_quality_features(raw_uv, source_hz)
        n_windows = quality.shape[0]
        if n_windows < 40:
            raise RuntimeError(f"smoke record too short for 31-window context: {record['relative_edf']}")
        center = int(n_windows // 2)
        target_indices = [max(30, min(n_windows - 1, center + offset)) for offset in (-6, -4, -2, 0, 2, 4, 6, 8)]
        record_data[record["relative_edf"]] = (raw_uv, source_hz, channels, quality, target_indices)

    results: list[dict[str, object]] = []
    probabilities: list[dict[str, object]] = []
    for model in MODELS:
        started = time.monotonic()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        adapter, encoder, head, checkpoint, threshold = _model_objects(model)
        if adapter is not None:
            adapter = adapter.cuda().eval()
        encoder = encoder.cuda().eval()
        head = head.cuda().eval()
        for record in selected:
            raw_uv, source_hz, channels, quality, target_indices = record_data[record["relative_edf"]]
            target_times, probs, diagnostics = _infer_record(model, adapter, encoder, head, raw_uv, source_hz, quality, target_indices)
            finite = bool(np.isfinite(probs).all())
            in_range = bool(((probs >= 0) & (probs <= 1)).all())
            row = {
                "model": model,
                "fold": FOLD,
                "seed": SEED,
                "record_role": "seizure_present" if record["seizure_events"] else "background_only",
                "relative_edf": record["relative_edf"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_update": 5000,
                "threshold": threshold,
                "threshold_source": "standard CHB validation-only final step 5000",
                "prediction_windows": len(probs),
                "probability_min": float(probs.min()),
                "probability_max": float(probs.max()),
                "probability_mean": float(probs.mean()),
                "probability_std": float(probs.std()),
                "probabilities_finite": finite,
                "probabilities_in_unit_interval": in_range,
                "pass": finite and in_range and len(probs) == 8,
                **diagnostics,
            }
            results.append(row)
            for target_time, probability in zip(target_times, probs, strict=True):
                probabilities.append({"model": model, "fold": FOLD, "seed": SEED, "record_role": row["record_role"], "relative_edf": record["relative_edf"], "window_end_s": float(target_time), "probability": float(probability), "threshold": threshold})
        torch.cuda.synchronize()
        runtime = time.monotonic() - started
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        for row in results:
            if row["model"] == model:
                row["model_runtime_s"] = runtime
                row["peak_gpu_memory_gib"] = peak_gib
        del adapter, encoder, head
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "release_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "standard_tusz_frozen_checkpoint_smoke",
        "cohort_manifest_sha256": _sha256(MANIFEST_PATH),
        "fold": FOLD,
        "seed": SEED,
        "checkpoint_update": 5000,
        "model_training": False,
        "threshold_retuning": False,
        "external_performance_reported": False,
        "results": results,
        "all_pass": bool(all(row["pass"] for row in results) and len(results) == 6),
        "next_gate": "start standard external inference queue" if all(row["pass"] for row in results) else "repair frozen checkpoint adapter",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(probabilities).to_csv(PROBABILITY_OUTPUT, index=False)
    print(json.dumps({"all_pass": output["all_pass"], "result_count": len(results), "output": str(OUTPUT), "probabilities": str(PROBABILITY_OUTPUT), "next_gate": output["next_gate"]}, indent=2))


if __name__ == "__main__":
    main()
