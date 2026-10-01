"""Run a two-record, three-model frozen TUSZ inference smoke.

The cohort and adapter are written before any model is loaded. This is a pipeline
test only: it does not report external performance or select a threshold.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.evaluation.eventize import eventize
from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.models.precomputed import PrecomputedProjectionEncoder
from bfa.models.shared_head import SharedContextHead
from bfa.models.singlem_adapter import SingLEMAdapter
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.preprocessing.channels import CHANNELS
from bfa.preprocessing.tuh import (
    LEGACY_TO_MODERN,
    MODEL_VIEW_CONFIG,
    load_tuh_canonical_raw_uv,
    tuh_model_view,
    tuh_quality_features,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports" / "v3-compact-extension-v1"
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
MODELS = ("singlem", "cbramod", "tcn_gat")
RUNS = {
    model: ROOT / "runs" / "BSN-pilot-v1" / model / "split17_seed17_main"
    for model in MODELS
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_bi(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def freeze_cohort(format_smoke: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = [r for r in format_smoke["records"] if r["partition"] == "eval"]  # type: ignore[index]
    enriched = []
    for row in candidates:
        edf = DATA_ROOT / str(row["relative_edf"])
        csv_bi = Path(str(edf.with_suffix(".csv")) + "_bi")
        annotations = parse_csv_bi(csv_bi)
        seizures = [a for a in annotations if a["label"].lower() == "seiz"]
        enriched.append({**row, "edf_path": edf, "csv_bi_path": csv_bi, "seizures": seizures})
    positive_pool = [r for r in enriched if r["seizures"]]
    negative_pool = [r for r in enriched if not r["seizures"]]
    if not positive_pool or not negative_pool:
        raise RuntimeError("frozen format-smoke eval pool lacks positive or background record")
    rank = lambda r: hashlib.sha256(f"v3-compact-extension-v1|{r['relative_edf']}".encode()).hexdigest()
    positive = min(positive_pool, key=rank)
    negative = min(negative_pool, key=rank)
    selected = []
    for role, row in (("seizure_present", positive), ("background_only", negative)):
        duration = float(row["duration_s"])
        if role == "seizure_present":
            anchor = float(row["seizures"][0]["start_time"])
            anchor_source = "first csv_bi seizure onset"
        else:
            low, high = 90.0, max(90.0, duration - 40.0)
            digest = int(hashlib.sha256(str(row["relative_edf"]).encode()).hexdigest()[:12], 16)
            anchor = low + (digest / float(16**12 - 1)) * (high - low)
            anchor_source = "SHA256-mapped time within eligible background interval"
        n_windows = int(np.floor((duration - 10.0) / 2.0)) + 1
        center = int(round(anchor / 2.0))
        first = min(max(30, center - 4), n_windows - 8)
        target_indices = list(range(first, first + 8))
        selected.append({
            "role": role,
            "patient_id": row["patient_id"],
            "relative_edf": row["relative_edf"],
            "edf_path": row["edf_path"],
            "csv_bi_path": row["csv_bi_path"],
            "edf_sha256": row["edf_sha256"],
            "csv_bi_sha256": sha256(row["csv_bi_path"]),
            "duration_s": duration,
            "source_sampling_hz": float(row["sfreq_hz"]),
            "anchor_s": anchor,
            "anchor_source": anchor_source,
            "target_window_indices": target_indices,
            "target_window_starts_s": [2.0 * i for i in target_indices],
            "annotation_labels": row["annotations"]["csv_bi"]["labels"],
        })
    cohort = {
        "release_id": "v3-compact-extension-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "TUSZ_v2.0.6 eval",
        "purpose": "two-record frozen-model inference smoke; not external performance estimation",
        "selection_pool": "three eval patients already frozen by tuh_real_format_smoke.json",
        "selection_rule": "lowest SHA256-ranked seizure-present and background-only record; anchors use annotation or path hash, never model output",
        "selection_depended_on_model_output": False,
        "records": [{k: v for k, v in r.items() if k not in {"edf_path", "csv_bi_path"}} for r in selected],
    }
    return cohort, selected


def freeze_adapter() -> dict[str, object]:
    return {
        "release_id": "v3-compact-extension-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": "tuh-reference-to-frozen-16-bipolar-v1",
        "channel_order": list(CHANNELS),
        "legacy_to_modern": LEGACY_TO_MODERN,
        "derivation": "left reference electrode minus right reference electrode; MNE volts converted to microvolts before model views",
        "powerline_notch_hz": 60.0,
        "model_views": MODEL_VIEW_CONFIG,
        "singlem_hard_clipping": False,
        "singlem_out_of_range_policy": "retain samples and report fraction outside [-1.5, 1.5] as QC",
        "quality_features": ["line_noise_ratio", "clipping_fraction", "flat_fraction"],
        "context_windows": 31,
        "window_s": 10.0,
        "stride_s": 2.0,
        "model_training": False,
        "threshold_retuning": False,
    }


def model_objects(model: str):
    checkpoint = RUNS[model] / "checkpoints" / "step_05000.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["update"]) != 5000:
        raise ValueError(f"unexpected {model} checkpoint update")
    adapter = None
    if model == "singlem":
        adapter = SingLEMAdapter(ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt")
        encoder = PrecomputedProjectionEncoder(16, adapter.projection)
    elif model == "cbramod":
        adapter = CBraModAdapter(ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
        encoder = PrecomputedProjectionEncoder(200, adapter.projection)
    else:
        encoder = MultiScaleTCNGAT()
        encoder.native_head.requires_grad_(False)
    encoder.load_state_dict(state["encoder"], strict=True)
    head = SharedContextHead()
    head.load_state_dict(state["head"], strict=True)
    completed = json.loads((RUNS[model] / "completed.json").read_text())
    threshold = float(completed["final_validation"]["selected_event_operating_point"]["threshold"])
    return adapter, encoder, head, checkpoint, threshold


def window_view(signal: np.ndarray, rate: int) -> np.ndarray:
    samples = 10 * rate
    return np.lib.stride_tricks.sliding_window_view(signal, samples, axis=-1)[:, :: 2 * rate]


@torch.inference_mode()
def infer_record(model: str, adapter, encoder, head, raw_uv, source_hz, quality, target_indices):
    view = tuh_model_view(raw_uv, source_hz, model)
    rate = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
    windows = window_view(view, rate)
    if windows.shape[1] != quality.shape[0]:
        difference = abs(windows.shape[1] - quality.shape[0])
        if difference > 1:
            raise ValueError(f"{model} window/quality mismatch {windows.shape[1]} vs {quality.shape[0]}")
        limit = min(windows.shape[1], quality.shape[0])
        windows, quality = windows[:, :limit], quality[:limit]
    if max(target_indices) >= windows.shape[1] or min(target_indices) < 30:
        raise ValueError("target context index out of bounds")

    target_times = np.array(target_indices, dtype=float) * 2.0 + 10.0
    if model in {"singlem", "cbramod"}:
        needed = list(range(min(target_indices) - 30, max(target_indices) + 1))
        batch_size = 32 if model == "singlem" else 8
        feature_chunks = []
        for offset in range(0, len(needed), batch_size):
            ids = needed[offset : offset + batch_size]
            batch = np.ascontiguousarray(windows[:, ids].transpose(1, 0, 2)).reshape(len(ids), 16, 10, rate)
            inputs = torch.from_numpy(batch).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature_chunks.append(adapter.backbone_features(inputs).float().cpu().numpy())
        features = np.concatenate(feature_chunks)
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
    probabilities = np.concatenate(probabilities)
    singlem_outside = float(np.mean(np.abs(view) > 1.5)) if model == "singlem" else None
    return target_times, probabilities, {
        "source_shape": list(raw_uv.shape),
        "model_view_shape": list(view.shape),
        "quality_shape": list(quality.shape),
        "singlem_fraction_outside_minus1_5_plus1_5": singlem_outside,
        "model_view_finite": bool(np.isfinite(view).all()),
    }


def main() -> None:
    format_path = REPORT_DIR / "tuh_real_format_smoke.json"
    format_smoke = json.loads(format_path.read_text())
    cohort, selected = freeze_cohort(format_smoke)
    adapter_manifest = freeze_adapter()
    cohort_path = REPORT_DIR / "tuh_inference_smoke_cohort.json"
    adapter_path = REPORT_DIR / "tuh_adapter_manifest.json"
    cohort_path.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    adapter_path.write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True) + "\n")

    record_data = {}
    for record in selected:
        raw_uv, source_hz, source_channels = load_tuh_canonical_raw_uv(record["edf_path"])
        quality = tuh_quality_features(raw_uv, source_hz)
        record_data[record["relative_edf"]] = (raw_uv, source_hz, source_channels, quality)

    all_results = []
    probability_rows = []
    gpu = torch.cuda.get_device_properties(0)
    for model in MODELS:
        started = time.monotonic()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        adapter, encoder, head, checkpoint, threshold = model_objects(model)
        if adapter is not None:
            adapter = adapter.cuda().eval()
        encoder = encoder.cuda().eval()
        head = head.cuda().eval()
        for record in selected:
            raw_uv, source_hz, source_channels, quality = record_data[record["relative_edf"]]
            target_times, probabilities, diagnostics = infer_record(
                model, adapter, encoder, head, raw_uv, source_hz, quality, record["target_window_indices"]
            )
            finite = bool(np.isfinite(probabilities).all())
            in_range = bool(((probabilities >= 0) & (probabilities <= 1)).all())
            warnings = []
            if float(np.std(probabilities)) < 1e-6:
                warnings.append("near_constant_probabilities")
            if float(np.mean(probabilities)) < 1e-5 or float(np.mean(probabilities)) > 1 - 1e-5:
                warnings.append("probabilities_near_boundary")
            if bool(np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0)):
                warnings.append("at_least_one_probability_exactly_saturated")
            smoke_events = eventize(target_times, probabilities, threshold=threshold)
            result = {
                "model": model,
                "record_role": record["role"],
                "relative_edf": record["relative_edf"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_update": 5000,
                "threshold": threshold,
                "threshold_source": "CHB validation-only split17 seed17 final step 5000",
                "prediction_windows": len(probabilities),
                "probability_min": float(np.min(probabilities)),
                "probability_max": float(np.max(probabilities)),
                "probability_mean": float(np.mean(probabilities)),
                "probability_std": float(np.std(probabilities)),
                "probabilities_finite": finite,
                "probabilities_in_unit_interval": in_range,
                "smoke_event_count": len(smoke_events),
                "smoke_events_not_scored": True,
                "warnings": warnings,
                "pass": finite and in_range and len(probabilities) == 8,
                **diagnostics,
            }
            all_results.append(result)
            for t, prob in zip(target_times, probabilities, strict=True):
                probability_rows.append({
                    "model": model, "record_role": record["role"], "relative_edf": record["relative_edf"],
                    "window_end_s": float(t), "probability": float(prob), "threshold": threshold,
                })
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        for result in all_results:
            if result["model"] == model:
                result["model_total_runtime_s"] = elapsed
                result["model_peak_gpu_memory_gib"] = peak_gib
        del adapter, encoder, head
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "release_id": "v3-compact-extension-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "two-record_three-model_frozen_inference_smoke",
        "dataset": "TUSZ_v2.0.6 eval",
        "cohort_sha256": sha256(cohort_path),
        "adapter_manifest_sha256": sha256(adapter_path),
        "gpu": {"name": gpu.name, "total_memory_gib": gpu.total_memory / (1024 ** 3)},
        "model_training": False,
        "threshold_retuning": False,
        "external_performance_reported": False,
        "results": all_results,
        "all_pass": all(r["pass"] for r in all_results) and len(all_results) == 6,
        "result_count": len(all_results),
        "next_gate": "freeze eval patient cohort and run external subset validation" if all(r["pass"] for r in all_results) else "repair inference adapter before external validation",
    }
    output_path = REPORT_DIR / "tuh_frozen_inference_smoke.json"
    probabilities_path = REPORT_DIR / "tuh_frozen_inference_smoke_probabilities.csv"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(probability_rows).to_csv(probabilities_path, index=False)
    print(json.dumps({
        "all_pass": output["all_pass"], "result_count": len(all_results),
        "cohort": str(cohort_path), "adapter": str(adapter_path), "output": str(output_path),
        "probabilities": str(probabilities_path), "next_gate": output["next_gate"],
        "models": [{"model": m, "runtime_s": next(r["model_total_runtime_s"] for r in all_results if r["model"] == m),
                    "peak_gpu_gib": next(r["model_peak_gpu_memory_gib"] for r in all_results if r["model"] == m)} for m in MODELS],
    }, indent=2))


if __name__ == "__main__":
    main()
