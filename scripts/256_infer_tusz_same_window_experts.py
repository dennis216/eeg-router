from __future__ import annotations

import argparse
import gc
import json
import os
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
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
CHECKPOINT_ROOT = ROOT / "outputs/reports/same-window-router-first-moe-v1/mit_experts/formal"
OUTPUT_ROOT = ROOT / "outputs/reports/same-window-router-first-moe-v1/expert_probabilities/tusz"
MANIFEST_ROOT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests"


def checkpoint_path(model: str) -> Path:
    default = CHECKPOINT_ROOT / model / "best.pt"
    override = os.environ.get(f"BFA_{model.upper()}_CHECKPOINT")
    return Path(override) if override else default


def load_model(model: str):
    state = torch.load(checkpoint_path(model), map_location="cpu", weights_only=False)
    adapter = None
    if model == "singlem":
        adapter = SingLEMAdapter(ROOT / "third_party/SingLEM/SingLEM/checkpoints/singlem_downstream_excluded.pt")
        encoder = PrecomputedProjectionEncoder(16, adapter.projection)
    elif model == "cbramod":
        adapter = CBraModAdapter(ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
        encoder = PrecomputedProjectionEncoder(200, adapter.projection)
    else:
        encoder = MultiScaleTCNGAT(); encoder.native_head.requires_grad_(False)
    encoder.load_state_dict(state["encoder"], strict=True)
    head = SharedContextHead(); head.load_state_dict(state["head"], strict=True)
    if adapter is not None: adapter = adapter.cuda().eval()
    return adapter, encoder.cuda().eval(), head.cuda().eval()


def windows(signal: np.ndarray, rate: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(signal, 10 * rate, axis=-1)[:, :: 2 * rate]


def label_windows(starts: np.ndarray, seizure_events: list[object]) -> np.ndarray:
    centers = starts + 5.0
    seizures = []
    for item in seizure_events:
        seizures.append((float(item.get("start_s", item.get("start", 0))), float(item.get("end_s", item.get("end", 0)))) if isinstance(item, dict) else (float(item[0]), float(item[1])))
    label = np.full(len(starts), np.nan, dtype=np.float32)
    if not seizures:
        label[:] = 0.0; return label
    for i, center in enumerate(centers):
        if any(a <= center <= b for a, b in seizures):
            label[i] = 1.0
        elif min(min(abs(center - a), abs(center - b)) for a, b in seizures) >= 30.0:
            label[i] = 0.0
    return label


@torch.inference_mode()
def predict_record(model: str, adapter, encoder, head, raw_uv: np.ndarray, source_hz: float, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    view = tuh_model_view(raw_uv, source_hz, model); rate = int(MODEL_VIEW_CONFIG[model]["sampling_hz"])
    if view.shape[-1] < 10 * rate:
        return np.empty(0, dtype=float), np.empty(0, dtype=np.float32)
    window_array = windows(view, rate); quality = tuh_quality_features(raw_uv, source_hz)
    limit = min(window_array.shape[1], len(quality)); window_array, quality = window_array[:, :limit], quality[:limit]
    batch_size = int(batch_size or {"singlem": 256, "cbramod": 16, "tcn_gat": 64}[model]); probabilities = []
    for start in range(0, limit, batch_size):
        stop = min(limit, start + batch_size); batch = np.ascontiguousarray(window_array[:, start:stop].transpose(1, 0, 2)); x = torch.from_numpy(batch).cuda(non_blocking=True); q = torch.from_numpy(np.ascontiguousarray(quality[start:stop])).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if model in {"singlem", "cbramod"}:
                frozen = adapter.backbone_features(x.reshape(len(batch), 16, 10, rate)); embedding = encoder.forward_window(frozen)
            else:
                embedding = encoder.forward_window(x, q)
            probability = torch.sigmoid(head(embedding[:, None]))
        probabilities.append(probability.float().cpu().numpy())
    starts = np.arange(limit, dtype=float) * 2.0
    return starts, np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=("singlem", "cbramod", "tcn_gat"), required=True); parser.add_argument("--partition", choices=("train", "val", "test"), required=True); parser.add_argument("--limit-records", type=int); parser.add_argument("--batch-size", type=int); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    source = {"train": "train", "val": "dev", "test": "eval"}[args.partition]
    manifest = json.loads((MANIFEST_ROOT / f"{source}_manifest.json").read_text()); records = manifest["records"][: args.limit_records]
    adapter, encoder, head = load_model(args.model); output = OUTPUT_ROOT / args.partition / args.model; output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for record in records:
        relative = Path(str(record["relative_edf"])); target = output / relative.with_suffix(".parquet")
        if target.exists(): completed += 1; continue
        raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(Path(manifest.get("dataset_root", DATA_ROOT)) / relative)
        starts, probability = predict_record(args.model, adapter, encoder, head, raw_uv, source_hz, args.batch_size); label = label_windows(starts, record.get("seizure_events", []))
        table = pd.DataFrame({"patient_id": str(record["patient_id"]), "recording": relative.as_posix(), "window_start_s": starts, "window_end_s": starts + 10.0, "label": label, "probability": probability})
        target.parent.mkdir(parents=True, exist_ok=True); table.to_parquet(target, index=False); completed += 1
        print(json.dumps({"partition": args.partition, "model": args.model, "record": completed, "total": len(records), "windows": len(table)}), flush=True)
        del raw_uv, table, probability; gc.collect()
    (output / "completed.json").write_text(json.dumps({"status": "PASS", "partition": args.partition, "model": args.model, "records": completed, "single_window": True, "stride_seconds": 2}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
