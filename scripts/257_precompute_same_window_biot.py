from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv


ROOT = Path("/root/b_false_alarm_atlas")
DATA_ROOT = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
LABEL_ROOT = ROOT / "outputs/reports/same-window-router-first-moe-v1/router_labels"
OUTPUT_ROOT = Path("/mnt/d/EEGData/same_window_router_first_moe_v1/biot_embeddings")
CHECKPOINT = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
MANIFEST_ROOT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests"


def manifest(partition: str) -> dict[str, object]:
    source = {"train": "train", "val": "dev", "test": "eval"}[partition]
    return json.loads((MANIFEST_ROOT / f"{source}_manifest.json").read_text())


def load_index(partition: str) -> pd.DataFrame:
    labels = LABEL_ROOT / f"{partition}.parquet"
    if labels.exists():
        frame = pd.read_parquet(labels)
        frame = frame[frame.policy_action_eligible.astype(bool)].copy()
    else:
        files = sorted((ROOT / f"outputs/reports/same-window-router-first-moe-v1/expert_probabilities/tusz/{partition}/singlem").rglob("*.parquet"))
        if not files: raise FileNotFoundError(labels)
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        frame = frame[["patient_id", "recording", "window_start_s", "window_end_s"]].copy()
    if any("future" in column.lower() for column in frame.columns): raise RuntimeError("future-policy columns are forbidden")
    frame = frame.sort_values(["recording", "window_start_s"], kind="stable").reset_index(drop=True)
    frame["embedding_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def load_encoder():
    sys.path.insert(0, str(ROOT / "third_party/BIOT")); from model.biot import BIOTEncoder
    model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100).cuda()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and not any(key.startswith("patch_embedding") for key in state): state = state["state_dict"]
    model.load_state_dict(state, strict=True); return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--partition", choices=("train", "val", "test"), required=True); parser.add_argument("--batch-size", type=int, default=64); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    frame = load_index(args.partition); cohort = manifest(args.partition); records = {str(x["relative_edf"]): x for x in cohort["records"]}; output = OUTPUT_ROOT / args.partition; output.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(output / "embeddings.float16.memmap", mode="w+", dtype=np.float16, shape=(len(frame), 256)); encoder = load_encoder(); started = time.monotonic(); done = 0
    for recording, positions in frame.groupby("recording", sort=False).indices.items():
        record = records[str(recording)]; raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(Path(cohort.get("dataset_root", DATA_ROOT)) / str(recording)); ratio = 200.0 / source_hz
        from fractions import Fraction
        fraction = Fraction(ratio).limit_denominator(10000); signal = resample_poly(raw_uv, fraction.numerator, fraction.denominator, axis=-1).astype(np.float32)
        index = np.asarray(positions, dtype=np.int64); starts = np.rint(frame.iloc[index].window_start_s.to_numpy(float) * 200).astype(np.int64)
        for offset in range(0, len(index), args.batch_size):
            ids = index[offset : offset + args.batch_size]; sample_starts = starts[offset : offset + args.batch_size]
            windows = np.stack([signal[:, start : start + 2000] for start in sample_starts]).astype(np.float32)
            if windows.shape[1:] != (16, 2000): raise RuntimeError(f"bad BIOT window in {recording}")
            q95 = np.quantile(np.abs(windows), 0.95, axis=-1, keepdims=True); windows /= q95 + 1e-8
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): embedded = encoder(torch.from_numpy(np.ascontiguousarray(windows)).cuda(non_blocking=True))
            mmap[frame.iloc[ids].embedding_row_id.to_numpy(np.int64)] = embedded.float().cpu().numpy().astype(np.float16); done += len(ids)
        print(json.dumps({"partition": args.partition, "records_done": int(frame.iloc[: max(index)+1].recording.nunique()), "rows_done": done, "rows_total": len(frame)}), flush=True)
    mmap.flush(); frame.to_parquet(output / "index.parquet", index=False); (output / "manifest.json").write_text(json.dumps({"status": "PASS", "partition": args.partition, "rows": len(frame), "embedding_dim": 256, "dtype": "float16", "single_window": True, "elapsed_seconds": time.monotonic() - started}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
