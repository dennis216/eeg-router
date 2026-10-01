from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "chb-mit-biot-router-minimal-v1"
sys.path.insert(0, str(ROOT / "third_party/BIOT"))
from model.biot import BIOTEncoder  # noqa: E402


class BIOTWindowDataset(Dataset[dict[str, object]]):
    def __init__(self, index: pd.DataFrame, cache_root: Path) -> None:
        self.index = index.reset_index(drop=True)
        self.cache_root = cache_root
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.index.iloc[item]
        relative = str(row.relative_path)
        if relative not in self._arrays:
            self._arrays[relative] = np.load(
                self.cache_root / Path(relative).with_suffix(".npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
        signal = self._arrays[relative]
        start = int(round(float(row.start) * 200.0))
        window = np.asarray(signal[:, start : start + 2000], dtype=np.float32)
        if window.shape != (16, 2000):
            raise ValueError(f"invalid cached BIOT window {relative} start={row.start}: {window.shape}")
        q95 = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
        window = window / (q95 + 1e-8)
        return {"x": torch.from_numpy(np.ascontiguousarray(window)), "row_id": int(row.window_row_id)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BIOT embedding precomputation requires CUDA")
    windows = pd.read_parquet(ROOT / "manifests/windows.parquet").loc[lambda x: ~x.warmup].reset_index(drop=True)
    # Use a contiguous ID for the filtered embedding table; original manifest rows include warmup gaps.
    windows["window_row_id"] = np.arange(len(windows), dtype=np.int64)
    cache_root = Path("/mnt/d/EEGData/bfa_biot_prest_input_v1")
    out_root = Path("/mnt/d/EEGData/bfa_biot_prest_embeddings_v1")
    out_root.mkdir(parents=True, exist_ok=True)
    progress = ROOT / "outputs/reports" / NAMESPACE / "biot_embedding_progress.json"
    manifest_path = ROOT / "outputs/reports" / NAMESPACE / "biot_embedding_manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text())
        return
    shape = (len(windows), 256)
    embedding_path = out_root / "embeddings.float16.memmap"
    dataset = BIOTWindowDataset(windows, cache_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    checkpoint = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
    model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    output = np.memmap(embedding_path, mode="w+", dtype=np.float16, shape=shape)
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            inputs = batch["x"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(inputs)
            ids = batch["row_id"].numpy().astype(np.int64, copy=False)
            output[ids] = embeddings.float().cpu().numpy().astype(np.float16)
            if batch_number % 50 == 0 or batch_number == len(loader):
                completed = min(batch_number * args.batch_size, len(dataset))
                payload = {"stage": "embedding", "completed": completed, "total": len(dataset), "batches": batch_number, "elapsed_seconds": time.monotonic() - started}
                progress.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                print("BIOT_EMBED_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)
    output.flush()
    windows[["window_row_id", "patient", "recording", "start", "end", "label", "relative_path"]].to_parquet(out_root / "embedding_index.parquet", index=False)
    payload = {
        "protocol_id": NAMESPACE,
        "status": "PASS",
        "rows": len(windows),
        "embedding_dim": 256,
        "dtype": "float16",
        "memmap": str(embedding_path),
        "embedding_index": str(out_root / "embedding_index.parquet"),
        "window_manifest_sha256": sha256_file(ROOT / "manifests/windows.parquet"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cache_root": str(cache_root),
        "normalization": "per_channel_abs_95th_quantile",
        "test_used_for_fitting": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
