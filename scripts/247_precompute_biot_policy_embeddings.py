"""Precompute frozen BIOT/EEG-PREST embeddings for four-action policy rows.

The policy head sees only a 256-dimensional frozen BIOT embedding.  This
script materializes those embeddings once so the five-fold policy training is
fast and its training/validation curves reflect the head rather than repeated
STFT work.  MIT is already cached at 200 Hz.  TUSZ source contexts are 128 Hz
60-second windows; the last causal 10 seconds are resampled to 200 Hz with
the same anti-aliased ``resample_poly(25, 16)`` contract used by the original
BIOT plan.
"""

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
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/root/b_false_alarm_atlas")
MIT_LABELS = ROOT / "outputs/reports/h6-causal-policy-router-v3-four-action-mit/labels/mit_four_action_policy.parquet"
TUSZ_LABEL_DIR = ROOT / "outputs/reports/h6-causal-policy-router-v3-four-action/labels"
MIT_INPUT_ROOT = Path("/mnt/d/EEGData/bfa_biot_prest_input_v1")
TUSZ_CACHE_ROOT = Path("/mnt/d/TUH_EEG/h6_tusz_raw_router_cache")
OUTPUT_ROOT = Path("/mnt/d/EEGData/bfa_biot_policy_embeddings_v1")
CHECKPOINT = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
SEED = 3407


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(dataset: str) -> pd.DataFrame:
    if dataset == "mit":
        frame = pd.read_parquet(MIT_LABELS)
        frame["source_table"] = str(MIT_LABELS.relative_to(ROOT))
        frame["source_row_id"] = np.arange(len(frame), dtype=np.int64)
        frame["source_path"] = frame["biot_input_relative_path"].astype(str)
    elif dataset == "tusz":
        parts = []
        for partition in ("train", "dev"):
            path = TUSZ_LABEL_DIR / f"{partition}_four_action_policy.parquet"
            part = pd.read_parquet(path)
            part["source_table"] = str(path.relative_to(ROOT))
            part["source_row_id"] = np.arange(len(part), dtype=np.int64)
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        frame["source_path"] = frame["source_cache_path_v1"].astype(str)
    else:
        raise ValueError(dataset)

    eligible = frame.policy_action_eligible.astype(bool)
    frame = frame.loc[eligible].copy()
    if frame.empty:
        raise RuntimeError(f"no eligible policy rows for {dataset}")
    if any(column.startswith("risk") for column in frame.columns):
        raise RuntimeError("risk field found in policy input")
    if dataset == "mit":
        sort_columns = ["source_path", "context_start_s", "recording"]
    else:
        sort_columns = ["source_path", "decision_index_v1", "recording"]
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    frame["embedding_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


class PolicyWindowDataset(Dataset[dict[str, object]]):
    def __init__(self, frame: pd.DataFrame, dataset: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset = dataset
        self._current_path: str | None = None
        self._current_array: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def _load_mit(self, row: pd.Series) -> np.ndarray:
        relative = str(row.biot_input_relative_path)
        if relative != self._current_path:
            path = MIT_INPUT_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            self._current_array = np.load(path, mmap_mode="r", allow_pickle=False)
            self._current_path = relative
        assert self._current_array is not None
        start = int(round(float(row.context_start_s) * 200.0))
        window = np.asarray(self._current_array[:, start : start + 2000], dtype=np.float32)
        if window.shape != (16, 2000):
            raise ValueError(f"bad MIT BIOT window {relative} start={start}: {window.shape}")
        return window

    def _load_tusz(self, row: pd.Series) -> np.ndarray:
        relative = str(row.source_cache_path_v1)
        if relative != self._current_path:
            path = TUSZ_CACHE_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive = np.load(path, allow_pickle=False)
            self._current_array = np.asarray(archive["x"])
            archive.close()
            self._current_path = relative
        assert self._current_array is not None
        cache_row = int(row.cache_row if "cache_row" in row.index else row.decision_index_v1)
        if self._current_array.ndim != 3:
            raise ValueError(f"bad TUSZ cache rank: {self._current_array.shape}")
        context = np.asarray(self._current_array[cache_row, :, -1280:], dtype=np.float32)
        if context.shape != (16, 1280):
            raise ValueError(f"bad TUSZ causal context {relative} row={cache_row}: {context.shape}")
        return np.asarray(resample_poly(context, up=25, down=16, axis=-1), dtype=np.float32)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.frame.iloc[item]
        window = self._load_mit(row) if self.dataset == "mit" else self._load_tusz(row)
        q95 = np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True)
        window = window / (q95 + 1e-8)
        return {"x": torch.from_numpy(np.ascontiguousarray(window)), "embedding_row_id": int(row.embedding_row_id)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mit", "tusz"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BIOT embedding extraction")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = load_rows(args.dataset)
    output_dir = OUTPUT_ROOT / args.dataset
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text())
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    index_path = output_dir / "index.parquet"
    embedding_path = output_dir / "embeddings.float16.memmap"

    dataset = PolicyWindowDataset(rows, args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    from model.biot import BIOTEncoder

    model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100).cuda()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and not any(key.startswith("patch_embedding") for key in state):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    output = np.memmap(embedding_path, mode="w+", dtype=np.float16, shape=(len(rows), 256))
    started = time.monotonic()
    total_batches = len(loader)
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            inputs = batch["x"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(inputs)
            ids = batch["embedding_row_id"].numpy().astype(np.int64, copy=False)
            output[ids] = embeddings.float().cpu().numpy().astype(np.float16)
            if batch_number % args.progress_every == 0 or batch_number == total_batches:
                payload = {"stage": "embedding", "dataset": args.dataset, "completed": min(batch_number * args.batch_size, len(rows)), "total": len(rows), "batches": batch_number, "elapsed_seconds": time.monotonic() - started}
                progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                print("BIOT_POLICY_EMBED_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)
    output.flush()
    rows.to_parquet(index_path, index=False)
    manifest = {
        "release_id": "h6-causal-policy-router-v3-four-action",
        "dataset": args.dataset,
        "status": "PASS",
        "seed": SEED,
        "rows": int(len(rows)),
        "embedding_dim": 256,
        "dtype": "float16",
        "memmap": str(embedding_path),
        "index": str(index_path),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "normalization": "per_channel_abs_95th_quantile",
        "context_contract": "MIT 200 Hz 10 s window; TUSZ last 10 s of causal 60 s context resampled 128->200 Hz",
        "policy_rows_only": True,
        "risk_head": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    progress_path.write_text(json.dumps({"stage": "complete", **manifest}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "third_party/BIOT"))
    main()
