from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from bfa.models.shared_head import SharedContextHead
from bfa.models.tcn_gat import MultiScaleTCNGAT
from bfa.training.dataset import CachedContextDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    windows = pd.read_parquet("manifests/windows.parquet")
    index = windows[
        (windows.recording == "chb01_01.edf") & windows.train_eligible
    ].head(args.batch_size)
    dataset = CachedContextDataset(index, Path("/mnt/d/EEGData/bfa_cache_v3"), "tcn_gat")
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
    encoder = MultiScaleTCNGAT().cuda()
    head = SharedContextHead().cuda()
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=3e-4, weight_decay=1e-4
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = encoder.forward_window_sequence(
            batch["x"].cuda(non_blocking=True), batch["quality"].cuda(non_blocking=True)
        )
        logits = head(embeddings)
        loss = functional.binary_cross_entropy_with_logits(logits, batch["y"].cuda())
    loss.backward()
    torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    print(
        f"REAL_BATCH_OK batch={args.batch_size} loss={float(loss):.6f} "
        f"seconds={time.monotonic() - started:.3f} "
        f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}"
    )


if __name__ == "__main__":
    main()
