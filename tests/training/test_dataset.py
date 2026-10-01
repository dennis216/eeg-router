from pathlib import Path

import numpy as np
import pandas as pd

from bfa.training.dataset import CachedContextDataset


def test_cached_dataset_returns_causal_context(tmp_path: Path) -> None:
    relative = Path("p1/r1.edf")
    signal_path = tmp_path / "tcn_gat" / relative.with_suffix(".npy")
    quality_path = tmp_path / "quality" / relative.with_suffix(".npy")
    signal_path.parent.mkdir(parents=True)
    quality_path.parent.mkdir(parents=True)
    signal = np.broadcast_to(np.arange(70 * 256), (16, 70 * 256)).astype(np.float32)
    quality = np.zeros((31, 16, 3), dtype=np.float32)
    np.save(signal_path, signal)
    np.save(quality_path, quality)
    index = pd.DataFrame(
        [{"relative_path": str(relative), "start": 60.0, "label": 1}]
    )

    sample = CachedContextDataset(index, tmp_path, "tcn_gat")[0]
    assert sample["x"].shape == (31, 16, 2560)
    assert sample["quality"].shape == (31, 16, 3)
    assert sample["x"][0, 0, 0].item() == 0
    assert sample["x"][-1, 0, 0].item() == 60 * 256
    assert sample["y"].item() == 1


def test_quality_can_come_from_a_separate_frozen_cache(tmp_path: Path) -> None:
    signal_root = tmp_path / "model"
    quality_root = tmp_path / "quality-cache"
    relative = Path("p1/r1.edf")
    (signal_root / "tcn_gat" / relative.parent).mkdir(parents=True)
    (quality_root / "quality" / relative.parent).mkdir(parents=True)
    np.save(signal_root / "tcn_gat" / relative.with_suffix(".npy"), np.zeros((16, 17920)))
    np.save(quality_root / "quality" / relative.with_suffix(".npy"), np.ones((31, 16, 3)))
    index = pd.DataFrame([{"relative_path": str(relative), "start": 60.0, "label": 0}])
    sample = CachedContextDataset(
        index, signal_root, "tcn_gat", quality_root=quality_root
    )[0]
    assert sample["quality"].sum().item() == 31 * 16 * 3
