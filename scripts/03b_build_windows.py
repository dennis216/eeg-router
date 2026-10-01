from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bfa.data.windows import build_window_index
from bfa.preprocessing.cache import cache_key


PREPROCESSING_CONFIG = {
    "channels": "canonical_bipolar_16_v1",
    "notch_hz": 60.0,
    "bandpass_hz": [0.5, 45.0],
    "window_s": 10.0,
    "stride_s": 2.0,
    "context_windows": 31,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, default=Path("manifests/recordings.parquet"))
    parser.add_argument("--seizures", type=Path, default=Path("manifests/seizures.parquet"))
    parser.add_argument("--output", type=Path, default=Path("manifests/windows.parquet"))
    args = parser.parse_args()
    recordings = pd.read_parquet(args.recordings).sort_values(
        ["patient_id", "recording_id"], kind="stable"
    )
    seizures = pd.read_parquet(args.seizures)
    frames = []
    for row in recordings.itertuples(index=False):
        recording_seizures = seizures[seizures.recording_id == row.recording_id]
        intervals = list(zip(recording_seizures.start_s, recording_seizures.end_s, strict=True))
        frame = build_window_index(
            row.patient_id,
            row.recording_id,
            row.duration_s,
            intervals,
            cache_key(row.sha256, PREPROCESSING_CONFIG),
        )
        frame["relative_path"] = row.relative_path
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        f"WINDOW_INDEX_OK rows={len(output)} train_eligible={int(output.train_eligible.sum())} "
        f"positive={int((output.label == 1).sum())}"
    )


if __name__ == "__main__":
    main()
