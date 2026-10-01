from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    script = ROOT / "scripts/11_run_bsn_pilot.py"
    spec = importlib.util.spec_from_file_location("bfa_pilot_runner", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch.manual_seed(17)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    encoder = runner._model("singlem").cuda().eval()
    head = runner.SharedContextHead().cuda().eval()
    checkpoint = torch.load(
        ROOT
        / "runs/BSN-pilot-v1/singlem/split17_seed17_main/checkpoints/step_01000.pt",
        map_location="cpu",
        weights_only=False,
    )
    incompatible = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("encoder.")]
    if incompatible.missing_keys or unexpected:
        raise ValueError(f"checkpoint mismatch: {incompatible}, unexpected={unexpected}")
    head.load_state_dict(checkpoint["head"], strict=True)
    split = json.loads((ROOT / "manifests/splits/split_17.json").read_text())
    windows = pd.read_parquet(ROOT / "manifests/windows.parquet")
    validation_index = windows[
        windows.patient.isin(split["validation"]) & ~windows.warmup
    ].reset_index(drop=True)
    metrics = runner._validate(
        encoder,
        head,
        validation_index,
        "singlem",
        ROOT / "runs/BSN-pilot-v1/singlem/split17_seed17_main",
        1000,
        pd.read_parquet(ROOT / "manifests/seizures.parquet"),
        pd.read_parquet(ROOT / "manifests/recordings.parquet"),
        4,
    )
    print("VALIDATION_BACKFILL_COMPLETE " + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
