"""Run final-checkpoint inference on train windows for the frozen H4 fit.

The original pilot persisted validation and test probabilities only.  H4 needs a
strict train-fit/validation-lock/test-once sequence, so this script adds train
probabilities without touching checkpoints or test outputs.  It loads the final
5000-update checkpoint, runs the frozen train index, and writes one immutable
train probability file per existing compact run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "pilot" / "bsn_pilot_v1.yaml"
RUNS = (
    ("BSN-pilot-v1", 17, (17, 42, 3407)),
    ("BSN-pilot-v1-multisplit", 42, (17, 42)),
    ("v3-lite-2split", 42, (3407,)),
)
MODELS = ("singlem", "cbramod", "tcn_gat")
MAX_TRAIN_FIT_WINDOWS = 5_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runner():
    path = ROOT / "scripts" / "11_run_bsn_pilot.py"
    spec = importlib.util.spec_from_file_location("bsn_pilot_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train_fit_index(windows: pd.DataFrame, split: int, split_spec: dict[str, object]) -> pd.DataFrame:
    """Freeze a contiguous train-only recording subset for the compact H4 fit."""
    manifest_path = ROOT / "manifests" / f"h4_train_fit_index_split{split}.parquet"
    if manifest_path.exists():
        return pd.read_parquet(manifest_path)
    eligible = windows[windows.patient.isin(split_spec["train"]) & windows.train_eligible].copy()
    recording_rows = (
        eligible.groupby(["patient", "recording"], sort=True)
        .agg(rows=("recording", "size"), positive_windows=("label", "sum"))
        .reset_index()
    )
    recording_rows["positive_windows"] = recording_rows["positive_windows"].fillna(0).astype(int)
    by_patient: dict[str, list[dict[str, object]]] = {}
    for patient, group in recording_rows.groupby("patient", sort=True):
        ordered = group.sort_values(
            ["positive_windows", "recording"], ascending=[False, True], kind="stable"
        )
        by_patient[str(patient)] = ordered.to_dict(orient="records")
    selected: list[str] = []
    used = 0
    # Round-robin by patient prevents the compact fit from being dominated by a
    # single long recording.  Whole recordings preserve event continuity.
    while True:
        progress = False
        for patient in sorted(by_patient):
            if not by_patient[patient]:
                continue
            row = by_patient[patient][0]
            rows = int(row["rows"])
            if selected and used + rows > MAX_TRAIN_FIT_WINDOWS:
                by_patient[patient].pop(0)
                continue
            by_patient[patient].pop(0)
            selected.append(str(row["recording"]))
            used += rows
            progress = True
        if not progress:
            break
    subset = eligible[eligible.recording.astype(str).isin(selected)].reset_index(drop=True)
    if subset.empty:
        raise RuntimeError(f"empty H4 train-fit subset for split {split}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(manifest_path, index=False)
    manifest_json = manifest_path.with_suffix(".json")
    manifest_json.write_text(
        json.dumps(
            {
                "release_id": "v3-compact-extension-v1",
                "purpose": "H4 train-only router fit",
                "split": split,
                "selection": "whole recordings, positive-first within patient, round-robin patients, max 5000 windows",
                "rows": int(len(subset)),
                "patients": int(subset.patient.nunique()),
                "recordings": int(subset.recording.nunique()),
                "test_read": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return subset


def main() -> None:
    runner = _load_runner()
    config = yaml.safe_load(CONFIG.read_text())
    windows = pd.read_parquet(ROOT / config["cohort"]["window_manifest"])
    split_cache = {
        split: json.loads((ROOT / "manifests" / "splits" / f"split_{split}.json").read_text())
        for split in (17, 42)
    }
    training = config["training"]
    workers = max(8, int(training["dataloader_workers"]))
    for namespace, split, seeds in RUNS:
        split_spec = split_cache[split]
        train_index = _train_fit_index(windows, split, split_spec)
        for seed in seeds:
            for model_name in MODELS:
                run = ROOT / "runs" / namespace / model_name / f"split{split}_seed{seed}_main"
                checkpoint_path = run / "checkpoints" / "step_05000.pt"
                output_path = run / "train" / "probabilities_step_05000.parquet"
                manifest_path = run / "train" / "inference_manifest.json"
                if output_path.exists() and manifest_path.exists():
                    existing = json.loads(manifest_path.read_text())
                    if int(existing.get("rows", -1)) == len(train_index):
                        print(f"TRAIN_INFERENCE_EXISTS {namespace} split={split} seed={seed} model={model_name}", flush=True)
                        continue
                    print(f"TRAIN_INFERENCE_STALE_REPLACE {namespace} split={split} seed={seed} model={model_name} old_rows={existing.get('rows')} new_rows={len(train_index)}", flush=True)
                print(f"TRAIN_INFERENCE_START {namespace} split={split} seed={seed} model={model_name} rows={len(train_index)}", flush=True)
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                encoder = runner._model(model_name).cuda()
                head = runner.SharedContextHead().cuda()
                encoder.load_state_dict(state["encoder"], strict=False)
                head.load_state_dict(state["head"], strict=True)
                encoder.eval()
                head.eval()
                probabilities = runner._predict(
                    encoder,
                    head,
                    train_index,
                    model_name,
                    batch_size=64,
                    workers=workers,
                )
                table = runner._probability_table(train_index, probabilities)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                table.to_parquet(output_path, index=False)
                manifest = {
                    "release_id": "v3-compact-extension-v1",
                    "partition": "train",
                    "namespace": namespace,
                    "split": split,
                    "seed": seed,
                    "model": model_name,
                    "optimizer_update": 5000,
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "probability_sha256": _sha256(output_path),
                    "rows": int(len(table)),
                    "test_read": False,
                    "threshold_fit": False,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                del encoder, head, state, probabilities, table
                torch.cuda.empty_cache()
                print(f"TRAIN_INFERENCE_DONE {namespace} split={split} seed={seed} model={model_name} rows={len(train_index)}", flush=True)


if __name__ == "__main__":
    main()
