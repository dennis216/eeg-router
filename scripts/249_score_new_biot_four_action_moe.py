"""Score the current four-action BIOT policy router at event level.

This evaluator is deliberately downstream of the completed four-action run:

* MIT uses the fold-specific outer-test predictions from the new policy-head
  run and the seed-3407 OOF expert probabilities.
* TUSZ uses a uniform ensemble of the five new policy heads on the locked
  official Eval cohort.  Expert thresholds remain those selected before Eval.
* Reject emits no expert alarm.  There is no risk head or risk threshold.

The script writes a new ``final_event_scoring`` report and never changes the
previous H6 scoring artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

ROOT = Path("/root/b_false_alarm_atlas")
OUTPUT_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v3-four-action-biot-kfold"
EMBED_ROOT = Path("/mnt/d/EEGData/bfa_biot_policy_embeddings_v1")
TUSZ_CACHE_ROOT = Path("/mnt/d/TUH_EEG/h6_tusz_raw_router_cache")
BIOT_CHECKPOINT = ROOT / "third_party/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
TUSZ_OLD_ROOT = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1"
TUSZ_EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1"
MIT_PROBABILITIES = ROOT / "outputs/reports/v3-groupkfold-confirmatory-v1/oof_probabilities_seed.parquet"
MIT_RECORDINGS = ROOT / "manifests/recordings.parquet"
MIT_SEIZURES = ROOT / "manifests/seizures.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
ACTIONS = MODELS + ("reject",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_router_class():
    path = ROOT / "scripts/248_train_biot_four_action_policy_kfold.py"
    spec = importlib.util.spec_from_file_location("new_policy_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RouterHead


def load_heads() -> tuple[type, list[torch.nn.Module], torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for new BIOT policy inference")
    router_class = load_router_class()
    device = torch.device("cuda")
    heads: list[torch.nn.Module] = []
    for fold in range(5):
        checkpoint_path = OUTPUT_ROOT / "tusz" / f"fold_{fold}" / "best_policy_head.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = router_class(
            hidden=int(checkpoint["config"].get("hidden", 64)),
            dropout=float(checkpoint["config"].get("dropout", 0.2)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        heads.append(model)
    return router_class, heads, device


def load_biot_encoder() -> torch.nn.Module:
    sys.path.insert(0, str(ROOT / "third_party/BIOT"))
    from model.biot import BIOTEncoder

    model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100).cuda()
    state = torch.load(BIOT_CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and not any(key.startswith("patch_embedding") for key in state):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_head_ensemble(embeddings: np.ndarray, heads: list[torch.nn.Module], device: torch.device, batch_size: int = 4096) -> np.ndarray:
    output: list[np.ndarray] = []
    tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device, non_blocking=True)
            probabilities = []
            for head in heads:
                probabilities.append(torch.softmax(head(batch), dim=1))
            output.append(torch.stack(probabilities, dim=0).mean(dim=0).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, len(ACTIONS)), dtype=np.float32)


def infer_tusz_eval(output_dir: Path) -> pd.DataFrame:
    prediction_path = output_dir / "policy_predictions.parquet"
    if prediction_path.exists():
        return pd.read_parquet(prediction_path)

    labels_path = TUSZ_OLD_ROOT / "labels/eval_formal_fold0_seed17.parquet"
    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    _, heads, device = load_heads()
    encoder = load_biot_encoder()
    embeddings = np.empty((len(labels), 256), dtype=np.float32)
    started = time.monotonic()
    for cache_path, group in labels.groupby("cache_path", sort=False):
        cache_file = TUSZ_CACHE_ROOT / str(cache_path)
        archive = np.load(cache_file, allow_pickle=False)
        array = np.asarray(archive["x"])
        archive.close()
        indices = group.index.to_numpy(dtype=np.int64)
        rows = group.cache_row.to_numpy(dtype=np.int64)
        windows = np.asarray(resample_poly(array[rows, :, -1280:], up=25, down=16, axis=-1), dtype=np.float32)
        q95 = np.quantile(np.abs(windows), 0.95, axis=-1, keepdims=True)
        windows = np.ascontiguousarray(windows / (q95 + 1e-8), dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embedded = encoder(torch.from_numpy(windows).to(device, non_blocking=True))
        embeddings[indices] = embedded.float().cpu().numpy()

    probabilities = infer_head_ensemble(embeddings, heads, device)
    prediction = labels[["sample_id", "patient_id", "recording", "cache_row", "future_start_s", "future_end_s"]].copy()
    prediction["embedding_row_id"] = np.arange(len(prediction), dtype=np.int64)
    prediction["predicted_action_id"] = probabilities.argmax(axis=1).astype(np.int8)
    prediction["predicted_action"] = prediction.predicted_action_id.map(dict(enumerate(ACTIONS)))
    prediction["predicted_max_probability"] = probabilities.max(axis=1).astype(np.float32)
    for index, action in enumerate(ACTIONS):
        prediction[f"policy_probability_{action}"] = probabilities[:, index].astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(prediction_path, index=False)
    (output_dir / "policy_inference_manifest.json").write_text(
        json.dumps(
            {
                "method": "uniform mean of five new TUSZ policy heads",
                "encoder_checkpoint": str(BIOT_CHECKPOINT.relative_to(ROOT)),
                "encoder_checkpoint_sha256": sha256(BIOT_CHECKPOINT),
                "source_labels": str(labels_path.relative_to(ROOT)),
                "source_labels_sha256": sha256(labels_path),
                "rows": int(len(prediction)),
                "elapsed_seconds": time.monotonic() - started,
                "risk_head": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for head in heads:
        del head
    del encoder
    torch.cuda.empty_cache()
    return prediction


def load_mit_predictions() -> pd.DataFrame:
    parts = []
    for fold in range(5):
        path = OUTPUT_ROOT / "mit" / f"fold_{fold}" / "test_predictions.parquet"
        part = pd.read_parquet(path)
        part["fold"] = fold
        parts.append(part)
    prediction = pd.concat(parts, ignore_index=True)
    if prediction.embedding_row_id.duplicated().any():
        raise RuntimeError("duplicate MIT test embedding rows")
    return prediction


def load_tusz_event_cache(manifest: dict[str, object]):
    from bfa.evaluation.eventize import eventize

    event_cache: dict[str, dict[str, list[object]]] = {}
    thresholds: dict[str, float] = {}
    for record in manifest["records"]:
        recording = str(record["relative_edf"])
        event_cache[recording] = {}
        for model in MODELS:
            probability = TUSZ_EXPERT_ROOT / "eval" / model / "fold0_seed17" / "probabilities" / Path(recording).with_suffix(".parquet")
            metadata = TUSZ_EXPERT_ROOT / "eval" / model / "fold0_seed17" / "metadata" / Path(recording).with_suffix(".json")
            table = pd.read_parquet(probability).sort_values("window_end_s")
            threshold = float(json.loads(metadata.read_text())["threshold"])
            thresholds[model] = threshold
            event_cache[recording][model] = eventize(
                table.window_end_s.to_numpy(dtype=float),
                table.probability.to_numpy(dtype=float),
                threshold=threshold,
            ) if len(table) >= 2 else []
    return event_cache, thresholds


def load_mit_event_cache():
    from bfa.evaluation.eventize import eventize

    table = pd.read_parquet(MIT_PROBABILITIES)
    table = table[table.seed.astype(int).eq(3407)].copy()
    cache: dict[str, dict[str, list[object]]] = {}
    thresholds: dict[tuple[str, str], float] = {}
    for (recording, model), group in table.groupby(["recording", "model"], sort=False):
        group = group.sort_values("end", kind="stable")
        model_name = str(model)
        if model_name not in MODELS:
            continue
        threshold_values = group.validation_locked_threshold.dropna().astype(float).unique()
        if len(threshold_values) != 1:
            raise RuntimeError(f"MIT threshold is not unique for {recording}/{model_name}: {threshold_values}")
        thresholds[(str(recording), model_name)] = float(threshold_values[0])
        cache.setdefault(str(recording), {})[model_name] = eventize(
            group.end.to_numpy(dtype=float), group.probability.to_numpy(dtype=float), threshold=float(threshold_values[0])
        ) if len(group) >= 2 else []
    return cache, thresholds


def dedupe(events: list[object]) -> list[object]:
    unique: dict[tuple[float, float], object] = {}
    for event in events:
        key = (round(float(event.start_s), 3), round(float(event.end_s), 3))
        previous = unique.get(key)
        if previous is None or event.peak_probability > previous.peak_probability:
            unique[key] = event
    return sorted(unique.values(), key=lambda event: (event.start_s, event.end_s))


def union_duration(events: list[object], start_s: float, end_s: float) -> float:
    """Return the clipped union duration of alarm events, in seconds."""
    intervals = sorted(
        (
            max(float(event.start_s), float(start_s)),
            min(float(event.end_s), float(end_s)),
        )
        for event in events
        if float(event.end_s) > float(start_s) and float(event.start_s) < float(end_s)
    )
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for interval_start, interval_end in intervals:
        if interval_end <= interval_start:
            continue
        if current_start is None:
            current_start, current_end = interval_start, interval_end
        elif interval_start <= float(current_end):
            current_end = max(float(current_end), interval_end)
        else:
            total += float(current_end) - float(current_start)
            current_start, current_end = interval_start, interval_end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return float(total)


def truths_mit(seizures: pd.DataFrame, recording: str, evaluation_start_s: float) -> list[tuple[float, float]]:
    rows = seizures[seizures.recording_id.astype(str).eq(str(recording))]
    return [
        (max(float(row.start_s), evaluation_start_s), float(row.end_s))
        for row in rows.itertuples(index=False)
        if float(row.end_s) > evaluation_start_s
    ]


def score(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    event_cache: dict[str, dict[str, list[object]]],
    records: dict[str, dict[str, object]],
    truth_fn,
    method: str,
) -> pd.DataFrame:
    from bfa.evaluation.match import match_events

    pred_by_id = predictions.set_index("embedding_row_id") if "embedding_row_id" in predictions.columns else None
    rows: list[dict[str, object]] = []
    for recording, group in labels.groupby("recording", sort=False):
        recording = str(recording)
        record = records[recording]
        duration = float(record["duration_s"])
        ordered = group.sort_values(["future_start_s", "future_end_s"], kind="stable")
        evaluation_start_s = float(ordered.future_start_s.iloc[0])
        truths = truth_fn(recording, evaluation_start_s, record)
        if method.startswith("fixed_"):
            action = method.removeprefix("fixed_")
            selected_predictions = event_cache[recording][action]
        else:
            selected: list[object] = []
            for row in ordered.itertuples(index=False):
                if pred_by_id is not None:
                    policy = pred_by_id.loc[int(row.embedding_row_id)]
                else:
                    policy = row
                action = str(policy.predicted_action)
                if action == "reject":
                    continue
                start, end = float(row.future_start_s), float(row.future_end_s)
                selected.extend(
                    event for event in event_cache[recording][action]
                    if event.end_s > start and event.start_s < end
                )
            selected_predictions = dedupe(selected)
        matching = match_events(selected_predictions, truths)
        delays = [
            max(0.0, float(selected_predictions[pair.prediction_index].start_s) - float(truths[pair.truth_index][0]))
            for pair in matching.pairs
        ]
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        evaluation_duration_s = max(0.0, duration - evaluation_start_s)
        rows.append(
            {
                "method": method,
                "patient_id": str(record["patient_id"]),
                "recording": recording,
                "truth_events": int(len(truths)),
                "tp": int(len(matching.pairs)),
                "fp": int(len(matching.unmatched_predictions)),
                "fn": int(len(matching.unmatched_truths)),
                "delay_count": int(len(delays)),
                "delay_sum_s": float(sum(delays)),
                "nonseizure_s": float(max(0.0, duration - evaluation_start_s - seizure_seconds)),
                "alarm_duration_s": union_duration(selected_predictions, evaluation_start_s, duration),
                "evaluation_duration_s": evaluation_duration_s,
                "policy_rows": int(len(ordered)),
            }
        )
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> dict[str, object]:
    truth = int(frame.truth_events.sum())
    tp = int(frame.tp.sum())
    fp = int(frame.fp.sum())
    nonseizure = float(frame.nonseizure_s.sum())
    count = int(frame.delay_count.sum())
    alarm_duration = float(frame.alarm_duration_s.sum())
    evaluation_duration = float(frame.evaluation_duration_s.sum())
    return {
        "patients": int(frame.patient_id.nunique()),
        "records": int(len(frame)),
        "truth_events": truth,
        "tp": tp,
        "fp": fp,
        "fn": truth - tp,
        "sensitivity": tp / truth if truth else float("nan"),
        "fa_per_24h": fp * 86400.0 / nonseizure if nonseizure else float("nan"),
        "mean_delay_s": float(frame.delay_sum_s.sum()) / count if count else float("nan"),
        "nonseizure_s": nonseizure,
        "matched_events": count,
        "alarm_duration_s": alarm_duration,
        "evaluation_duration_s": evaluation_duration,
        "alarm_time_pct": 100.0 * alarm_duration / evaluation_duration if evaluation_duration else float("nan"),
    }


def action_summary(prediction: pd.DataFrame) -> dict[str, object]:
    counts = prediction.predicted_action.value_counts().reindex(ACTIONS, fill_value=0)
    return {"rows": int(len(prediction)), "counts": {name: int(counts[name]) for name in ACTIONS}, "fractions": {name: float(counts[name] / len(prediction)) if len(prediction) else float("nan") for name in ACTIONS}}


def evaluate_mit(output_dir: Path) -> dict[str, object]:
    labels = pd.read_parquet(EMBED_ROOT / "mit/index.parquet")
    labels = labels[labels.partition.astype(str).eq("chbmit_oof")].copy()
    labels["embedding_row_id"] = labels.embedding_row_id.astype(np.int64)
    predictions = load_mit_predictions()
    labels = labels[labels.embedding_row_id.isin(predictions.embedding_row_id)].copy()
    event_cache, _ = load_mit_event_cache()
    recordings = pd.read_parquet(MIT_RECORDINGS)
    seizures = pd.read_parquet(MIT_SEIZURES)
    records = {
        str(row.recording_id): {"patient_id": str(row.patient_id), "duration_s": float(row.duration_s)}
        for row in recordings.itertuples(index=False)
    }
    truth_fn = lambda recording, start, record: truths_mit(seizures, recording, start)
    methods = ["fixed_singlem", "fixed_cbramod", "fixed_tcn_gat", "new_moe"]
    frames = []
    summaries = []
    for method in methods:
        frame = score(labels, predictions, event_cache, records, truth_fn, method)
        frames.append(frame)
        summaries.append({"method": method, **summary(frame)})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output_dir / "record_metrics.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    prediction_action = predictions.copy()
    prediction_action.to_parquet(output_dir / "policy_predictions.parquet", index=False)
    report = {"dataset": "mit", "protocol": "outer-test OOF; fold-specific new policy head; seed=3407", "action_summary": action_summary(predictions), "summary": summaries}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def evaluate_tusz(output_dir: Path) -> dict[str, object]:
    manifest = json.loads((TUSZ_OLD_ROOT / "manifests/eval_manifest.json").read_text())
    labels = pd.read_parquet(TUSZ_OLD_ROOT / "labels/eval_formal_fold0_seed17.parquet").copy()
    labels["embedding_row_id"] = np.arange(len(labels), dtype=np.int64)
    predictions = infer_tusz_eval(output_dir)
    event_cache, thresholds = load_tusz_event_cache(manifest)
    records = {str(record["relative_edf"]): record for record in manifest["records"]}
    truth_fn = lambda recording, start, record: [
        (max(70.0, float(event["start_s"])), float(event["end_s"]))
        for event in record["seizure_events"]
        if float(event["end_s"]) > 70.0
    ]
    methods = ["fixed_singlem", "fixed_cbramod", "fixed_tcn_gat", "new_moe"]
    frames = []
    summaries = []
    for method in methods:
        frame = score(labels, predictions, event_cache, records, truth_fn, method)
        frames.append(frame)
        summaries.append({"method": method, **summary(frame)})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output_dir / "record_metrics.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    report = {
        "dataset": "tusz",
        "protocol": "official Eval; uniform mean of five new policy heads; expert thresholds frozen before Eval",
        "risk_head": False,
        "reject_semantics": "reject suppresses all expert alarms in the 60 s routed interval",
        "expert_thresholds": thresholds,
        "action_summary": action_summary(predictions),
        "summary": summaries,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mit", "tusz", "both"), default="both")
    args = parser.parse_args()
    if args.dataset in {"mit", "both"}:
        mit_report = evaluate_mit(OUTPUT_ROOT / "mit/final_event_scoring")
        print(json.dumps(mit_report, indent=2, sort_keys=True, default=float), flush=True)
    if args.dataset in {"tusz", "both"}:
        tusz_report = evaluate_tusz(OUTPUT_ROOT / "tusz/final_event_scoring")
        print(json.dumps(tusz_report, indent=2, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
