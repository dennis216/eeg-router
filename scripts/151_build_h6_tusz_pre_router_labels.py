"""Build offline expert-competence labels for the causal TUSZ pre-router.

The raw EEG router never receives these values.  They are derived from frozen
expert probabilities only to define the future-60-s supervision target.  A
multi-label target is retained whenever more than one expert is correct;
``reject`` is positive only for an all-FP background horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ("singlem", "cbramod", "tcn_gat")
TAU = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _probability_and_metadata(root: Path, recording: str) -> tuple[pd.DataFrame, dict[str, object], Path, Path]:
    relative = Path(recording)
    probability = root / "probabilities" / relative.with_suffix(".parquet")
    metadata = root / "metadata" / relative.with_suffix(".json")
    if not probability.exists() or not metadata.exists():
        raise FileNotFoundError(probability if not probability.exists() else metadata)
    return pd.read_parquet(probability), json.loads(metadata.read_text()), probability, metadata


def _events(table: pd.DataFrame, threshold: float) -> list[Event]:
    if len(table) < 2:
        return []
    table = table.sort_values("window_end_s")
    times = table["window_end_s"].to_numpy(dtype=float)
    probabilities = table["probability"].to_numpy(dtype=float)
    return eventize(times, probabilities, threshold=threshold)


def _calibrate(raw: np.ndarray, correct: np.ndarray) -> dict[str, object]:
    raw = np.asarray(raw, dtype=float)
    correct = np.asarray(correct, dtype=int)
    finite = np.isfinite(raw) & np.isfinite(correct)
    raw, correct = raw[finite], correct[finite]
    if len(raw) < 8 or len(np.unique(correct)) < 2:
        return {"method": "identity_insufficient_classes", "a": 1.0, "b": 0.0, "n": int(len(raw))}
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260829)
    model.fit(raw.reshape(-1, 1), correct)
    return {"method": "platt_logistic", "a": float(model.coef_[0, 0]), "b": float(model.intercept_[0]), "n": int(len(raw))}


def _apply_calibration(raw: float, calibration: dict[str, object]) -> float:
    if calibration.get("method") == "platt_logistic":
        value = float(calibration["a"]) * float(raw) + float(calibration["b"])
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))
    return float(np.clip(raw, 0.0, 1.0))


def _record_labels(record: dict[str, object], index_rows: pd.DataFrame, model_roots: dict[str, Path], calibration: dict[str, dict[str, object]]):
    tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, object]] = {}
    events: dict[str, list[Event]] = {}
    source_paths: dict[str, tuple[Path, Path]] = {}
    source_hashes: dict[str, str] = {}
    for model in MODELS:
        table, meta, probability, metadata_path = _probability_and_metadata(model_roots[model], str(record["relative_edf"]))
        tables[model], metadata[model] = table, meta
        events[model] = _events(table, float(meta["threshold"]))
        source_paths[model] = (probability, metadata_path)
        # Hash once per EDF/model, never once per decision window. The frozen
        # inference metadata already records this content hash, so avoid
        # rereading large parquet files during label construction.
        source_hashes[model] = str(meta.get("probability_sha256")) if meta.get("probability_sha256") else sha256(probability)
    rows: list[dict[str, object]] = []
    truths_all = [(float(event["start_s"]), float(event["end_s"])) for event in record["seizure_events"]]
    for row in index_rows.sort_values("cache_row").itertuples(index=False):
        start, end = float(row.future_start_s), float(row.future_end_s)
        truths = [(a, b) for a, b in truths_all if b > start and a < end]
        has_seizure = bool(truths)
        correct: dict[str, int] = {}
        confidence_raw: dict[str, float] = {}
        confidence: dict[str, float] = {}
        delay: dict[str, float] = {}
        event_counts: dict[str, int] = {}
        for model in MODELS:
            in_horizon = [event for event in events[model] if event.end_s > start and event.start_s < end]
            event_counts[model] = len(in_horizon)
            matching = match_events(in_horizon, truths) if has_seizure else None
            is_correct = bool(matching and matching.pairs) if has_seizure else not in_horizon
            correct[model] = int(is_correct)
            if matching and matching.pairs:
                pair = matching.pairs[0]
                confidence_raw[model] = float(in_horizon[pair.prediction_index].peak_probability)
                delay[model] = float(max(0.0, in_horizon[pair.prediction_index].start_s - truths[pair.truth_index][0]))
            else:
                future_prob = tables[model][(tables[model].window_end_s >= start) & (tables[model].window_end_s <= end)]
                max_probability = float(future_prob.probability.max()) if len(future_prob) else 0.0
                confidence_raw[model] = float(1.0 - max_probability) if not has_seizure else max_probability
                delay[model] = float(60.0)
            confidence[model] = _apply_calibration(confidence_raw[model], calibration[model])
        correct_models = [model for model in MODELS if correct[model]]
        reject = int((not has_seizure) and all(event_counts[model] > 0 for model in MODELS))
        unroutable = int(has_seizure and not correct_models)
        weights = {model: 0.0 for model in MODELS}
        if correct_models:
            scores = np.array([(confidence[m] - 0.01 * delay[m]) / TAU for m in correct_models], dtype=float)
            scores -= float(scores.max())
            softmax = np.exp(scores) / np.exp(scores).sum()
            for model, value in zip(correct_models, softmax, strict=True):
                weights[model] = float(0.3 / len(correct_models) + 0.7 * value)
        row_out = {
            "sample_id": str(row.sample_id),
            "partition": str(row.partition),
            "patient_id": str(row.patient_id),
            "recording": str(row.recording),
            "cache_path": str(row.cache_path),
            "cache_row": int(row.cache_row),
            "context_start_s": float(row.context_start_s),
            "context_end_s": float(row.context_end_s),
            "future_start_s": start,
            "future_end_s": end,
            "future_has_seizure": int(has_seizure),
            "reject_label": reject,
            "unroutable": unroutable,
            "correct_mask": "".join(str(correct[m]) for m in MODELS),
        }
        for model in MODELS:
            row_out.update({
                f"correct_{model}": correct[model],
                f"confidence_raw_{model}": confidence_raw[model],
                f"confidence_{model}": confidence[model],
                f"delay_s_{model}": delay[model],
                f"event_count_{model}": event_counts[model],
                f"target_weight_{model}": weights[model],
                f"threshold_{model}": float(metadata[model]["threshold"]),
                f"probability_sha256_{model}": source_hashes[model],
            })
        rows.append(row_out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1")
    parser.add_argument("--output-name", default="fold0_seed17")
    parser.add_argument("--label-id", help="immutable label filename suffix; defaults to --output-name")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--fit-calibration", action="store_true")
    args = parser.parse_args()
    manifest_path = (args.manifest or (OUT / "manifests" / f"{args.partition}_manifest.json")).resolve()
    index_path = (args.index or (OUT / "indices" / f"{args.partition}_index.parquet")).resolve()
    manifest = json.loads(manifest_path.read_text())
    index = pd.read_parquet(index_path)
    model_roots = {model: args.input_root.resolve() / model / args.output_name for model in MODELS}
    raw_calibration = {model: {"method": "identity", "a": 1.0, "b": 0.0, "n": 0} for model in MODELS}
    if args.calibration and args.calibration.exists():
        raw_calibration.update(json.loads(args.calibration.read_text()).get("models", {}))

    # Generate raw labels once using identity/previous calibration.  Train is
    # then used to fit Platt maps, after which its table is rewritten with the
    # frozen calibrated values; no validation/eval values enter this fit.
    model_records = {str(r["relative_edf"]): r for r in manifest["records"]}
    all_rows: list[dict[str, object]] = []
    for recording, group in index.groupby("recording", sort=False):
        if recording not in model_records:
            raise KeyError(f"record absent from manifest: {recording}")
        all_rows.extend(_record_labels(model_records[recording], group, model_roots, raw_calibration))
    frame = pd.DataFrame(all_rows)
    calibration_path = (args.calibration or (OUT / "calibration" / "train_platt.json")).resolve()
    if args.fit_calibration:
        fitted = {"models": {}, "fit_partition": args.partition, "fit_manifest_sha256": sha256(manifest_path), "created_utc": datetime.now(timezone.utc).isoformat()}
        for model in MODELS:
            fitted["models"][model] = _calibrate(frame[f"confidence_raw_{model}"].to_numpy(), frame[f"correct_{model}"].to_numpy())
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
        raw_calibration = fitted["models"]
        frame = pd.DataFrame()
        all_rows = []
        for recording, group in index.groupby("recording", sort=False):
            all_rows.extend(_record_labels(model_records[recording], group, model_roots, raw_calibration))
        frame = pd.DataFrame(all_rows)
    label_id = args.label_id or args.output_name
    output_path = OUT / "labels" / f"{args.partition}_{label_id}.parquet"
    audit_path = OUT / "labels" / f"{args.partition}_{label_id}_audit.json"
    if output_path.exists() or audit_path.exists():
        raise RuntimeError(f"refusing to overwrite labels: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    masks = frame["correct_mask"].value_counts().sort_index().to_dict()
    audit = {
        "release_id": NAMESPACE,
        "partition": args.partition,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "index_sha256": sha256(index_path),
        "calibration": str(calibration_path.relative_to(ROOT)) if calibration_path.exists() else None,
        "calibration_sha256": sha256(calibration_path) if calibration_path.exists() else None,
        "expert_roots": {model: str(model_roots[model].relative_to(ROOT)) for model in MODELS},
        "rows": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "correct_mask_counts": {str(k): int(v) for k, v in masks.items()},
        "reject_positive": int(frame.reject_label.sum()),
        "unroutable": int(frame.unroutable.sum()),
        "reject_definition": "future background and all three frozen experts emit at least one FP event",
        "router_input_excludes_expert_outputs": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_path.with_suffix(".sha256").write_text(sha256(output_path) + "\n")
    audit_path.with_suffix(".sha256").write_text(sha256(audit_path) + "\n")
    print(json.dumps({"labels": str(output_path), "audit": str(audit_path), **audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
