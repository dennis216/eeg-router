"""Select Dev thresholds and score the TUSZ raw-EEG pre-router on Eval.

The router sees only past raw EEG.  Expert probabilities are used here only
to construct the fixed reference event streams and the interval oracle.  The
Eval branch is deliberately single-use: it refuses to overwrite a previous
evaluation manifest and never searches thresholds on Eval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
NAMESPACE = "h6-tusz-raw-pre-router-v1"
OUT = ROOT / "outputs/reports" / NAMESPACE
MODELS = ("singlem", "cbramod", "tcn_gat")
ENSEMBLE_AMENDMENT = OUT / "raw_router_config_amendment_equal_probability_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RawPreRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(16, 32, 15, stride=4, padding=7), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 9, stride=4, padding=4), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.competence_head = nn.Linear(128, 3)
        self.reject_head = nn.Linear(128, 1)
        self.seizure_risk_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.features(x).squeeze(-1)
        return {
            "competence_logits": self.competence_head(representation),
            "reject_logit": self.reject_head(representation).squeeze(-1),
            "seizure_risk_logit": self.seizure_risk_head(representation).squeeze(-1),
        }


def _load_router_predictions(labels: pd.DataFrame, cache_root: Path, checkpoint_path: Path, batch_size: int = 64) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for raw-router inference")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = RawPreRouter().cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    outputs = np.zeros((len(labels), 5), dtype=np.float32)
    with torch.inference_mode():
        for cache_path, group in labels.groupby("cache_path", sort=False):
            cache_file = cache_root / cache_path
            if not cache_file.exists():
                cache_file = OUT / cache_path
            array = np.load(cache_file, allow_pickle=False)["x"]
            group = group.sort_values("cache_row")
            indices = group.index.to_numpy()
            rows = group.cache_row.to_numpy(dtype=int)
            for offset in range(0, len(rows), batch_size):
                selected = rows[offset : offset + batch_size]
                x = torch.from_numpy(array[selected].astype(np.float32, copy=False)).cuda(non_blocking=True)
                pred = model(x)
                values = np.concatenate([
                    torch.sigmoid(pred["competence_logits"]).cpu().numpy(),
                    torch.sigmoid(pred["reject_logit"]).cpu().numpy()[:, None],
                    torch.sigmoid(pred["seizure_risk_logit"]).cpu().numpy()[:, None],
                ], axis=1)
                outputs[indices[offset : offset + len(selected)]] = values
    prediction = labels.copy()
    for i, model_name in enumerate(MODELS):
        prediction[f"router_competence_{model_name}"] = outputs[:, i]
    prediction["router_reject_probability"] = outputs[:, 3]
    prediction["router_seizure_risk"] = outputs[:, 4]
    return prediction


def _read_event_cache(manifest: dict[str, object], expert_root: Path):
    event_cache: dict[str, dict[str, list[Event]]] = {}
    ensemble_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    thresholds: dict[str, float] = {}
    for record in manifest["records"]:
        recording = str(record["relative_edf"])
        event_cache[recording] = {}
        tables: dict[str, pd.DataFrame] = {}
        for model in MODELS:
            probability = expert_root / model / "fold0_seed17" / "probabilities" / Path(recording).with_suffix(".parquet")
            metadata_path = expert_root / model / "fold0_seed17" / "metadata" / Path(recording).with_suffix(".json")
            if not probability.exists() or not metadata_path.exists():
                raise FileNotFoundError(probability if not probability.exists() else metadata_path)
            table = pd.read_parquet(probability).sort_values("window_end_s")
            tables[model] = table
            metadata = json.loads(metadata_path.read_text())
            threshold = float(metadata["threshold"])
            thresholds[model] = threshold
            event_cache[recording][model] = eventize(
                table.window_end_s.to_numpy(dtype=float), table.probability.to_numpy(dtype=float), threshold=threshold
            ) if len(table) >= 2 else []
        # Equal-probability ensemble: use only time points present in all
        # frozen expert outputs, then average the three probabilities.  The
        # event threshold is selected on Dev and frozen for Eval.
        if all(len(tables[m]) for m in MODELS):
            merged = tables[MODELS[0]][["window_end_s", "probability"]].rename(columns={"probability": "p0"})
            for i, model in enumerate(MODELS[1:], start=1):
                merged = merged.merge(
                    tables[model][["window_end_s", "probability"]].rename(columns={"probability": f"p{i}"}),
                    on="window_end_s", how="inner", validate="one_to_one",
                )
            if len(merged):
                ensemble_cache[recording] = (
                    merged.window_end_s.to_numpy(dtype=float),
                    merged[[f"p{i}" for i in range(len(MODELS))]].mean(axis=1).to_numpy(dtype=float),
                )
            else:
                ensemble_cache[recording] = (np.empty(0, dtype=float), np.empty(0, dtype=float))
        else:
            ensemble_cache[recording] = (np.empty(0, dtype=float), np.empty(0, dtype=float))
    return event_cache, thresholds, ensemble_cache


def _dedupe(events: list[Event]) -> list[Event]:
    unique: dict[tuple[float, float], Event] = {}
    for event in events:
        key = (round(float(event.start_s), 3), round(float(event.end_s), 3))
        previous = unique.get(key)
        if previous is None or event.peak_probability > previous.peak_probability:
            unique[key] = event
    return sorted(unique.values(), key=lambda event: (event.start_s, event.end_s))


def _truths(record: dict[str, object]) -> list[tuple[float, float]]:
    return [(max(70.0, float(e["start_s"])), float(e["end_s"])) for e in record["seizure_events"] if float(e["end_s"]) > 70.0]


def _summary(record_rows: pd.DataFrame) -> dict[str, float | int]:
    if record_rows.empty:
        return {"patients": 0, "records": 0, "truth_events": 0, "tp": 0, "fp": 0, "fn": 0, "sensitivity": np.nan, "fa_per_24h": np.nan, "mean_delay_s": np.nan}
    truth = int(record_rows.truth_events.sum())
    tp = int(record_rows.tp.sum())
    nonseizure = float(record_rows.nonseizure_s.sum())
    count = int(record_rows.delay_count.sum())
    return {"patients": int(record_rows.patient_id.nunique()), "records": int(len(record_rows)), "truth_events": truth, "tp": tp, "fp": int(record_rows.fp.sum()), "fn": truth - tp, "sensitivity": tp / truth if truth else np.nan, "fa_per_24h": float(record_rows.fp.sum()) / (nonseizure / 86400.0) if nonseizure > 0 else np.nan, "mean_delay_s": float(record_rows.delay_sum_s.sum()) / count if count else np.nan}


def _score_actions(labels: pd.DataFrame, manifest: dict[str, object], events: dict[str, dict[str, list[Event]]], method: str, thresholds: dict[str, float] | None, best_fixed: str | None = None, ensemble_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    record_by_path = {str(r["relative_edf"]): r for r in manifest["records"]}
    all_rows: list[dict[str, object]] = []
    for recording, group in labels.groupby("recording", sort=False):
        record = record_by_path[str(recording)]
        truths = _truths(record)
        duration = float(record["duration_s"])
        if method.startswith("fixed_"):
            model = method.removeprefix("fixed_")
            predictions = events[recording][model]
        elif method == "best_fixed":
            predictions = events[recording][str(best_fixed)]
        elif method == "equal_ensemble":
            if ensemble_cache is None or recording not in ensemble_cache:
                raise RuntimeError("equal-probability ensemble cache is missing")
            ensemble_times, ensemble_probabilities = ensemble_cache[recording]
            ensemble_threshold = float((thresholds or {}).get("equal_ensemble_threshold", 0.5))
            predictions = eventize(ensemble_times, ensemble_probabilities, threshold=ensemble_threshold) if len(ensemble_times) >= 2 else []
        elif method in {"router", "router_no_reject", "oracle_interval"}:
            selected: list[Event] = []
            for row in group.itertuples(index=False):
                start, end = float(row.future_start_s), float(row.future_end_s)
                if method == "oracle_interval":
                    candidates = [m for m in MODELS if int(getattr(row, f"correct_{m}"))]
                    chosen = max(candidates, key=lambda m: float(getattr(row, f"target_weight_{m}"))) if candidates else None
                    if chosen is None:
                        continue
                else:
                    competence = {m: float(getattr(row, f"router_competence_{m}")) for m in MODELS}
                    chosen = max(MODELS, key=competence.get)
                    if method == "router" and thresholds is not None:
                        allow_reject = float(row.router_reject_probability) >= float(thresholds["reject_probability"]) and max(competence.values()) < float(thresholds["expert_competence"]) and float(row.router_seizure_risk) < float(thresholds["seizure_risk"])
                        if allow_reject:
                            continue
                selected.extend(event for event in events[recording][chosen] if event.end_s > start and event.start_s < end)
            predictions = _dedupe(selected)
        else:
            raise ValueError(method)
        matching = match_events(predictions, truths)
        delays = [max(0.0, predictions[pair.prediction_index].start_s - truths[pair.truth_index][0]) for pair in matching.pairs]
        seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
        all_rows.append({"method": method, "patient_id": str(record["patient_id"]), "recording": str(recording), "truth_events": len(truths), "tp": len(matching.pairs), "fp": len(matching.unmatched_predictions), "fn": len(matching.unmatched_truths), "delay_count": len(delays), "delay_sum_s": float(sum(delays)), "median_delay_s": float(np.median(delays)) if delays else np.nan, "nonseizure_s": max(0.0, duration - 70.0 - seizure_seconds)})
    return pd.DataFrame(all_rows)


def _bootstrap(record_rows: pd.DataFrame, method: str, replicates: int, seed: int) -> pd.DataFrame:
    patients = sorted(record_rows.patient_id.unique().tolist())
    rng = np.random.default_rng(seed)
    output = []
    for replicate in range(replicates):
        chosen = rng.choice(patients, len(patients), replace=True)
        sample = pd.concat([record_rows[record_rows.patient_id == patient] for patient in chosen], ignore_index=True)
        summary = _summary(sample)
        output.append({"method": method, "replicate": replicate, "sensitivity": summary["sensitivity"], "fa_per_24h": summary["fa_per_24h"], "mean_delay_s": summary["mean_delay_s"], "patients_sampled": len(patients)})
    return pd.DataFrame(output)


def select_dev(args: argparse.Namespace) -> None:
    manifest_path = (args.manifest or OUT / "manifests/dev_manifest.json").resolve()
    labels = pd.read_parquet(args.labels.resolve())
    manifest = json.loads(manifest_path.read_text())
    event_cache, expert_thresholds, ensemble_cache = _read_event_cache(manifest, args.expert_root.resolve())
    labels = _load_router_predictions(labels, OUT, args.checkpoint.resolve())
    fixed = {model: _summary(_score_actions(labels, manifest, event_cache, f"fixed_{model}", None, ensemble_cache=ensemble_cache)) for model in MODELS}
    best_fixed = max(MODELS, key=lambda model: (float(fixed[model]["sensitivity"]), -float(fixed[model]["fa_per_24h"])))
    baseline_sensitivity = float(fixed[best_fixed]["sensitivity"])
    candidates = []
    grid = [float(v) for v in args.grid.split(",")]
    for reject_probability in grid:
        for expert_competence in grid:
            for seizure_risk in grid:
                thresholds = {"reject_probability": reject_probability, "expert_competence": expert_competence, "seizure_risk": seizure_risk}
                rows = _score_actions(labels, manifest, event_cache, "router", thresholds, ensemble_cache=ensemble_cache)
                summary = _summary(rows)
                candidates.append({**thresholds, **summary, "safe_sensitivity": bool(float(summary["sensitivity"]) >= baseline_sensitivity - 0.03)})
    # Select the equal-probability ensemble threshold independently on Dev
    # under the same sensitivity constraint as the router.
    ensemble_candidates = []
    for ensemble_threshold in grid:
        summary = _summary(_score_actions(labels, manifest, event_cache, "equal_ensemble", {"equal_ensemble_threshold": ensemble_threshold}, ensemble_cache=ensemble_cache))
        ensemble_candidates.append({"equal_ensemble_threshold": ensemble_threshold, **summary, "safe_sensitivity": bool(float(summary["sensitivity"]) >= baseline_sensitivity - 0.03)})
    ensemble_frame = pd.DataFrame(ensemble_candidates)
    ensemble_safe = ensemble_frame[ensemble_frame.safe_sensitivity]
    if ensemble_safe.empty:
        ensemble_selected = ensemble_frame.sort_values(["sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0].to_dict()
    else:
        ensemble_selected = ensemble_safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0].to_dict()
    candidate_frame = pd.DataFrame(candidates)
    safe = candidate_frame[candidate_frame.safe_sensitivity]
    if safe.empty:
        selected = candidate_frame.sort_values(["sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0].to_dict()
        status = "no_safe_dev_point"
    else:
        selected = safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0].to_dict()
        status = "safe_dev_point_selected"
    selected["best_fixed_model"] = best_fixed
    selected["best_fixed_sensitivity"] = baseline_sensitivity
    selected["expert_thresholds"] = expert_thresholds
    selected["equal_ensemble_threshold"] = float(ensemble_selected["equal_ensemble_threshold"])
    selected["equal_ensemble_dev_summary"] = ensemble_selected
    selected["status"] = status
    selected["threshold_source"] = "TUSZ Dev only"
    out_dir = (args.output_dir or OUT / "scoring" / "dev").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "threshold_selection.json").write_text(json.dumps(selected, indent=2, sort_keys=True, default=float) + "\n")
    candidate_frame.to_csv(out_dir / "threshold_candidates.csv", index=False)
    ensemble_frame.to_csv(out_dir / "equal_ensemble_threshold_candidates.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"release_id": NAMESPACE, "partition": "dev", "status": status, "manifest_sha256": sha256(manifest_path), "labels_sha256": sha256(args.labels.resolve()), "checkpoint_sha256": sha256(args.checkpoint.resolve()), "protocol_amendment": str(ENSEMBLE_AMENDMENT.relative_to(ROOT)), "protocol_amendment_sha256": sha256(ENSEMBLE_AMENDMENT), "threshold_source": "dev-only", "eval_used_for_selection": False, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"partition": "dev", "best_fixed": best_fixed, "selected": selected, "output": str(out_dir)}, indent=2, sort_keys=True, default=float))


def score_eval(args: argparse.Namespace) -> None:
    manifest_path = (args.manifest or OUT / "manifests/eval_manifest.json").resolve()
    out_dir = (args.output_dir or OUT / "scoring" / "eval").resolve()
    if (out_dir / "manifest.json").exists():
        raise RuntimeError(f"refusing to re-evaluate locked Eval: {out_dir / 'manifest.json'}")
    selection = json.loads(args.thresholds.resolve().read_text())
    labels = pd.read_parquet(args.labels.resolve())
    manifest = json.loads(manifest_path.read_text())
    event_cache, _, ensemble_cache = _read_event_cache(manifest, args.expert_root.resolve())
    labels = _load_router_predictions(labels, OUT, args.checkpoint.resolve())
    best_fixed = str(selection["best_fixed_model"])
    thresholds = {key: float(selection[key]) for key in ("reject_probability", "expert_competence", "seizure_risk")}
    methods = ["fixed_singlem", "fixed_cbramod", "fixed_tcn_gat", "best_fixed", "equal_ensemble", "router_no_reject", "router", "oracle_interval"]
    rows_by_method = {}
    summaries = []
    bootstrap = []
    for method in methods:
        action_thresholds = thresholds if method == "router" else ({"equal_ensemble_threshold": float(selection["equal_ensemble_threshold"])} if method == "equal_ensemble" else None)
        rows = _score_actions(labels, manifest, event_cache, method, action_thresholds, best_fixed, ensemble_cache=ensemble_cache)
        rows_by_method[method] = rows
        summaries.append({"method": method, **_summary(rows)})
        bootstrap.append(_bootstrap(rows, method, args.bootstrap, args.bootstrap_seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(rows_by_method.values(), ignore_index=True).to_parquet(out_dir / "record_metrics.parquet", index=False)
    pd.concat(bootstrap, ignore_index=True).to_parquet(out_dir / "patient_bootstrap.parquet", index=False)
    manifest_payload = {"release_id": NAMESPACE, "partition": "eval", "status": "complete_single_evaluation", "cohort_manifest": str(manifest_path.relative_to(ROOT)), "cohort_manifest_sha256": sha256(manifest_path), "labels_sha256": sha256(args.labels.resolve()), "checkpoint_sha256": sha256(args.checkpoint.resolve()), "protocol_amendment": str(ENSEMBLE_AMENDMENT.relative_to(ROOT)), "protocol_amendment_sha256": sha256(ENSEMBLE_AMENDMENT), "thresholds": {**thresholds, "equal_ensemble_threshold": float(selection["equal_ensemble_threshold"])}, "threshold_source": "TUSZ Dev only", "best_fixed_model": best_fixed, "methods": methods, "bootstrap_replicates": args.bootstrap, "bootstrap_unit": "patient", "eval_evaluation_count_per_method": 1, "holdout_used_for_selection": False, "model_training": "router_only", "created_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    (out_dir / "manifest.sha256").write_text(sha256(out_dir / "manifest.json") + "\n")
    print(json.dumps({"partition": "eval", "summary": summaries, "output": str(out_dir)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    dev = sub.add_parser("dev")
    dev.add_argument("--labels", type=Path, required=True)
    dev.add_argument("--checkpoint", type=Path, required=True)
    dev.add_argument("--expert-root", type=Path, required=True)
    dev.add_argument("--manifest", type=Path)
    dev.add_argument("--grid", default="0.25,0.5,0.75")
    dev.add_argument("--output-dir", type=Path)
    dev.set_defaults(func=select_dev)
    ev = sub.add_parser("eval")
    ev.add_argument("--labels", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--expert-root", type=Path, required=True)
    ev.add_argument("--thresholds", type=Path, required=True)
    ev.add_argument("--manifest", type=Path)
    ev.add_argument("--bootstrap", type=int, default=2000)
    ev.add_argument("--bootstrap-seed", type=int, default=20260829)
    ev.add_argument("--output-dir", type=Path)
    ev.set_defaults(func=score_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
