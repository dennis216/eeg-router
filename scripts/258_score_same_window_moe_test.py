from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.same_window_moe import ACTIONS, EXPERTS, aggregate_components, eventize_common, false_positive_overlap, score_events, threshold_normalize


ROOT = Path("/root/b_false_alarm_atlas")
REPORT = ROOT / "outputs/reports/same-window-router-first-moe-v1"
EMBED = Path("/mnt/d/EEGData/same_window_router_first_moe_v1/biot_embeddings/test")
MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/eval_manifest.json"


def load_label_builder():
    path = ROOT / "scripts/253_build_same_window_router_labels.py"; spec = importlib.util.spec_from_file_location("same_window_label_builder", path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def truths(record):
    return [(float(x.get("start_s", x.get("start", 0))), float(x.get("end_s", x.get("end", 0)))) if isinstance(x, dict) else (float(x[0]), float(x[1])) for x in record.get("seizure_events", [])]


def score(frame: pd.DataFrame, values: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int]:
    manifest = json.loads(MANIFEST.read_text()); records = {str(x["relative_edf"]): x for x in manifest["records"]}; parts = []
    for recording, positions in frame.groupby("recording", sort=False).indices.items():
        index = np.asarray(positions); order = index[np.argsort(frame.iloc[index].window_end_s.to_numpy(float))]; times = frame.iloc[order].window_end_s.to_numpy(float); events = eventize_common(times, values[order], threshold=threshold); record = records[str(recording)]; parts.append(score_events(events, truths(record), float(times.min()), float(record["duration_s"])))
    return aggregate_components(parts)


def main() -> None:
    builder = load_label_builder(); frame = builder.load_aligned(REPORT / "expert_probabilities/tusz", "test"); thresholds = json.loads((REPORT / "expert_audit/thresholds.json").read_text())
    index = pd.read_parquet(EMBED / "index.parquet"); checkpoint = torch.load(REPORT / "router_training/best.pt", map_location="cpu", weights_only=False)
    spec = importlib.util.spec_from_file_location("same_window_router", ROOT / "scripts/254_train_same_window_router.py"); router_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(router_module); model = router_module.RouterHead().cuda().eval(); model.load_state_dict(checkpoint["state_dict"])
    mmap = np.memmap(EMBED / "embeddings.float16.memmap", mode="r", dtype=np.float16, shape=(len(index), 256)); probabilities = []
    with torch.inference_mode():
        for start in range(0, len(index), 4096):
            x = torch.from_numpy(np.asarray(mmap[start : start + 4096], dtype=np.float32)).cuda(); probabilities.append(torch.softmax(model(x), dim=1).cpu().numpy())
    router_probability = np.concatenate(probabilities); actions = router_probability.argmax(axis=1); keys = ["patient_id", "recording", "window_start_s", "window_end_s"]
    aligned = index[keys + ["embedding_row_id"]].merge(frame, on=keys, how="left", validate="one_to_one")
    if aligned[[f"probability_{m}" for m in EXPERTS]].isna().any().any(): raise RuntimeError("test embedding/expert alignment failed")
    routed = np.zeros(len(aligned), dtype=float)
    fixed_scores = {}
    rows = []
    for action_id, model_name in enumerate(EXPERTS):
        raw = aligned[f"probability_{model_name}"].to_numpy(float); values = threshold_normalize(raw, float(thresholds[model_name])); fixed_scores[model_name] = values; rows.append({"method": model_name, "router": "no", **score(aligned, raw, threshold=float(thresholds[model_name]))}); mask = actions == action_id; routed[mask] = values[mask]
    rows.append({"method": "same_window_moe", "router": "yes", **score(aligned, routed)})
    result = pd.DataFrame(rows); best_name = str(checkpoint["best_fixed_expert"]); best = result[result.method == best_name].iloc[0]; moe = result[result.method == "same_window_moe"].iloc[0]
    result = pd.concat([result, pd.DataFrame([{**best.to_dict(), "method": "best_fixed_expert"}])], ignore_index=True)
    deltas = []
    for reference in (*EXPERTS, "best_fixed_expert"):
        row = result[result.method == reference].iloc[0]; deltas.append({"comparison": f"moe_vs_{reference}", "delta_sensitivity": float(moe.sensitivity-row.sensitivity), "delta_fa_per_24h": float(moe.fa_per_24h-row.fa_per_24h), "fa_relative_change": float((moe.fa_per_24h-row.fa_per_24h)/row.fa_per_24h) if row.fa_per_24h else np.nan, "delta_false_alarm_time_min_per_24h": float(moe.false_alarm_time_min_per_24h-row.false_alarm_time_min_per_24h), "false_alarm_time_relative_change": float((moe.false_alarm_time_min_per_24h-row.false_alarm_time_min_per_24h)/row.false_alarm_time_min_per_24h) if row.false_alarm_time_min_per_24h else np.nan})
    binary = pd.DataFrame({"label": aligned.label})
    for model_name in EXPERTS: binary[f"positive_{model_name}"] = fixed_scores[model_name] >= 0.5
    overlap = false_positive_overlap(binary); output = REPORT / "test_scoring"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "comparison.csv", index=False); pd.DataFrame(deltas).to_csv(output / "deltas.csv", index=False); (output / "false_positive_window_overlap.json").write_text(json.dumps(overlap, indent=2, sort_keys=True) + "\n")
    pd.DataFrame({"embedding_row_id": index.embedding_row_id, "predicted_action_id": actions, "predicted_action": [ACTIONS[x] for x in actions], "final_score": routed}).to_parquet(output / "router_test_predictions.parquet", index=False)
    print(result[["method", "router", "sensitivity", "fa_per_24h", "false_alarm_time_min_per_24h", "alarm_time_pct"]].to_string(index=False)); print(json.dumps(overlap, indent=2, sort_keys=True))


if __name__ == "__main__": main()
