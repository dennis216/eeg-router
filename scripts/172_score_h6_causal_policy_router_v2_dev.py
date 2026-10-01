"""Score the trained H6-v2 router on TUSZ Dev only.

This is a development-stage safety check.  It selects the three router
thresholds on Dev under the pre-registered sensitivity constraint and reports
event-level metrics plus patient bootstrap intervals.  TUSZ Eval is not read;
therefore these numbers are not a confirmatory final test.
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


ROOT = Path("/root/b_false_alarm_atlas")
LABEL_NS = "h6-causal-policy-router-v2-corrected"
LABEL_PATH = ROOT / "outputs/reports" / LABEL_NS / "labels/dev_causal_policy.parquet"
LABEL_MANIFEST = ROOT / "outputs/reports" / LABEL_NS / "label_manifest.json"
CACHE_ROOT = ROOT / "outputs/reports/h6-causal-policy-router-v2"
CKPT = ROOT / "outputs/reports" / LABEL_NS / "training_formal_train_only/checkpoint.pt"
EXPERT_ROOT = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1/dev"
EXPERT_MANIFEST = ROOT / "outputs/reports/h6-tusz-raw-pre-router-v1/manifests/dev_manifest.json"
OUT = ROOT / "outputs/reports" / LABEL_NS / "scoring_dev_formal_train_only"
MODELS = ("singlem", "cbramod", "tcn_gat")
NORMALIZE_INPUT = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def router_predictions(labels: pd.DataFrame, model_mod, batch_size: int = 256) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    model = model_mod.CausalPolicyNet().to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    values = np.zeros((len(labels), 5), dtype=np.float32)
    with torch.inference_mode():
        for cache_rel, group in labels.groupby("raw_cache_path_v2", sort=False):
            path = CACHE_ROOT / str(cache_rel)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as z:
                array = z["x"]
                group = group.sort_values("decision_index_v1")
                out_indices = group.index.to_numpy(dtype=int)
                rows = group.decision_index_v1.to_numpy(dtype=int)
                for offset in range(0, len(rows), batch_size):
                    selected = rows[offset : offset + batch_size]
                    raw = array[selected].astype(np.float32, copy=False)
                    if NORMALIZE_INPUT:
                        med = np.median(raw, axis=2, keepdims=True)
                        q25 = np.quantile(raw, 0.25, axis=2, keepdims=True)
                        q75 = np.quantile(raw, 0.75, axis=2, keepdims=True)
                        scale = np.maximum(q75 - q25, 1.0)
                        raw = np.clip((raw - med) / scale, -12.0, 12.0).astype(np.float32, copy=False)
                    x = torch.from_numpy(raw).to(device)
                    policy, reject, risk = model(x)
                    chunk = np.concatenate([
                        torch.sigmoid(policy).cpu().numpy(),
                        torch.sigmoid(reject).cpu().numpy()[:, None],
                        torch.sigmoid(risk).cpu().numpy()[:, None],
                    ], axis=1)
                    values[out_indices[offset : offset + len(selected)]] = chunk
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite router predictions")
    result = labels.copy()
    for i, name in enumerate(MODELS):
        result[f"router_competence_{name}"] = values[:, i]
        result[f"target_weight_{name}"] = result[f"policy_target_{name}"]
    result["router_reject_probability"] = values[:, 3]
    result["router_seizure_risk"] = values[:, 4]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="0.20,0.35,0.50,0.65,0.80,0.90")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    if not LABEL_PATH.exists() or not CKPT.exists():
        raise FileNotFoundError("corrected labels or formal checkpoint missing")
    scorer = load_module(ROOT / "scripts/155_score_h6_tusz_raw_pre_router.py", "h6_scorer_v2_dev")
    model_mod = load_module(ROOT / "scripts/170_smoke_h6_causal_policy_router_v2.py", "h6_model_v2_dev")
    labels = pd.read_parquet(LABEL_PATH)
    manifest = json.loads(EXPERT_MANIFEST.read_text())
    events, expert_thresholds, ensemble_cache = scorer._read_event_cache(manifest, EXPERT_ROOT)
    labels = router_predictions(labels, model_mod)
    fixed_rows = {m: scorer._score_actions(labels, manifest, events, f"fixed_{m}", None) for m in MODELS}
    fixed_summaries = {m: scorer._summary(v) for m, v in fixed_rows.items()}
    fixed_frame = pd.DataFrame([{"method": f"fixed_{m}", **s} for m, s in fixed_summaries.items()])
    best_sens = float(fixed_frame.sensitivity.max())
    safe_fixed = fixed_frame[fixed_frame.sensitivity >= best_sens - 0.03]
    best_fixed_row = safe_fixed.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0]
    best_fixed = str(best_fixed_row.method)
    baseline_sens = float(best_fixed_row.sensitivity)

    # No-reject policy is a fixed diagnostic; the routed policy grid is the
    # only threshold selection, and it is entirely on Dev.
    no_reject_rows = scorer._score_actions(labels, manifest, events, "router_no_reject", None, ensemble_cache=ensemble_cache)
    no_reject_summary = scorer._summary(no_reject_rows)
    grid = [float(v) for v in args.grid.split(",") if v.strip()]
    candidates: list[dict[str, object]] = []
    for rp in grid:
        for ec in grid:
            for sr in grid:
                thresholds = {"reject_probability": rp, "expert_competence": ec, "seizure_risk": sr}
                rows = scorer._score_actions(labels, manifest, events, "router", thresholds, ensemble_cache=ensemble_cache)
                summary = scorer._summary(rows)
                candidates.append({**thresholds, **summary, "safe_sensitivity": bool(float(summary["sensitivity"]) >= baseline_sens - 0.03)})
    candidate_frame = pd.DataFrame(candidates)
    safe = candidate_frame[candidate_frame.safe_sensitivity]
    if safe.empty:
        selected = candidate_frame.sort_values(["sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0].to_dict()
        selection_status = "no_safe_dev_point"
    else:
        selected = safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0].to_dict()
        selection_status = "safe_dev_point_selected"
    selected.update({"status": selection_status, "best_fixed_method": best_fixed, "best_fixed_sensitivity": baseline_sens, "threshold_source": "TUSZ Dev only", "eval_used_for_selection": False, "expert_thresholds": expert_thresholds})

    # Equal ensemble threshold is selected under the same constraint.
    ensemble_candidates: list[dict[str, object]] = []
    for et in grid:
        rows = scorer._score_actions(labels, manifest, events, "equal_ensemble", {"equal_ensemble_threshold": et}, ensemble_cache=ensemble_cache)
        summary = scorer._summary(rows)
        ensemble_candidates.append({"equal_ensemble_threshold": et, **summary, "safe_sensitivity": bool(float(summary["sensitivity"]) >= baseline_sens - 0.03)})
    ensemble_frame = pd.DataFrame(ensemble_candidates)
    ens_safe = ensemble_frame[ensemble_frame.safe_sensitivity]
    ens_selected = (ens_safe.sort_values(["fa_per_24h", "sensitivity"], ascending=[True, False]).iloc[0] if not ens_safe.empty else ensemble_frame.sort_values(["sensitivity", "fa_per_24h"], ascending=[False, True]).iloc[0]).to_dict()
    selected["equal_ensemble_threshold"] = float(ens_selected["equal_ensemble_threshold"])

    # Save selected action rows and patient-bootstrap intervals for the
    # development result.  These are not used to touch Eval.
    selected_thresholds = {k: float(selected[k]) for k in ("reject_probability", "expert_competence", "seizure_risk")}
    selected_rows = scorer._score_actions(labels, manifest, events, "router", selected_thresholds, ensemble_cache=ensemble_cache)
    methods_rows = {"best_fixed": fixed_rows[best_fixed.removeprefix("fixed_")], "router_no_reject": no_reject_rows, "router": selected_rows}
    summaries = [{"method": method, **scorer._summary(rows)} for method, rows in methods_rows.items()]
    bootstrap_frames = [scorer._bootstrap(rows, method, args.bootstrap, args.bootstrap_seed) for method, rows in methods_rows.items()]
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = []
    for method, g in bootstrap.groupby("method", sort=False):
        bootstrap_summary.append({"method": method, "sensitivity_ci_low": float(g.sensitivity.quantile(0.025)), "sensitivity_ci_high": float(g.sensitivity.quantile(0.975)), "fa_per_24h_ci_low": float(g.fa_per_24h.quantile(0.025)), "fa_per_24h_ci_high": float(g.fa_per_24h.quantile(0.975)), "mean_delay_s_ci_low": float(g.mean_delay_s.quantile(0.025)), "mean_delay_s_ci_high": float(g.mean_delay_s.quantile(0.975)), "bootstrap_replicates": int(len(g))})
    summary_frame = pd.DataFrame(summaries).merge(pd.DataFrame(bootstrap_summary), on="method", how="left")
    # Decision shares are useful for detecting policy collapse, but are not a
    # performance claim.
    action = labels[["router_competence_singlem", "router_competence_cbramod", "router_competence_tcn_gat", "router_reject_probability", "router_seizure_risk"]].to_numpy(float)
    chosen = action[:, :3].argmax(axis=1)
    allow_reject = (action[:, 3] >= float(selected["reject_probability"])) & (action[:, :3].max(axis=1) < float(selected["expert_competence"])) & (action[:, 4] < float(selected["seizure_risk"]))
    decision_counts = {"singlem": int((chosen == 0).sum()), "cbramod": int((chosen == 1).sum()), "tcn_gat": int((chosen == 2).sum()), "reject": int(allow_reject.sum()), "rows": int(len(labels)), "reject_share": float(allow_reject.mean())}
    OUT.mkdir(parents=True, exist_ok=False)
    labels.to_parquet(OUT / "dev_router_predictions.parquet", index=False)
    summary_frame.to_csv(OUT / "summary.csv", index=False)
    fixed_frame.to_csv(OUT / "fixed_baselines.csv", index=False)
    candidate_frame.to_csv(OUT / "threshold_candidates.csv", index=False)
    ensemble_frame.to_csv(OUT / "equal_ensemble_candidates.csv", index=False)
    bootstrap.to_csv(OUT / "patient_bootstrap.csv", index=False)
    (OUT / "threshold_selection.json").write_text(json.dumps(selected, indent=2, sort_keys=True, default=float) + "\n")
    manifest_out = {
        "release_id": LABEL_NS,
        "stage": "dev_only_scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eval_read": False,
        "threshold_source": "dev-only",
        "bootstrap_replicates": args.bootstrap,
        "labels_sha256": sha256(LABEL_PATH),
        "label_manifest_sha256": sha256(LABEL_MANIFEST),
        "checkpoint_sha256": sha256(CKPT),
        "expert_manifest_sha256": sha256(EXPERT_MANIFEST),
        "checkpoint_update": 5000,
        "model_training": "router_only",
        "input_normalization": "per-window channel median/IQR robust z-score clipped [-12,12]" if NORMALIZE_INPUT else "none",
        "selection_status": selection_status,
        "best_fixed": best_fixed,
        "selected_thresholds": {k: float(selected[k]) for k in ("reject_probability", "expert_competence", "seizure_risk")},
        "decision_counts": decision_counts,
        "ordinary_background_fallback": True,
        "note": "Development-only event scoring; TUSZ Eval intentionally excluded because prior H6-v1 exposed it.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "  manifest.json\n")
    print(json.dumps({"status": selection_status, "best_fixed": best_fixed, "selected": selected, "summary": summaries, "decision_counts": decision_counts, "out": str(OUT)}, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
