"""Phase A: pair-specific learned routing and validation complementarity.

Reuses the frozen v3 GroupKFold detector events/probabilities.  It never
trains a base detector and never imports the obsolete background-action audit.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
RUN_ID = "v3-groupkfold-confirmatory-v1"
SOURCE = ROOT / "outputs" / "reports" / RUN_ID
OUT = ROOT / "outputs" / "reports" / "pair-router-complementarity-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
FOLDS, SEEDS = tuple(range(5)), (17, 42, 3407)
N_BOOTSTRAP, RNG_SEED = 2000, 20260920


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pair_name(pair: tuple[str, str]) -> str:
    return "+".join(pair)


def candidates(h4, frame: pd.DataFrame, detectors: tuple[str, ...]) -> pd.DataFrame:
    selected = frame[frame.model.astype(str).isin(detectors)].copy()
    result = h4._candidate_groups(selected)
    excluded = set(MODELS) - set(detectors)
    for model in excluded:
        if not result.empty and ((result[f"group_peak_{model}"] != 0).any() or (result[f"group_uncertainty_{model}"] != 0).any()):
            raise RuntimeError(f"excluded detector {model} has nonzero group features")
        if not result.empty and (result.source_model.astype(str) == model).any():
            raise RuntimeError(f"excluded detector {model} contributed a candidate")
    return result


def combined_events(h4, tables: dict[str, pd.DataFrame], detectors: tuple[str, ...], weights: dict[str, float], threshold: float) -> pd.DataFrame:
    merged = None
    keys = ["patient", "recording", "end"]
    for model in detectors:
        current = tables[model].rename(columns={"probability": f"probability_{model}"})
        merged = current if merged is None else merged.merge(current, on=keys, how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability"])
    merged["score"] = sum(weights[m] * merged[f"probability_{m}"] for m in detectors)
    rows = []
    for recording, group in merged.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        for event in h4.eventize(ordered.end.to_numpy(float), ordered.score.to_numpy(float), threshold=float(threshold)):
            rows.append({"patient": str(ordered.patient.iloc[0]), "recording": str(recording), "event_start": float(event.start_s), "event_end": float(event.end_s), "peak_probability": float(event.peak_probability)})
    return pd.DataFrame(rows)


def choose_fusion(h4, tables, detectors, target, seizures, durations, starts, patients):
    best = None
    for w_a in np.arange(0.0, 1.0001, 0.25):
        weights = {model: 0.0 for model in MODELS}
        weights[detectors[0]], weights[detectors[1]] = float(w_a), float(1.0 - w_a)
        merged = None
        for model in detectors:
            f = tables[model].rename(columns={"probability": f"probability_{model}"})
            merged = f if merged is None else merged.merge(f, on=["patient", "recording", "end"], how="inner", validate="one_to_one")
        values = sum(weights[m] * merged[f"probability_{m}"] for m in detectors).to_numpy(float)
        thresholds = sorted(set(np.round(np.linspace(0.01, 0.99, 49), 3)) | set(np.round(np.quantile(values, np.linspace(0.50, 0.995, 20)), 3)))
        scored = []
        for threshold in thresholds:
            ev = combined_events(h4, tables, detectors, weights, float(threshold))
            scored.append((float(threshold), h4._score_events(ev, starts, patients, seizures, durations)))
        eligible = [x for x in scored if x[1]["sensitivity"] >= target - 0.03]
        pool = eligible if eligible else scored
        threshold, score = sorted(pool, key=lambda x: (x[1]["fa_per_24h"], -x[1]["sensitivity"], x[1]["mean_delay_s"] if math.isfinite(x[1]["mean_delay_s"]) else math.inf))[0]
        item = (weights, threshold, score)
        if best is None or (score["fa_per_24h"], -score["sensitivity"], score["mean_delay_s"] if math.isfinite(score["mean_delay_s"]) else math.inf) < (best[2]["fa_per_24h"], -best[2]["sensitivity"], best[2]["mean_delay_s"] if math.isfinite(best[2]["mean_delay_s"]) else math.inf):
            best = item
    return best


def direct_complementarity(h4, frame, starts, seizures, pair, fold, seed, partition):
    counts = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for recording, start in starts.items():
        rec = frame[frame.recording.astype(str) == str(recording)]
        for truth_start, truth_end in h4._truths(seizures, str(recording), float(start)):
            flags = []
            for model in pair:
                model_rows = rec[rec.model.astype(str) == model]
                flags.append(any(min(float(r.end_s), truth_end) - max(float(r.start_s), truth_start) >= 1.0 for r in model_rows.itertuples(index=False)))
            counts["both" if all(flags) else "a_only" if flags[0] else "b_only" if flags[1] else "neither"] += 1
    eligible = sum(counts.values()); union = eligible - counts["neither"]
    return {"pair": pair_name(pair), "model_a": pair[0], "model_b": pair[1], "fold": fold, "seed": seed, "partition": partition, "n_eligible": eligible, "n_both_detected": counts["both"], "n_a_only_detected": counts["a_only"], "n_b_only_detected": counts["b_only"], "n_neither_detected": counts["neither"], "union_detected": union, "union_coverage": union / eligible if eligible else math.nan, "unique_a_fraction_of_eligible": counts["a_only"] / eligible if eligible else math.nan, "unique_b_fraction_of_eligible": counts["b_only"] / eligible if eligible else math.nan, "unique_union_fraction_of_eligible": (counts["a_only"] + counts["b_only"]) / eligible if eligible else math.nan}


def add_fp_sharing(h1, record, frame, pair, patients):
    fp = frame[frame.event_type.astype(str) == "fp"]
    left = h1._event_index(fp[fp.model.astype(str) == pair[0]])
    right = h1._event_index(fp[fp.model.astype(str) == pair[1]])
    metric = h1._pair_metrics(left, right, tuple(sorted(set(patients))))
    matched = int(metric["matched_events"]); n_a = int(metric["left_events"]); n_b = int(metric["right_events"])
    record.update({"fp_temporal_f1_patient_median": metric["patient_median_f1"], "fp_temporal_f1_patient_mean": metric["patient_mean_f1"], "n_fp_a": n_a, "n_fp_b": n_b, "n_fp_temporally_matched": matched, "n_fp_a_only": n_a - matched, "n_fp_b_only": n_b - matched})
    return record


def patient_rows(h4, policies, starts, patients, seizures, durations, pair, fold, seed):
    rows = []
    for method, events in policies.items():
        frame = h4._patient_stats(events, starts, patients, seizures, durations)
        frame.insert(0, "method", method); frame.insert(0, "seed", seed); frame.insert(0, "fold", fold); frame.insert(0, "pair", pair_name(pair))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def paired_bootstrap(patient_df, pair, comparator):
    sub = patient_df[(patient_df.pair == pair_name(pair)) & patient_df.method.isin(["pair_router", comparator])]
    wide = sub.groupby(["patient", "method"])[["sensitivity", "fa_per_24h"]].mean().unstack("method").dropna()
    ds = wide[("sensitivity", "pair_router")] - wide[("sensitivity", comparator)]
    df = (wide[("fa_per_24h", "pair_router")] - wide[("fa_per_24h", comparator)]) / 24.0
    rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in pair_name(pair) + comparator))
    boot_s, boot_f = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(wide), len(wide)); boot_s.append(float(ds.iloc[idx].mean())); boot_f.append(float(df.iloc[idx].mean()))
    return {"pair": pair_name(pair), "comparison": f"pair_router_vs_{comparator}", "patients": len(wide), "bootstrap_replicates": N_BOOTSTRAP, "delta_sensitivity": float(ds.mean()), "delta_sensitivity_ci_low": float(np.quantile(boot_s, .025)), "delta_sensitivity_ci_high": float(np.quantile(boot_s, .975)), "delta_fa_per_h": float(df.mean()), "delta_fa_per_h_ci_low": float(np.quantile(boot_f, .025)), "delta_fa_per_h_ci_high": float(np.quantile(boot_f, .975))}


def sanity(h4, features, seizures, durations):
    canonical_val = pd.read_csv(SOURCE / "h4_groupkfold_router_validation.csv")
    canonical_test = pd.read_csv(SOURCE / "h4_groupkfold_router_test.csv")
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            test = features[(features.split == fold) & (features.seed == seed) & (features.partition == "test")].copy()
            starts, patients = h4._evaluation_meta(test); cand = candidates(h4, test, MODELS)
            for method, aware in (("router_model_aware", True), ("router_no_model_id", False)):
                vr = canonical_val[(canonical_val.fold == fold) & (canonical_val.seed == seed) & (canonical_val.method == method)].iloc[0]
                model = joblib.load(SOURCE / f"h4_{method}_fold{fold}_seed{seed}.joblib")
                matrix, _ = h4._router_matrix(cand, aware); scores = model.predict_proba(matrix)[:, 1]
                got = h4._score_events(h4._router_select(cand, scores, float(vr.threshold)), starts, patients, seizures, durations)
                expected = canonical_test[(canonical_test.fold == fold) & (canonical_test.seed == seed) & (canonical_test.method == method)].iloc[0]
                rows.append({"fold": fold, "seed": seed, "method": method, "sensitivity_recomputed": got["sensitivity"], "sensitivity_canonical": expected.sensitivity, "fa_per_24h_recomputed": got["fa_per_24h"], "fa_per_24h_canonical": expected.fa_per_24h, "abs_sensitivity_error": abs(got["sensitivity"] - expected.sensitivity), "abs_fa_error": abs(got["fa_per_24h"] - expected.fa_per_24h)})
    result = pd.DataFrame(rows); result.to_csv(OUT / "three_detector_sanity.csv", index=False)
    if result[["abs_sensitivity_error", "abs_fa_error"]].to_numpy().max() > 1e-10:
        raise RuntimeError("generalized three-detector path failed canonical parity")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--sanity-only", action="store_true"); args = parser.parse_args()
    start_time = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    h4 = load_module("h4_canonical", ROOT / "scripts" / "84_run_groupkfold_h4_router.py")
    h1 = load_module("h1_canonical", ROOT / "scripts" / "75_run_groupkfold_h1.py")
    features = pd.read_parquet(SOURCE / "h4_groupkfold_event_features.parquet")
    seizures = pd.read_parquet(ROOT / "manifests" / "seizures.parquet")
    recordings = pd.read_parquet(ROOT / "manifests" / "recordings.parquet")
    durations = recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    sanity_frame = sanity(h4, features, seizures, durations)
    if args.sanity_only:
        print(sanity_frame.groupby("method")[["sensitivity_recomputed", "fa_per_24h_recomputed"]].mean()); return
    result_rows, comp_val, comp_test, patient_frames, artifact_rows = [], [], [], [], []
    for pair in PAIRS:
        for fold in FOLDS:
            for seed in SEEDS:
                fs = features[(features.split == fold) & (features.seed == seed)].copy()
                train, validation, test = (fs[fs.partition == p].copy() for p in ("train", "validation", "test"))
                train_c = candidates(h4, train, pair); val_c = candidates(h4, validation, pair); test_c = candidates(h4, test, pair)
                val_starts, val_patients = h4._evaluation_meta(validation); test_starts, test_patients = h4._evaluation_meta(test)
                model_scores, model_thresholds = {}, {}
                for model in pair:
                    metrics = json.loads((ROOT / "runs" / RUN_ID / model / f"split{fold}_seed{seed}_main/validation/metrics_step_05000.json").read_text())
                    model_thresholds[model] = float(metrics["selected_event_operating_point"]["threshold"])
                    model_scores[model] = h4._score_events(h4._event_rows_from_model(validation, model), val_starts, val_patients, seizures, durations)
                target = max(x["sensitivity"] for x in model_scores.values())
                eligible = [(m, x) for m, x in model_scores.items() if x["sensitivity"] >= target - .03]
                best_model, _ = sorted(eligible, key=lambda x: (x[1]["fa_per_24h"], -x[1]["sensitivity"]))[0]
                val_tables = {m: h4._load_probability_table(fold, seed, m, "validation") for m in pair}
                weights, fusion_threshold, _ = choose_fusion(h4, val_tables, pair, target, seizures, durations, val_starts, val_patients)
                router, val_scores, thresholds, columns = h4._fit_router(train_c, val_c, target, True, split=fold, seed=seed)
                scored = [(float(t), h4._score_events(h4._router_select(val_c, val_scores, float(t)), val_starts, val_patients, seizures, durations)) for t in thresholds]
                pool = [x for x in scored if x[1]["sensitivity"] >= target - .03] or scored
                router_threshold, _ = sorted(pool, key=lambda x: (x[1]["fa_per_24h"], -x[1]["sensitivity"], x[1]["mean_delay_s"] if math.isfinite(x[1]["mean_delay_s"]) else math.inf))[0]
                artifact = OUT / f"pair_router_{pair[0]}_{pair[1]}_fold{fold}_seed{seed}.joblib"; joblib.dump(router, artifact)
                artifact_rows.append({"pair": pair_name(pair), "fold": fold, "seed": seed, "path": artifact.name, "sha256": sha256(artifact), "feature_count": len(columns)})
                test_tables = {m: h4._load_probability_table(fold, seed, m, "test") for m in pair}
                matrix, _ = h4._router_matrix(test_c, True); router_scores = router.predict_proba(matrix)[:, 1]
                policies = {"pair_best_single": h4._event_rows_from_model(test, best_model), "pair_fusion": combined_events(h4, test_tables, pair, weights, fusion_threshold), "pair_router": h4._router_select(test_c, router_scores, router_threshold)}
                scores = {m: h4._score_events(e, test_starts, test_patients, seizures, durations) for m, e in policies.items()}
                bs, fu, ro = scores["pair_best_single"], scores["pair_fusion"], scores["pair_router"]
                result_rows.append({"pair": pair_name(pair), "model_a": pair[0], "model_b": pair[1], "fold": fold, "seed": seed, "best_single_selected_model": best_model, "best_single_sensitivity": bs["sensitivity"], "best_single_fa_per_h": bs["fa_per_24h"] / 24, "best_single_mean_delay_s": bs["mean_delay_s"], "fusion_weight_a": weights[pair[0]], "fusion_weight_b": weights[pair[1]], "fusion_sensitivity": fu["sensitivity"], "fusion_fa_per_h": fu["fa_per_24h"] / 24, "fusion_mean_delay_s": fu["mean_delay_s"], "router_sensitivity": ro["sensitivity"], "router_fa_per_h": ro["fa_per_24h"] / 24, "router_mean_delay_s": ro["mean_delay_s"], "delta_router_vs_single_sensitivity": ro["sensitivity"] - bs["sensitivity"], "delta_router_vs_single_fa_per_h": (ro["fa_per_24h"] - bs["fa_per_24h"]) / 24, "delta_router_vs_fusion_sensitivity": ro["sensitivity"] - fu["sensitivity"], "delta_router_vs_fusion_fa_per_h": (ro["fa_per_24h"] - fu["fa_per_24h"]) / 24, "router_validation_threshold": router_threshold, "fusion_validation_threshold": fusion_threshold, "test_evaluation_count": 1, "threshold_source": "validation_only"})
                patient_frames.append(patient_rows(h4, policies, test_starts, test_patients, seizures, durations, pair, fold, seed))
                vc = direct_complementarity(h4, validation, val_starts, seizures, pair, fold, seed, "validation"); comp_val.append(add_fp_sharing(h1, vc, validation, pair, val_patients.values()))
                tc = direct_complementarity(h4, test, test_starts, seizures, pair, fold, seed, "test_descriptive_only"); comp_test.append(add_fp_sharing(h1, tc, test, pair, test_patients.values()))
                print(f"PAIR_DONE pair={pair_name(pair)} fold={fold} seed={seed}", flush=True)
    results = pd.DataFrame(result_rows); results.to_csv(OUT / "pair_router_fold_seed_results.csv", index=False)
    pd.DataFrame(comp_val).to_csv(OUT / "pair_complementarity_validation.csv", index=False); pd.DataFrame(comp_test).to_csv(OUT / "pair_complementarity_test_descriptive.csv", index=False)
    patient_df = pd.concat(patient_frames, ignore_index=True); patient_df.to_csv(OUT / "pair_router_patient_metrics.csv", index=False)
    boot = pd.DataFrame([paired_bootstrap(patient_df, p, c) for p in PAIRS for c in ("pair_best_single", "pair_fusion")]); boot.to_csv(OUT / "pair_router_bootstrap.csv", index=False)
    mean_cols = [c for c in results.columns if c.endswith(("sensitivity", "fa_per_h"))]
    summary = results.groupby("pair", as_index=False)[mean_cols].mean()
    for row in boot.itertuples(index=False):
        prefix = "router_vs_single" if row.comparison.endswith("pair_best_single") else "router_vs_fusion"
        mask = summary.pair == row.pair
        summary.loc[mask, f"{prefix}_delta_sensitivity_ci_low"] = row.delta_sensitivity_ci_low
        summary.loc[mask, f"{prefix}_delta_sensitivity_ci_high"] = row.delta_sensitivity_ci_high
        summary.loc[mask, f"{prefix}_delta_fa_per_h_ci_low"] = row.delta_fa_per_h_ci_low
        summary.loc[mask, f"{prefix}_delta_fa_per_h_ci_high"] = row.delta_fa_per_h_ci_high
    summary.to_csv(OUT / "pair_router_summary.csv", index=False)
    val_summary = pd.DataFrame(comp_val).groupby("pair", as_index=False).mean(numeric_only=True)
    pair_level = summary.merge(val_summary[["pair", "union_coverage", "unique_union_fraction_of_eligible", "fp_temporal_f1_patient_median"]], on="pair", how="left")
    pair_level.to_csv(OUT / "pair_level_comparison.csv", index=False)
    runtime = time.time() - start_time
    manifest = {"analysis_id": "pair-router-complementarity-v1", "status": "complete", "runtime_seconds": runtime, "base_detectors_retrained": False, "source_protocol": RUN_ID, "pairs": [list(x) for x in PAIRS], "folds": list(FOLDS), "seeds": list(SEEDS), "feature_schema": "fixed canonical 24-D; excluded detector peak/uncertainty/source slots are zero", "test_use": "one evaluation after validation locks; test complementarity descriptive only", "three_detector_sanity_max_abs_error": float(sanity_frame[["abs_sensitivity_error", "abs_fa_error"]].to_numpy().max()), "source_hashes": {"features": sha256(SOURCE / "h4_groupkfold_event_features.parquet"), "canonical_router": sha256(ROOT / "scripts" / "84_run_groupkfold_h4_router.py"), "h1": sha256(ROOT / "scripts" / "75_run_groupkfold_h1.py")}, "artifacts": artifact_rows}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = "# Pair router complementarity v1\n\nFrozen detector outputs were reused; no base detector was retrained. Pair groups were reconstructed before routing. Complementarity was measured on validation patients; test measurements are descriptive only. The fixed 24-D schema retained zero-valued slots for the excluded detector.\n\n" + pair_level.to_markdown(index=False) + "\n"
    (OUT / "summary.md").write_text(readme)
    print(pair_level.to_string(index=False)); print(json.dumps({"output": str(OUT), "runtime_seconds": runtime}, indent=2))


if __name__ == "__main__": main()
