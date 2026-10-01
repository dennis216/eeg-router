"""H7 label audit and no-training oracle for patient-specific shared-FP suppression.

The script reads frozen expert probabilities.  It makes no model updates and
creates a new immutable namespace only when no formal output already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events

ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_NS = "h6-tusz-raw-pre-router-v1"
NS = "h7-personalized-consensus-fp-suppressor-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
PAIRS = tuple(combinations(MODELS, 2))
MIN_EVENTS, MIN_RECORDS, MIN_ORACLE = 20, 3, 0.20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def phase(t: float, total: float) -> str:
    return "fit" if t < .60 * total else "dev" if t < .80 * total else "holdout"


def overlap(a: Event, b: Event) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def f1(left: list[Event], right: list[Event]) -> float:
    used: set[int] = set(); n = 0
    for a in sorted(left, key=lambda x: x.start_s):
        choices = [j for j, b in enumerate(right) if j not in used and overlap(a, b) > 0]
        if choices:
            used.add(max(choices, key=lambda j: overlap(a, right[j]))); n += 1
    return 2 * n / (len(left) + len(right)) if left or right else 0.0


def shift(events: list[Event], duration: float, offset: float) -> list[Event]:
    result: list[Event] = []
    for e in events:
        width, start = min(e.end_s - e.start_s, duration), (e.start_s + offset) % duration
        end = start + width
        if end <= duration:
            result.append(Event(start, end, e.peak_probability))
        else:
            result += [Event(start, duration, e.peak_probability), Event(0., end - duration, e.peak_probability)]
    return result


def bh(values: list[float]) -> list[float]:
    order = np.argsort(values); out = np.empty(len(values)); running = 1.
    for rank in range(len(values), 0, -1):
        i = int(order[rank - 1]); running = min(running, values[i] * len(values) / rank); out[i] = running
    return out.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", default="train,dev,eval")
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args(); parts = tuple(x.strip() for x in args.partitions.split(",") if x.strip())
    out = ROOT / "outputs/reports" / NS
    if (out / "manifest.json").exists():
        raise RuntimeError(f"refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Chronology is reconstructed from the TUSZ session/date/file path; each
    # target patient receives a separate 60/20/20 history/dev/holdout split.
    recs: dict[tuple[str, str], dict[str, object]] = {}
    patient_recs: dict[tuple[str, str], list[dict[str, object]]] = {}
    for part in parts:
        mf = ROOT / "outputs/reports" / SOURCE_NS / "manifests" / f"{part}_manifest.json"
        for r in json.loads(mf.read_text())["records"]:
            patient_recs.setdefault((part, str(r["patient_id"])), []).append(r)
    for (part, patient), rows in patient_recs.items():
        offset, total = 0., sum(float(r["duration_s"]) for r in rows)
        for r in sorted(rows, key=lambda x: str(x["relative_edf"])):
            recs[(part, str(r["relative_edf"]))] = {"patient": patient, "offset": offset, "total": total, "duration": float(r["duration_s"]), "truth": [(float(x["start_s"]), float(x["end_s"])) for x in r["seizure_events"]]}
            offset += float(r["duration_s"])

    event_rows: list[dict[str, object]] = []; shared_rows: list[dict[str, object]] = []
    fp_index: dict[tuple[str, str], dict[str, list[Event]]] = {}
    hashes: dict[str, str] = {}
    for n, ((part, recording), info) in enumerate(recs.items(), 1):
        by_model: dict[str, list[Event]] = {}
        for model in MODELS:
            base = ROOT / "outputs/tuh_external/h6_tusz_raw_pre_router_v1" / part / model / "fold0_seed17"
            pp = base / "probabilities" / Path(recording).with_suffix(".parquet")
            mp = base / "metadata" / Path(recording).with_suffix(".json")
            table, meta = pd.read_parquet(pp).sort_values("window_end_s"), json.loads(mp.read_text())
            # Very short EDFs have fewer than two valid model windows.  They
            # cannot emit a valid event under the frozen eventization rule.
            events = eventize(table.window_end_s.to_numpy(float), table.probability.to_numpy(float), threshold=float(meta["threshold"])) if len(table) >= 2 else []
            matched = match_events(events, info["truth"], minimum_overlap_s=1.0)
            matched_ids = {x.prediction_index for x in matched.pairs}; by_model[model] = [events[i] for i in matched.unmatched_predictions]
            hashes[str(pp.relative_to(ROOT))] = str(meta["probability_sha256"])
            for i, e in enumerate(events):
                mid = float(info["offset"]) + (e.start_s + e.end_s) / 2
                event_rows.append({"partition": part, "patient_id": info["patient"], "recording": recording, "model": model, "start_s": e.start_s, "end_s": e.end_s, "phase": phase(mid, float(info["total"])), "is_fp": int(i not in matched_ids), "matched_seizure": int(i in matched_ids)})
        fp_index[(part, recording)] = by_model
        intervals: list[tuple[float, float, set[str]]] = []
        for a, b in PAIRS:
            for ea in by_model[a]:
                for eb in by_model[b]:
                    s, e = max(ea.start_s, eb.start_s), min(ea.end_s, eb.end_s)
                    if e > s: intervals.append((s, e, {a, b}))
        intervals.sort(key=lambda x: (x[0], x[1])); merged: list[tuple[float, float, set[str]]] = []
        for s, e, mask in intervals:
            if merged and s < merged[-1][1]:
                old = merged[-1]; merged[-1] = (old[0], max(old[1], e), old[2] | mask)
            else: merged.append((s, e, set(mask)))
        for s, e, mask in merged:
            mid = float(info["offset"]) + (s + e) / 2
            shared_rows.append({"partition": part, "patient_id": info["patient"], "recording": recording, "start_s": s, "end_s": e, "phase": phase(mid, float(info["total"])), "model_mask": "|".join(sorted(mask)), "model_count": len(mask), "label": "2of3_shared_fp_strict_interval_overlap"})
        if n % 250 == 0: print(json.dumps({"records": n, "total": len(recs)}), flush=True)

    events = pd.DataFrame(event_rows); shared = pd.DataFrame(shared_rows)
    events.to_parquet(out / "frozen_expert_events.parquet", index=False); shared.to_parquet(out / "shared_fp_events.parquet", index=False)
    support: list[dict[str, object]] = []
    for (part, patient), e in events.groupby(["partition", "patient_id"], sort=False):
        s = shared[(shared.partition == part) & (shared.patient_id == patient)]; fit = s[s.phase == "fit"]
        dev, test = e[e.phase == "dev"], e[e.phase == "holdout"]
        candidates = []
        for model in MODELS:
            d = dev[dev.model == model]; candidates.append((model, int(d.matched_seizure.sum()), int(d.is_fp.sum())))
        max_tp = max(x[1] for x in candidates); best = min([x for x in candidates if x[1] >= max_tp - 1], key=lambda x: (x[2], -x[1], x[0]))[0]
        b = test[test.model == best]; fp = b[b.is_fp == 1]; st = s[s.phase == "holdout"]
        suppress = sum(bool(((st.start_s < x.end_s) & (st.end_s > x.start_s) & st.model_mask.str.contains(best, regex=False)).any()) for x in fp.itertuples(index=False)) if len(st) else 0
        support.append({"partition": part, "patient_id": patient, "best_fixed_detector_dev": best, "fit_shared_events": len(fit), "fit_shared_records": fit.recording.nunique(), "dev_matched_events_best": int(dev[dev.model == best].matched_seizure.sum()), "holdout_fp_best": len(fp), "holdout_shared_fp_oracle_removed": int(suppress), "oracle_fa_reduction": float(suppress / len(fp)) if len(fp) else 0., "oracle_sensitivity_loss": 0.})
    summary = pd.DataFrame(support)
    candidates = summary[(summary.fit_shared_events >= MIN_EVENTS) & (summary.fit_shared_records >= MIN_RECORDS) & (summary.oracle_fa_reduction >= MIN_ORACLE)]
    tests: list[dict[str, object]] = []
    for r in candidates.itertuples(index=False):
        pair_results: list[dict[str, object]] = []
        part, patient = str(r.partition), str(r.patient_id)
        records = [(rec, i) for (p, rec), i in recs.items() if p == part and i["patient"] == patient]
        for pair in PAIRS:
            a, b = pair; observed_scores = []; null_scores = []; rng = np.random.default_rng(abs(hash((part, patient, pair, 20260830))) % 2**32)
            for rec, info in records:
                fit_end = min(float(info["duration"]), .60 * float(info["total"]) - float(info["offset"]))
                if fit_end <= 2: continue
                left = [x for x in fp_index[(part, rec)][a] if x.end_s <= fit_end]; right = [x for x in fp_index[(part, rec)][b] if x.end_s <= fit_end]
                observed_scores.append(f1(left, right));
                m = min(300., .45 * fit_end); offsets = rng.uniform(m, fit_end - m, size=args.permutations) if fit_end > 2*m else np.full(args.permutations, fit_end/2)
                null_scores.append(np.asarray([f1(left, shift(right, fit_end, float(o))) for o in offsets]))
            observed = float(np.mean(observed_scores)) if observed_scores else 0.; null = np.mean(np.vstack(null_scores), axis=0) if null_scores else np.zeros(args.permutations)
            pair_results.append({"partition": part, "patient_id": patient, "pair": "|".join(pair), "observed_f1": observed, "null_median_f1": float(np.median(null)), "null_q975_f1": float(np.quantile(null,.975)), "p_raw": (1+float(np.sum(null>=observed)))/(args.permutations+1)})
        q = bh([x["p_raw"] for x in pair_results])
        for x, qq in zip(pair_results, q, strict=True): x["q_fdr_3pairs"] = qq; x["significant"] = bool(qq < .05 and x["observed_f1"] > x["null_q975_f1"]); tests.append(x)
        if len(tests) % 60 == 0: print(json.dumps({"pair_tests":len(tests)}),flush=True)
    nulls = pd.DataFrame(tests); nulls.to_csv(out / "fit_pairwise_circular_shift.csv", index=False)
    sig = set(zip(nulls.loc[nulls.significant, "partition"], nulls.loc[nulls.significant, "patient_id"])) if len(nulls) else set()
    summary["fit_pair_sharing_significant"] = [(p, q) in sig for p,q in zip(summary.partition, summary.patient_id, strict=True)]
    summary["gate_train"] = (summary.fit_shared_events >= MIN_EVENTS) & (summary.fit_shared_records >= MIN_RECORDS) & (summary.oracle_fa_reduction >= MIN_ORACLE) & summary.fit_pair_sharing_significant
    summary.to_csv(out / "patient_support_and_oracle.csv", index=False)
    manifest = {"release_id": NS, "status": "feasibility_complete_no_training", "source_release": SOURCE_NS, "data": "TUSZ v2.0.6", "partitions": list(parts), "chronological_split": "patient-local 60% fit / 20% development / 20% holdout reconstructed from TUSZ session/date/file paths", "experts": "three frozen CHB-trained experts with CHB validation-only thresholds", "shared_label": "at least 2 of 3 frozen expert FP events, strict interval overlap, no >=1 s seizure overlap", "null": {"permutations":args.permutations,"method":"within-recording circular shift","positive":"observed F1 > null 97.5th percentile and BH-FDR q < .05 among 3 pairs"}, "training_gate": {"fit_shared_events_ge":MIN_EVENTS,"fit_records_ge":MIN_RECORDS,"holdout_oracle_fa_reduction_ge":MIN_ORACLE,"oracle_sensitivity_loss":0.}, "counts":{"patients":int(len(summary)),"shared_events":int(len(shared)),"eligible":int(summary.gate_train.sum())}, "source_probability_hashes":hashes,"created_utc":datetime.now(timezone.utc).isoformat()}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); (out / "manifest.sha256").write_text(digest(out / "manifest.json")+"\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__": main()
