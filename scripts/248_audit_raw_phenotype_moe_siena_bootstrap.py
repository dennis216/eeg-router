"""Patient-level paired audit for the frozen raw-phenotype gate on Siena."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
R = ROOT / "outputs/reports"
IN = R / "h6-raw-phenotype-moe-v4"
OUT = R / "h6-raw-phenotype-moe-v4-audit-corrected3"
PRED = R / "siena-external-adapter-v1/siena_external_prediction_events.parquet"
TRUTH = R / "siena-external-adapter-v1/siena_external_truth_events.parquet"
RECORD = R / "siena-external-adapter-v1/siena_external_record_metrics.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def merge(iv):
    vals = sorted((float(a), float(b)) for a, b in iv if float(b) > float(a))
    if not vals: return []
    out = [vals[0]]
    for a, b in vals[1:]:
        if a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def one_patient(pred: pd.DataFrame, truth: pd.DataFrame, duration: float, gate_clusters: pd.DataFrame | None = None) -> dict[str, float]:
    truths = truth.drop_duplicates(["recording", "truth_index"])
    detected = 0
    false_sec = 0.0
    false_events = 0
    recordings = sorted(set(truth.recording) | set(pred.recording) | (set(gate_clusters.recording) if gate_clusters is not None else set()))
    for rec in recordings:
        tr = truths[truths.recording.eq(rec)]
        if gate_clusters is None:
            pr = pred[pred.recording.eq(rec)]
            alarm_iv = merge([(r.start_s, r.end_s) for r in pr.itertuples(index=False)])
            false_events += int(sum(1 for r in pr.itertuples(index=False) if not bool(getattr(r, "is_true_positive", False))))
        else:
            cc = gate_clusters[gate_clusters.recording.eq(rec)]
            alarm_iv = merge([(r.start_s, r.end_s) for r in cc[cc.gate_keep].itertuples(index=False)])
            false_events += int(sum(1 for r in cc[cc.gate_keep].itertuples(index=False) if not bool(r.has_truth)))
        truths_iv = merge([(r.start_s, r.end_s) for r in tr.itertuples(index=False)])
        for t in tr.itertuples(index=False):
            if any(min(float(b), float(t.end_s)) - max(float(a), float(t.start_s)) > 0 for a, b in alarm_iv): detected += 1
        for a, b in alarm_iv:
            overlap = sum(max(0.0, min(b, d) - max(a, c)) for c, d in truths_iv)
            false_sec += max(0.0, (b - a) - min(b - a, overlap))
    n_truth = len(truths)
    # false_sec/duration is a fraction of monitoring time; multiplying by
    # 1,440 converts it to minutes per 24 h (not seconds per 24 h).
    return {"truth_events": float(n_truth), "detected_truth_events": float(detected), "sensitivity": detected / max(1, n_truth), "false_alarm_events": float(false_events), "false_alarm_minutes_per_24h": false_sec / max(1.0, duration) * 1440.0, "false_alarm_minutes_raw": false_sec / 60.0}


def main():
    if (OUT / "manifest.json").exists(): raise RuntimeError(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(PRED); truth = pd.read_parquet(TRUTH); rec = pd.read_parquet(RECORD)
    gate = pd.read_parquet(IN / "siena_gate_decisions.parquet")
    gate["gate_keep"] = gate.gate_keep.astype(bool)
    durations = {(str(r.patient_id), str(r.recording)): float(r.duration_s) for r in rec.itertuples(index=False)}
    patients = sorted(set(pred.patient_id) | set(truth.patient_id))
    rows = []
    for patient in patients:
        p = pred[pred.patient_id.eq(patient)]; t = truth[truth.patient_id.eq(patient)]; g = gate[gate.patient_id.eq(patient)]
        row = {"patient_id": patient}
        for m in MODELS:
            score = one_patient(p[p.model.eq(m)], t, sum(durations.get((str(patient), str(r)), 0.0) for r in set(p.recording) | set(t.recording)))
            row.update({f"{k}_{m}": v for k, v in score.items()})
        score = one_patient(p, t, sum(durations.get((str(patient), str(r)), 0.0) for r in set(p.recording) | set(t.recording)), g)
        row.update({f"{k}_gate": v for k, v in score.items()})
        rows.append(row)
    pm = pd.DataFrame(rows); pm.to_csv(OUT / "siena_gate_patient_metrics.csv", index=False)
    rng = np.random.default_rng(20260902); boots = []
    for b in range(2000):
        sample = rng.choice(patients, size=len(patients), replace=True)
        z = pm.set_index("patient_id").loc[sample]
        boots.append({"replicate": b, "delta_sensitivity_gate_minus_cbramod_pp": float((z["sensitivity_gate"] - z["sensitivity_cbramod"]).mean() * 100), "delta_alarm_minutes_gate_minus_cbramod": float((z["false_alarm_minutes_per_24h_gate"] - z["false_alarm_minutes_per_24h_cbramod"]).mean()), "delta_fa_events_gate_minus_cbramod": float((z["false_alarm_events_gate"] - z["false_alarm_events_cbramod"]).mean()), "delta_sensitivity_gate_minus_tcn_pp": float((z["sensitivity_gate"] - z["sensitivity_tcn_gat"]).mean() * 100), "delta_alarm_minutes_gate_minus_tcn": float((z["false_alarm_minutes_per_24h_gate"] - z["false_alarm_minutes_per_24h_tcn_gat"]).mean())})
    boot = pd.DataFrame(boots); boot.to_parquet(OUT / "siena_gate_patient_bootstrap.parquet", index=False)
    def ci(col):
        q = boot[col].quantile([0.025, .5, .975]); return {"median": float(q.loc[.5]), "ci95_low": float(q.loc[.025]), "ci95_high": float(q.loc[.975])}
    audit = {"release_id": OUT.name, "status": "complete", "patients": len(patients), "bootstrap_replicates": 2000, "comparisons": {c: ci(c) for c in boot.columns if c != "replicate"}, "safety_rule": "sensitivity loss <=3 pp and lower false-alarm minutes/24h versus fixed comparator", "gate_threshold_source": "frozen TUSZ dev", "holdout_used_for_selection": False, "interpretation": "Siena is an untouched external holdout. The gate is safe only if the predeclared sensitivity and alarm-time criteria are both met; this audit does not retune it.", "source_hashes": {"gate_manifest": sha(IN / "manifest.json"), "prediction": sha(PRED), "truth": sha(TRUTH), "record": sha(RECORD)}}
    (OUT / "safety_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha(OUT / "manifest.json") + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()
