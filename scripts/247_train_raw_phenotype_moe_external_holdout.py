"""Train a raw-phenotype FP suppressor/MoE on TUSZ and freeze it on Siena.

The gate sees only ten EEG-derived phenotype features from a causal raw
waveform window.  Frozen detector outputs are used to make multi-label FP
targets on TUSZ train, and TUSZ dev is used once to choose the safety
threshold.  Siena is an untouched external holdout: no Siena labels are used
to fit, select, or retune the gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/root/b_false_alarm_atlas")
REPORTS = ROOT / "outputs/reports"
OUT_NAME = "h6-raw-phenotype-moe-v4"
OUT = REPORTS / OUT_NAME
TUSZ_EVENTS = REPORTS / "h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet"
TUSZ_FEATURES = REPORTS / "h6-mask-phenotype-holdout-v4/tusz_fit_dev_phenotype_mask.parquet"
SIENA_BASE = REPORTS / "h6-mask-phenotype-holdout-v4"
SIENA_PRED = REPORTS / "siena-external-adapter-v1/siena_external_prediction_events.parquet"
SIENA_TRUTH = REPORTS / "siena-external-adapter-v1/siena_external_truth_events.parquet"
SIENA_RECORD = REPORTS / "siena-external-adapter-v1/siena_external_record_metrics.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
BIT = {m: 1 << i for i, m in enumerate(MODELS)}
FEATURES = [
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion", "spatial_rms_max_min_log",
    "relative_delta_power", "relative_theta_power", "relative_alpha_power",
    "relative_beta_power", "relative_gamma_power", "high_low_ratio",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clusters_from_events(events: pd.DataFrame, collar_s: float = 10.0) -> pd.DataFrame:
    """Make onset-collar policy units without merging long alarms by overlap."""
    rows: list[dict[str, object]] = []
    for (patient, recording), df in events.groupby(["patient_id", "recording"], sort=False):
        recs = df.sort_values(["start_s", "end_s", "model"]).to_dict("records")
        clusters: list[list[dict[str, object]]] = []
        for r in recs:
            put = False
            for c in clusters:
                if any(abs(float(r["start_s"]) - float(q["start_s"])) <= collar_s for q in c):
                    c.append(r); put = True; break
            if not put:
                clusters.append([r])
        for idx, c in enumerate(clusters):
            fp_mask = 0; tp_mask = 0; tids: set[int] = set()
            for r in c:
                model = str(r["model"])
                is_fp = bool(r.get("is_fp", False))
                if is_fp:
                    fp_mask |= BIT[model]
                else:
                    tp_mask |= BIT[model]
                    if pd.notna(r.get("matched_truth_index")):
                        tids.add(int(r["matched_truth_index"]))
            # frozen_expert_events uses matched_seizure rather than an index;
            # each TP-containing cluster is still a valid sensitivity unit.
            rows.append({
                "patient_id": patient, "recording": recording, "cluster_index": idx,
                "fp_mask": fp_mask, "mask": fp_mask, "mask_bits": format(fp_mask, "03b"),
                "tp_mask": tp_mask, "tp_mask_bits": format(tp_mask, "03b"),
                "truth_indices": json.dumps(sorted(tids)), "truth_count": len(tids),
                "has_truth": int(tp_mask != 0 or bool(tids)),
                "start_s": min(float(q["start_s"]) for q in c), "end_s": max(float(q["end_s"]) for q in c),
                "detectors": ",".join(sorted({str(q["model"]) for q in c})), "member_events": len(c),
            })
    return pd.DataFrame(rows)


def map_features(clusters: pd.DataFrame, features: pd.DataFrame, partition: str | None = None) -> pd.DataFrame:
    """Nearest same-record causal context feature for each event cluster."""
    f = features if partition is None else features[features.partition.eq(partition)]
    out = clusters.copy()
    values = []
    for r in out.itertuples(index=False):
        q = f[(f.patient_id == r.patient_id) & (f.recording == r.recording)]
        if q.empty:
            values.append(None); continue
        q = q.assign(_d=(q.context_end_s.astype(float) - float(r.start_s)).abs()).sort_values("_d")
        # Never use a context from a different temporal phase; a 40-s collar
        # is enough to account for the frozen event context convention.
        if float(q.iloc[0]._d) > 40.0:
            values.append(None)
        else:
            values.append(q.iloc[0][FEATURES].to_numpy(dtype=np.float32))
    out["feature_vector"] = values
    return out[out.feature_vector.notna()].reset_index(drop=True)


class RawPhenotypeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def multilabel_auc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) > 1:
            vals.append(float(roc_auc_score(y[:, j], p[:, j])))
    return float(np.mean(vals)) if vals else 0.5


def train_gate(train: pd.DataFrame, dev: pd.DataFrame, device: torch.device) -> tuple[RawPhenotypeGate, StandardScaler, pd.DataFrame, dict[str, object]]:
    scaler = StandardScaler().fit(np.stack(train.feature_vector.to_numpy()))
    Xtr = scaler.transform(np.stack(train.feature_vector.to_numpy())).astype("float32")
    Xdv = scaler.transform(np.stack(dev.feature_vector.to_numpy())).astype("float32")
    def targets(df: pd.DataFrame) -> np.ndarray:
        y = np.zeros((len(df), 4), dtype="float32")
        for i, r in enumerate(df.itertuples(index=False)):
            y[i, 0] = 1.0 if int(r.mask) & 1 else 0.0
            y[i, 1] = 1.0 if int(r.mask) & 2 else 0.0
            y[i, 2] = 1.0 if int(r.mask) & 4 else 0.0
            y[i, 3] = float(r.seizure)
        return y
    ytr = targets(train); ydv = targets(dev)
    pos = ytr.sum(axis=0); neg = len(ytr) - pos
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0).astype("float32")
    model = RawPhenotypeGate().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(3407))
    best = -math.inf; best_state = None; stale = 0; hist = []
    for epoch in range(60):
        model.train(); loss_sum = 0.0; n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); loss = crit(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            loss_sum += float(loss.detach()) * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(torch.from_numpy(Xdv).to(device))).cpu().numpy()
        auc = multilabel_auc(ydv, p)
        hist.append({"epoch": epoch + 1, "train_loss": loss_sum / max(1, n), "dev_macro_auroc": auc})
        if auc > best + 1e-4:
            best = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 8: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, scaler, pd.DataFrame(hist), {"best_dev_macro_auroc": best, "epochs_run": len(hist), "pos_counts": pos.tolist(), "neg_counts": neg.tolist(), "pos_weight": pw.tolist()}


def predict_gate(model: RawPhenotypeGate, scaler: StandardScaler, df: pd.DataFrame, device: torch.device) -> np.ndarray:
    X = scaler.transform(np.stack(df.feature_vector.to_numpy())).astype("float32")
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()


def policy_metrics(clusters: pd.DataFrame, probs: np.ndarray, tau_fp: float, tau_seizure: float) -> dict[str, float]:
    keep = []
    for i, r in enumerate(clusters.itertuples(index=False)):
        active = [j for j, bit in enumerate((1, 2, 4)) if int(r.fp_mask) & bit or int(r.tp_mask) & bit]
        if not active:
            keep.append(False); continue
        risk = float(probs[i, 3])
        safe_active = [j for j in active if float(probs[i, j]) < tau_fp]
        keep.append(bool(risk >= tau_seizure or safe_active))
    keep = np.asarray(keep, dtype=bool)
    truth_total = int(clusters.has_truth.sum())
    detected = int(clusters.loc[keep, "has_truth"].sum())
    false = int((keep & (clusters.has_truth.to_numpy() == 0)).sum())
    return {"tau_fp": tau_fp, "tau_seizure": tau_seizure, "kept_clusters": int(keep.sum()), "truth_units": truth_total, "detected_truth_units": detected, "sensitivity": detected / max(1, truth_total), "false_alarm_clusters": false, "gate_safety": int(detected / max(1, truth_total) >= 0.0)}


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    vals = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out = [vals[0]]
    for a, b in vals[1:]:
        if a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def interval_metrics(pred: pd.DataFrame, truth: pd.DataFrame, durations: dict[tuple[str, str], float] | None = None, keep_mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute event sensitivity and alarm minutes using raw intervals."""
    truth_u = truth.drop_duplicates(["patient_id", "recording", "truth_index"])
    total_truth = len(truth_u)
    kept = pred if keep_mask is None else pred.loc[keep_mask]
    detected = 0
    alarm_minutes = 0.0
    total_monitor = 0.0
    false_event_count = 0
    for (patient, recording), tr in truth_u.groupby(["patient_id", "recording"], sort=False):
        pr = kept[(kept.patient_id == patient) & (kept.recording == recording)]
        rr = pred[(pred.patient_id == patient) & (pred.recording == recording)]
        duration = float((durations or {}).get((patient, recording), max(rr.end_s.max() if len(rr) else 0.0, tr.end_s.max() if len(tr) else 0.0)))
        total_monitor += max(0.0, duration)
        for t in tr.itertuples(index=False):
            if any(max(0.0, min(float(p.end_s), float(t.end_s)) - max(float(p.start_s), float(t.start_s))) > 0 for p in pr.itertuples(index=False)):
                detected += 1
        alarms = merge_intervals([(float(p.start_s), float(p.end_s)) for p in pr.itertuples(index=False)])
        false_event_count += int(sum(1 for p in pr.itertuples(index=False) if not bool(getattr(p, "is_true_positive", False))))
        truths = merge_intervals([(float(t.start_s), float(t.end_s)) for t in tr.itertuples(index=False)])
        for a, b in alarms:
            covered = 0.0
            for c, d in truths: covered += max(0.0, min(b, d) - max(a, c))
            alarm_minutes += max(0.0, (b - a) - min(b - a, covered)) / 60.0
    return {"truth_events": float(total_truth), "detected_truth_events": float(detected), "sensitivity": detected / max(1, total_truth), "false_alarm_events": float(false_event_count), "false_alarm_events_per_24h": false_event_count / max(total_monitor, 1.0) * 86400.0, "false_alarm_minutes_per_24h": alarm_minutes / max(total_monitor, 1.0) * 86400.0, "alarm_minutes_raw": alarm_minutes, "monitor_seconds_proxy": total_monitor}


def main() -> None:
    if (OUT / "manifest.json").exists(): raise RuntimeError(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(TUSZ_EVENTS)
    feat = pd.read_parquet(TUSZ_FEATURES)
    # fit/dev labels are constructed from the already frozen TUSZ expert event
    # table; eval is never read by this training stage.
    train_feat = feat[feat.partition.eq("train") & feat.phase.eq("fit")].copy()
    dev_feat = feat[feat.partition.eq("dev") & feat.phase.eq("dev")].copy()
    # The v4 table stores scalar columns; reconstruct vectors explicitly.
    train_feat["feature_vector"] = list(train_feat[FEATURES].to_numpy(dtype=np.float32))
    dev_feat["feature_vector"] = list(dev_feat[FEATURES].to_numpy(dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, history, train_summary = train_gate(train_feat, dev_feat, device)
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "seed": 3407}, OUT / "gate_checkpoint.pt")
    history.to_csv(OUT / "training_history.csv", index=False)
    (OUT / "scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "features": FEATURES}, indent=2) + "\n")
    # Build policy units from event-level frozen expert outputs; map each unit
    # to a same-record feature in the corresponding train/dev partition.
    tr_ev = ev[ev.partition.eq("train")].copy(); dv_ev = ev[ev.partition.eq("dev")].copy()
    tr_ev["is_fp"] = tr_ev.is_fp.astype(bool)
    dv_ev["is_fp"] = dv_ev.is_fp.astype(bool)
    tr_clusters = map_features(clusters_from_events(tr_ev), feat, "train")
    dv_clusters = map_features(clusters_from_events(dv_ev), feat, "dev")
    tr_probs = predict_gate(model, scaler, tr_clusters, device) if len(tr_clusters) else np.zeros((0, 4))
    dv_probs = predict_gate(model, scaler, dv_clusters, device) if len(dv_clusters) else np.zeros((0, 4))
    # Fixed-detector cluster baseline and gate threshold search on dev only.
    fixed = []
    for j, name in enumerate(MODELS):
        active = (((dv_clusters["fp_mask"].to_numpy() & BIT[name]) != 0) | ((dv_clusters["tp_mask"].to_numpy() & BIT[name]) != 0))
        sens = float(dv_clusters.loc[active, "has_truth"].sum() / max(1, int(dv_clusters.has_truth.sum())))
        fa = int((active & (dv_clusters.has_truth.to_numpy() == 0)).sum())
        fixed.append({"method": name, "sensitivity": sens, "false_alarm_clusters": fa})
    best_fixed = max(fixed, key=lambda z: z["sensitivity"]) if fixed else {"method": "none", "sensitivity": 0.0, "false_alarm_clusters": 0}
    curve = []
    for tau_fp in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for tau_risk in (0.20, 0.40, 0.60, 0.80):
            m = policy_metrics(dv_clusters, dv_probs, tau_fp, tau_risk); m["safe_requirement"] = best_fixed["sensitivity"] - 0.03; m["meets_sensitivity"] = int(m["sensitivity"] >= m["safe_requirement"]); curve.append(m)
    curve_df = pd.DataFrame(curve); curve_df.to_csv(OUT / "tusz_dev_gate_curve.csv", index=False)
    eligible = curve_df[curve_df.meets_sensitivity.eq(1)]
    if eligible.empty:
        selected = {"status": "no_safe_dev_point", "tau_fp": None, "tau_seizure": None, "best_fixed": best_fixed}
    else:
        selected_row = eligible.sort_values(["false_alarm_clusters", "tau_fp", "tau_seizure"]).iloc[0]
        selected = {"status": "selected", "tau_fp": float(selected_row.tau_fp), "tau_seizure": float(selected_row.tau_seizure), "best_fixed": best_fixed, "dev_metrics": selected_row.to_dict()}
    (OUT / "threshold_freeze.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    # Evaluate one frozen policy on untouched Siena.  All raw event features
    # and truth intervals are already materialized by the independent audit;
    # they are only read after the threshold file is frozen.
    siena_clusters = pd.read_parquet(SIENA_BASE / "siena_event_clusters.parquet")
    siena_feat = pd.read_parquet(SIENA_BASE / "siena_event_cluster_eeg10.parquet")
    join = siena_clusters.merge(siena_feat[FEATURES + ["patient_id", "recording", "cluster_index"]], on=["patient_id", "recording", "cluster_index"], how="inner")
    join["feature_vector"] = list(join[FEATURES].to_numpy(dtype=np.float32))
    siena_probs = predict_gate(model, scaler, join.rename(columns={}), device) if len(join) else np.zeros((0, 4))
    if selected["status"] == "selected":
        tau_fp, tau_risk = float(selected["tau_fp"]), float(selected["tau_seizure"])
        keep = []
        for i, r in enumerate(join.itertuples(index=False)):
            active = int(r.fp_mask) != 0 or int(r.tp_mask) != 0
            keep.append(active and (float(siena_probs[i, 3]) >= tau_risk or any(float(siena_probs[i, j]) < tau_fp for j, bit in enumerate((1, 2, 4)) if int(r.fp_mask) & bit or int(r.tp_mask) & bit)))
        keep = np.asarray(keep, dtype=bool)
    else:
        keep = np.zeros(len(join), dtype=bool)
    truth = pd.read_parquet(SIENA_TRUTH); pred = pd.read_parquet(SIENA_PRED)
    record = pd.read_parquet(SIENA_RECORD)
    durations = {(str(r.patient_id), str(r.recording)): float(r.duration_s) for r in record.itertuples(index=False)}
    kept_pred = []
    for i, r in enumerate(join.itertuples(index=False)):
        if not keep[i]: continue
        for _, p in pred[(pred.patient_id == r.patient_id) & (pred.recording == r.recording) & (pred.start_s >= r.start_s) & (pred.start_s <= r.end_s)].iterrows(): kept_pred.append(p)
    kept_pred_df = pd.DataFrame(kept_pred) if kept_pred else pred.iloc[:0].copy()
    gate_score = interval_metrics(kept_pred_df, truth, durations=durations)
    base_rows = []
    for m in MODELS:
        pp = pred[pred.model.eq(m)].copy(); base_rows.append({"method": m, **interval_metrics(pp, truth, durations=durations)})
    base_rows.append({"method": "raw_phenotype_moe", **gate_score, "selected_tau_fp": selected["tau_fp"], "selected_tau_seizure": selected["tau_seizure"]})
    pd.DataFrame(base_rows).to_csv(OUT / "siena_metrics.csv", index=False)
    join["gate_keep"] = keep; join["gate_fp_prob_singlem"] = siena_probs[:, 0] if len(join) else []; join["gate_fp_prob_cbramod"] = siena_probs[:, 1] if len(join) else []; join["gate_fp_prob_tcn_gat"] = siena_probs[:, 2] if len(join) else []; join["gate_seizure_risk"] = siena_probs[:, 3] if len(join) else []
    join.to_parquet(OUT / "siena_gate_decisions.parquet", index=False)
    manifest = {"release_id": OUT_NAME, "status": "complete", "task": "raw EEG phenotype FP suppressor/MoE", "raw_only_features": FEATURES, "training_source": "TUSZ train fit contexts only", "selection_source": "TUSZ dev only", "external_holdout": "Siena scalp EEG v1.0.0", "holdout_used_for_fitting": False, "holdout_used_for_selection": False, "threshold_source": "TUSZ dev frozen gate; no Siena tuning", "train_summary": train_summary, "tusz_train_contexts": int(len(train_feat)), "tusz_dev_contexts": int(len(dev_feat)), "tusz_dev_clusters": int(len(dv_clusters)), "siena_clusters": int(len(join)), "siena_kept_clusters": int(keep.sum()), "siena_metrics": base_rows, "source_hashes": {"tusz_events": sha256(TUSZ_EVENTS), "tusz_features": sha256(TUSZ_FEATURES), "siena_prediction": sha256(SIENA_PRED), "siena_truth": sha256(SIENA_TRUTH), "siena_independent_audit": sha256(SIENA_BASE / "manifest.json")}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "interpretation": "A selected gate would be an external policy test; if no safe TUSZ-dev point exists, no Siena confirmatory policy claim is made."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
