"""Audit detector FP-mask structure and raw EEG phenotype similarity.

This is a read-only analysis of frozen detector outputs.  Siena is treated as
an untouched external holdout for mask composition and event-level structure;
no Siena labels or predictions are used to fit a scaler, prototype, policy or
threshold.  The TUSZ phenotype table is restricted to the already frozen
fit/dev contexts and is used only for a descriptive raw-phenotype/mask
association analysis.  No model is trained here.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "/root/b_false_alarm_atlas/src")
from bfa.preprocessing import signal


ROOT = Path("/root/b_false_alarm_atlas")
OUT_ROOT = ROOT / "outputs/reports"
OUT_NAME = "h6-mask-phenotype-holdout-v4"
OUT = OUT_ROOT / OUT_NAME
SIENA_BASE = ROOT / "outputs/reports/siena-external-adapter-v1"
SIENA_ROOT = Path("/mnt/d/TUH_EEG/Siena_Scalp_EEG/v1.0.0")
TUSZ_EVENTS = ROOT / "outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet"
TUSZ_FEAT = ROOT / "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-with-tp-v1/context_eeg10_with_tp.parquet"
TUSZ_INDEX = ROOT / "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v1/complete_context_index.parquet"
TUSZ_RAW_ROOT = ROOT / "outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
BIT = {m: 1 << i for i, m in enumerate(MODELS)}
FEATURES = [
    "rms_median_uv", "rms_iqr_uv", "spatial_rms_dispersion",
    "spatial_rms_max_min_log", "relative_delta_power", "relative_theta_power",
    "relative_alpha_power", "relative_beta_power", "relative_gamma_power",
    "high_low_ratio",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def siena_electrode_name(channel: str) -> str:
    name = channel.upper().strip()
    for prefix in ("EEG ", "EEG-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "01": "O1", "1": "O1"}
    return aliases.get(name, name)


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def union_clusters(rows: pd.DataFrame, collar_s: float = 10.0) -> pd.DataFrame:
    """Cluster FP intervals by direct overlap or onset collar within a record."""
    result: list[dict[str, object]] = []
    for (patient, recording), df in rows.groupby(["patient_id", "recording"], sort=False):
        records = df.sort_values(["start_s", "end_s", "model"]).to_dict("records")
        clusters: list[list[dict[str, object]]] = []
        for row in records:
            placed = False
            for cluster in clusters:
                # Cross-detector intervals can be very long; merging on any
                # overlap would collapse an entire recording into one alarm.
                # Use an onset collar for policy units, matching the frozen
                # event-sharing definition.
                if any(abs(float(row["start_s"]) - float(old["start_s"])) <= collar_s for old in cluster):
                    cluster.append(row); placed = True; break
            if not placed:
                clusters.append([row])
        for idx, cluster in enumerate(clusters):
            mask = 0
            for row in cluster:
                mask |= BIT[str(row["model"])]
            result.append({
                "patient_id": patient, "recording": recording,
                "cluster_index": idx, "mask": mask, "mask_bits": format(mask, "03b"),
                "start_s": min(float(r["start_s"]) for r in cluster),
                "end_s": max(float(r["end_s"]) for r in cluster),
                "detectors": ",".join(sorted({str(r["model"]) for r in cluster})),
                "member_events": len(cluster),
            })
    return pd.DataFrame(result)


def union_prediction_clusters(pred: pd.DataFrame, collar_s: float = 10.0) -> pd.DataFrame:
    """Cluster all frozen alarms so FP suppression can be scored for TP safety."""
    result: list[dict[str, object]] = []
    for (patient, recording), df in pred.groupby(["patient_id", "recording"], sort=False):
        records = df.sort_values(["start_s", "end_s", "model"]).to_dict("records")
        clusters: list[list[dict[str, object]]] = []
        for row in records:
            placed = False
            for cluster in clusters:
                if any(abs(float(row["start_s"]) - float(old["start_s"])) <= collar_s for old in cluster):
                    cluster.append(row); placed = True; break
            if not placed:
                clusters.append([row])
        for idx, cluster in enumerate(clusters):
            fp_mask = 0; tp_mask = 0; truth_indices: set[int] = set()
            for row in cluster:
                model = str(row["model"])
                is_tp = bool(row.get("is_true_positive", False))
                if is_tp:
                    tp_mask |= BIT[model]
                    if pd.notna(row.get("matched_truth_index")):
                        truth_indices.add(int(row["matched_truth_index"]))
                else:
                    fp_mask |= BIT[model]
            result.append({
                "patient_id": patient, "recording": recording, "cluster_index": idx,
                "mask": fp_mask, "mask_bits": format(fp_mask, "03b"), "fp_mask": fp_mask,
                "tp_mask": tp_mask, "tp_mask_bits": format(tp_mask, "03b"),
                "truth_indices": json.dumps(sorted(truth_indices)), "truth_count": len(truth_indices),
                "has_truth": int(bool(truth_indices)),
                "start_s": min(float(r["start_s"]) for r in cluster),
                "end_s": max(float(r["end_s"]) for r in cluster),
                "detectors": ",".join(sorted({str(r["model"]) for r in cluster})),
                "member_events": len(cluster),
            })
    return pd.DataFrame(result)


def siena_mask_windows(pred: pd.DataFrame, truth: pd.DataFrame, record: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = pred[(pred["is_true_positive"].fillna(False) == False)].copy()  # noqa: E712
    fp = fp[["patient_id", "recording", "start_s", "end_s", "model"]]
    clusters = union_clusters(fp, collar_s=10.0)
    truth_unique = truth.drop_duplicates(["patient_id", "recording", "truth_index"])
    rec = record.drop_duplicates(["patient_id", "recording"]).copy()
    windows: list[dict[str, object]] = []
    for rr in rec.itertuples(index=False):
        patient, recording = str(rr.patient_id), str(rr.recording)
        duration = float(rr.duration_s)
        obs_start = float(getattr(rr, "observation_start_s", 0.0) or 0.0)
        fps = fp[(fp.patient_id == patient) & (fp.recording == recording)]
        tr = truth_unique[(truth_unique.patient_id == patient) & (truth_unique.recording == recording)]
        fp_records = fps.to_dict("records")
        tr_records = tr.to_dict("records")
        stop = max(obs_start, duration - 10.0)
        for start in np.arange(obs_start, stop + 1e-6, 10.0):
            end = min(duration, float(start) + 10.0)
            seizure = int(any(interval_overlap(start, end, float(x["start_s"]), float(x["end_s"])) > 0 for x in tr_records))
            mask = 0
            for x in fp_records:
                if interval_overlap(start, end, float(x["start_s"]), float(x["end_s"])) > 0:
                    mask |= BIT[str(x["model"])]
            windows.append({
                "patient_id": patient, "recording": recording,
                "start_s": float(start), "end_s": float(end), "mask": mask,
                "mask_bits": format(mask, "03b"), "seizure": seizure,
            })
    return clusters, pd.DataFrame(windows)


def eeg10_one(x: np.ndarray) -> dict[str, float]:
    if x.shape != (16, 1280):
        raise ValueError(f"expected 16x1280, got {x.shape}")
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-8)
    rms_log = np.log1p(rms)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], d=1 / 128.0)
    valid = (f >= 0.5) & (f <= 45.0)
    total = spec[:, valid].sum(axis=1) + 1e-8
    out: dict[str, float] = {
        "rms_median_uv": float(np.median(rms)),
        "rms_iqr_uv": float(np.quantile(rms, .75) - np.quantile(rms, .25)),
        "spatial_rms_dispersion": float(np.std(rms_log)),
        "spatial_rms_max_min_log": float(np.max(rms_log) - np.min(rms_log)),
    }
    bands = {"delta": (.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    for name, (lo, hi) in bands.items():
        band = valid & (f >= lo) & (f < hi)
        out[f"relative_{name}_power"] = float(np.mean(spec[:, band].sum(axis=1) / total))
    out["high_low_ratio"] = float((out["relative_beta_power"] + out["relative_gamma_power"]) / (out["relative_delta_power"] + out["relative_theta_power"] + 1e-8))
    return out


def siena_raw_event_features(clusters: pd.DataFrame) -> pd.DataFrame:
    """Extract the same 10 EEG features for Siena FP clusters, without fitting anything."""
    if clusters.empty:
        return clusters.copy()
    signal._electrode_name = siena_electrode_name
    outputs: list[dict[str, object]] = []
    for recording, df in clusters.groupby("recording", sort=False):
        path = SIENA_ROOT / str(recording)
        if not path.exists():
            # Some old manifests use a path without the patient prefix.
            candidates = list(SIENA_ROOT.glob(f"*/{Path(str(recording)).name}"))
            if not candidates:
                raise FileNotFoundError(path)
            path = candidates[0]
        raw_uv, fs = signal.load_canonical_uv(path)
        target = 128.0
        if abs(fs - target) > 1e-6:
            raw_uv = resample_poly(raw_uv, 1, int(round(fs / target)), axis=-1).astype(np.float32)
        n = raw_uv.shape[1]
        for row in df.to_dict("records"):
            # 10-s causal window ending at the candidate onset; left-pad only
            # when an alarm starts before 10 seconds of usable data.
            end = int(round(float(row["start_s"]) * target))
            start = end - 1280
            z = np.zeros((16, 1280), dtype=np.float32)
            a, b = max(0, start), min(n, end)
            if b > a:
                z[:, (a - start):(b - start)] = raw_uv[:, a:b]
            values = eeg10_one(z)
            values.update({k: row[k] for k in ("patient_id", "recording", "cluster_index", "mask", "mask_bits", "start_s", "end_s", "member_events")})
            outputs.append(values)
        del raw_uv
    return pd.DataFrame(outputs)


def build_tusz_phenotype_mask() -> tuple[pd.DataFrame, dict[str, object]]:
    feat = pd.read_parquet(TUSZ_FEAT)
    feat = feat[feat["partition"] != "eval"].copy()
    # A context can have one row per detector.  Pool duplicate contexts and
    # retain the FP mask as the set of detector rows marked fp_m.  Seizure and
    # TP contexts remain safety-labelled but are not used as all-FP negatives.
    group_cols = ["patient_id", "recording", "context_end_s"]
    rows: list[dict[str, object]] = []
    for key, df in feat.groupby(group_cols, sort=False):
        mask = 0
        for m in df.loc[df.kind.eq("fp_m") & df.model.isin(MODELS), "model"].astype(str):
            mask |= BIT[m]
        seizure = int(df.kind.eq("seizure").any())
        tp_mask = 0
        for m in df.loc[df.kind.eq("tp_m") & df.model.isin(MODELS), "model"].astype(str):
            tp_mask |= BIT[m]
        # For a pure TN context the mask is explicitly 000.  If a FP row is
        # present, it determines the mask even if a TN row for another model is
        # also present.
        base = df.iloc[0]
        out = {c: float(df[c].mean()) for c in FEATURES}
        out.update({"patient_id": key[0], "recording": key[1], "context_end_s": float(key[2]), "mask": mask, "mask_bits": format(mask, "03b"), "seizure": seizure, "tp_mask": tp_mask, "n_rows": len(df), "phase": str(base.phase), "partition": str(base.partition)})
        rows.append(out)
    tab = pd.DataFrame(rows)
    train = tab[tab.partition.eq("train") & tab.phase.eq("fit")].copy()
    dev = tab[tab.partition.eq("dev") & tab.phase.eq("dev")].copy()
    scaler = StandardScaler().fit(train[FEATURES]) if len(train) else None
    result: dict[str, object] = {"train_groups": len(train), "dev_groups": len(dev), "train_patients": int(train.patient_id.nunique()), "dev_patients": int(dev.patient_id.nunique())}
    if scaler is not None and len(train) and len(dev):
        Xtr = scaler.transform(train[FEATURES]); Xdv = scaler.transform(dev[FEATURES])
        # Descriptive nearest-centroid mask association (classes with support).
        cents = {}
        for mask, sub in train.groupby("mask"):
            if len(sub) >= 5:
                cents[int(mask)] = Xtr[train["mask"].to_numpy() == mask].mean(axis=0)
        classes = sorted(cents)
        if classes:
            dist = np.stack([np.linalg.norm(Xdv - cents[c], axis=1) for c in classes], axis=1)
            pred = np.asarray(classes)[dist.argmin(axis=1)]
            result["centroid_classes"] = classes
            result["centroid_balanced_accuracy"] = float(balanced_accuracy_score(dev["mask"], pred)) if len(np.unique(dev["mask"])) > 1 else None
            # Binary same-vs-different mask separability using nearest train
            # neighbour.  This tests whether phenotype similarity tracks mask,
            # not whether a classifier is production-ready.
            nn = NearestNeighbors(n_neighbors=min(8, len(Xtr)), metric="euclidean").fit(Xtr)
            d, idx = nn.kneighbors(Xdv)
            same = []
            for i, neighbours in enumerate(idx):
                same.append(any(int(train.iloc[j]["mask"]) == int(dev.iloc[i]["mask"]) for j in neighbours))
            result["nearest_train_mask_match_rate"] = float(np.mean(same))
        # Raw waveform similarity for the same contexts, using the frozen
        # context cache only; no prototype is fit from dev/holdout.
    return tab, result


def raw_similarity_summary(tab: pd.DataFrame) -> dict[str, object]:
    """Compare raw-waveform nearest-neighbour distances for same/different masks."""
    index = pd.read_parquet(TUSZ_INDEX)
    index = index[index.partition != "eval"].copy()
    idx_map = index.set_index("context_id")["cache_path"].to_dict()
    # Reconstruct one context id per pooled group; use the feature table to
    # avoid inventing an identity for contexts that were already aggregated.
    feat = pd.read_parquet(TUSZ_FEAT)
    feat = feat[feat.partition != "eval"].copy()
    keys = ["patient_id", "recording", "context_end_s"]
    first = feat.drop_duplicates(keys).copy()
    first["group_key"] = list(zip(first.patient_id, first.recording, first.context_end_s))
    wanted = set(zip(tab.patient_id, tab.recording, tab.context_end_s))
    first = first[first.group_key.isin(wanted)]
    embeddings: list[np.ndarray] = []; labels: list[int] = []; parts: list[str] = []
    for r in first.itertuples(index=False):
        rel = idx_map.get(str(r.context_id))
        if rel is None:
            continue
        p = ROOT / str(rel)
        if not p.exists():
            continue
        try:
            with np.load(p) as z:
                x = np.asarray(z["x"], dtype=np.float32)
        except Exception:
            continue
        # Channel-wise z-normalized coarse waveform; absolute amplitude is
        # intentionally removed so this is a shape similarity check.
        x = x - x.mean(axis=1, keepdims=True)
        x = x / (x.std(axis=1, keepdims=True) + 1e-6)
        emb = x[:, ::40].reshape(-1)
        group_row = tab[(tab.patient_id == r.patient_id) & (tab.recording == r.recording) & (tab.context_end_s == r.context_end_s)].iloc[0]
        embeddings.append(emb); labels.append(int(group_row["mask"])); parts.append(str(r.partition))
    if len(embeddings) < 10:
        return {"raw_contexts_loaded": len(embeddings), "status": "insufficient"}
    E = np.stack(embeddings); labels_arr = np.asarray(labels); parts_arr = np.asarray(parts)
    tr = parts_arr == "train"; dv = parts_arr == "dev"
    if tr.sum() < 5 or dv.sum() < 1:
        return {"raw_contexts_loaded": int(len(E)), "status": "insufficient_split"}
    nn = NearestNeighbors(n_neighbors=min(16, int(tr.sum())), metric="euclidean").fit(E[tr])
    dist, ind = nn.kneighbors(E[dv])
    tr_labels = labels_arr[tr]; dv_labels = labels_arr[dv]
    same_d: list[float] = []; diff_d: list[float] = []; nearest_same: list[float] = []; nearest_diff: list[float] = []
    for i in range(len(dv_labels)):
        s = dist[i][tr_labels[ind[i]] == dv_labels[i]]
        d = dist[i][tr_labels[ind[i]] != dv_labels[i]]
        if len(s): same_d.append(float(s[0]))
        if len(d): diff_d.append(float(d[0]))
        nearest_same.append(float(s[0]) if len(s) else math.nan)
        nearest_diff.append(float(d[0]) if len(d) else math.nan)
    valid = np.isfinite(nearest_same) & np.isfinite(nearest_diff)
    return {
        "raw_contexts_loaded": int(len(E)), "train_raw_contexts": int(tr.sum()), "dev_raw_contexts": int(dv.sum()),
        "nearest_same_mask_distance_mean": float(np.mean(np.asarray(nearest_same)[valid])) if valid.any() else None,
        "nearest_different_mask_distance_mean": float(np.mean(np.asarray(nearest_diff)[valid])) if valid.any() else None,
        "nearest_same_mask_distance_median": float(np.median(np.asarray(nearest_same)[valid])) if valid.any() else None,
        "nearest_different_mask_distance_median": float(np.median(np.asarray(nearest_diff)[valid])) if valid.any() else None,
        "same_nearest_closer_fraction": float(np.mean(np.asarray(nearest_same)[valid] < np.asarray(nearest_diff)[valid])) if valid.any() else None,
        "status": "complete",
    }


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(SIENA_BASE / "siena_external_prediction_events.parquet")
    truth = pd.read_parquet(SIENA_BASE / "siena_external_truth_events.parquet")
    record = pd.read_parquet(SIENA_BASE / "siena_external_record_metrics.parquet")
    siena_clusters, siena_windows = siena_mask_windows(pred, truth, record)
    siena_all_clusters = union_prediction_clusters(pred, collar_s=10.0)
    siena_event_feat = siena_raw_event_features(siena_clusters)
    siena_all_event_feat = siena_raw_event_features(siena_all_clusters)
    tusz_tab, phenotype_summary = build_tusz_phenotype_mask()
    raw_summary = raw_similarity_summary(tusz_tab)
    # Keep all analysis tables in a new namespace; raw input files are never
    # modified.  The Siena grid contains 000 as well as all FP combinations,
    # while clusters summarize the event-level non-zero masks.
    siena_clusters.to_parquet(OUT / "siena_fp_clusters.parquet", index=False)
    siena_all_clusters.to_parquet(OUT / "siena_event_clusters.parquet", index=False)
    siena_windows.to_parquet(OUT / "siena_mask_windows.parquet", index=False)
    siena_event_feat.to_parquet(OUT / "siena_fp_cluster_eeg10.parquet", index=False)
    siena_all_event_feat.to_parquet(OUT / "siena_event_cluster_eeg10.parquet", index=False)
    tusz_tab.to_parquet(OUT / "tusz_fit_dev_phenotype_mask.parquet", index=False)
    mask_counts = siena_windows.groupby(["mask", "mask_bits"], as_index=False).agg(windows=("mask", "size"), patients=("patient_id", "nunique"))
    mask_counts["proportion"] = mask_counts.windows / max(1, len(siena_windows))
    mask_counts.to_csv(OUT / "siena_mask_counts.csv", index=False)
    # Patient bootstrap for the key [1,1,0]/[0,1,1]/all-FP proportions.
    rng = np.random.default_rng(20260901)
    pats = sorted(siena_windows.patient_id.unique()); boots = []
    for b in range(2000):
        sampled = rng.choice(pats, size=len(pats), replace=True)
        chunks = [siena_windows[siena_windows.patient_id == p] for p in sampled]
        z = pd.concat(chunks, ignore_index=True) if chunks else siena_windows.iloc[:0]
        boots.append({"replicate": b, **{f"mask_{m:03b}_proportion": float(np.mean(z["mask"].to_numpy() == m)) if len(z) else math.nan for m in range(8)}})
    pd.DataFrame(boots).to_parquet(OUT / "siena_mask_patient_bootstrap.parquet", index=False)
    summary = {
        "release_id": OUT_NAME, "status": "complete", "external_holdout": "Siena scalp EEG v1.0.0",
        "holdout_labels_used_for_fitting": False, "holdout_used_for_selection": False,
        "threshold_source": "frozen CHB validation-only expert thresholds", "mask_collar_seconds": 10.0,
        "grid_window_seconds": 10.0, "grid_stride_seconds": 10.0, "siena_patients": int(siena_windows.patient_id.nunique()),
        "siena_records": int(siena_windows.recording.nunique()), "siena_windows": int(len(siena_windows)), "siena_fp_clusters": int(len(siena_clusters)), "siena_event_clusters": int(len(siena_all_clusters)),
        "siena_sources": {"prediction_sha256": sha256(SIENA_BASE / "siena_external_prediction_events.parquet"), "truth_sha256": sha256(SIENA_BASE / "siena_external_truth_events.parquet"), "record_metrics_sha256": sha256(SIENA_BASE / "siena_external_record_metrics.parquet")},
        "siena_mask_counts": mask_counts.to_dict("records"), "tusz_phenotype_summary": phenotype_summary, "raw_waveform_similarity": raw_summary,
        "tusz_source_features_sha256": sha256(TUSZ_FEAT), "tusz_source_events_sha256": sha256(TUSZ_EVENTS), "tusz_source_index_sha256": sha256(TUSZ_INDEX),
        "patient_bootstrap_replicates": 2000, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpretation": "Siena mask composition is external descriptive evidence; TUSZ phenotype association is fit/dev-only and does not establish mechanism or a deployable policy.",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "manifest.sha256").write_text(sha256(OUT / "manifest.json") + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
