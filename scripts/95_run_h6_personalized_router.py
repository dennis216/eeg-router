"""H6 supervised within-patient personalized router.

This extension never retrains the three seizure detectors.  For every patient
in the frozen outer-test OOF cohort, complete EDF recordings are ordered and
split chronologically into calibration and holdout halves.  Calibration labels
may select a patient-specific model and phenotype-to-model map; holdout labels
are opened only for final scoring.

The current outer-test results were already inspected before H6 was proposed,
so this is explicitly an exploratory proof-of-concept rather than a pristine
new confirmatory test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bfa.evaluation.eventize import Event, eventize
from bfa.evaluation.match import match_events


ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_RELEASE = "v3-groupkfold-confirmatory-v1"
H6_ID = "h6-supervised-personalized-router-v1"
SOURCE = ROOT / "outputs" / "reports" / SOURCE_RELEASE
OUT = ROOT / "outputs" / "reports" / H6_ID
FEATURES_PATH = SOURCE / "h4_groupkfold_event_features.parquet"
OOF_PATH = SOURCE / "oof_probabilities_seed.parquet"
MODELS = ("singlem", "cbramod", "tcn_gat")
SEEDS = (17, 42, 3407)
FOLDS = tuple(range(5))
BUFFER_S = 30.0
MIN_PHENOTYPE_CALIBRATION_FP = 5
SENSITIVITY_TOLERANCE = 0.03
BOOTSTRAP_REPLICATES = 2000
RNG_SEED = 20260823
SIGNAL_FEATURES = (
    "event_duration_s", "peak_probability", "rms_median_uv", "rms_iqr_uv",
    "spatial_rms_dispersion", "spatial_rms_max_min_log", "line_noise_ratio",
    "clipping_fraction", "flat_fraction", "relative_delta_power",
    "relative_theta_power", "relative_alpha_power", "relative_beta_power",
    "relative_gamma_power", "high_low_ratio",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_h4_helpers():
    path = ROOT / "scripts" / "84_run_groupkfold_h4_router.py"
    spec = importlib.util.spec_from_file_location("h4_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def natural_record_key(row) -> tuple:
    source = str(row.source_case_id)
    numbers = tuple(int(x) for x in re.findall(r"\d+", str(row.recording_id)))
    return (source, numbers, str(row.recording_id))


def build_chronological_manifest(recordings: pd.DataFrame, cv: dict) -> dict:
    test_fold = {}
    for fold in cv["folds"]:
        for patient in fold["test"]:
            test_fold[str(patient)] = int(fold["fold"])
    patients = []
    for patient in sorted(test_fold):
        rows = list(recordings[recordings.patient_id.astype(str) == patient].itertuples(index=False))
        rows.sort(key=natural_record_key)
        if len(rows) < 2:
            raise RuntimeError(f"patient {patient} has fewer than two EDFs")
        durations = np.asarray([float(row.duration_s) for row in rows])
        total = float(durations.sum())
        candidates = range(1, len(rows))
        split_index = min(candidates, key=lambda i: abs(float(durations[:i].sum()) - total / 2.0))
        calibration = rows[:split_index]
        holdout = rows[split_index:]
        patients.append({
            "patient": patient,
            "fold": test_fold[patient],
            "ordering": "source_case_id_then_numeric_recording_id",
            "calibration_recordings": [str(r.recording_id) for r in calibration],
            "holdout_recordings": [str(r.recording_id) for r in holdout],
            "calibration_hours": float(sum(float(r.duration_s) for r in calibration) / 3600.0),
            "holdout_hours": float(sum(float(r.duration_s) for r in holdout) / 3600.0),
            "calibration_boundary_end_excluded_s": BUFFER_S,
            "holdout_boundary_start_excluded_s": BUFFER_S,
        })
    return {
        "h6_id": H6_ID,
        "source_release": SOURCE_RELEASE,
        "status": "frozen_before_holdout_scoring",
        "scenario": "within_patient_supervised_personalization",
        "claim_boundary": "not patient-independent generalization; exploratory because the parent outer-test results were previously inspected",
        "patient_count": len(patients),
        "split_fraction_target": 0.5,
        "whole_edf_assignment": True,
        "boundary_buffer_s": BUFFER_S,
        "patients": patients,
    }


def fit_phenotypes(features: pd.DataFrame, fold: int, selected_k: int):
    train = features[(features.split == fold) & (features.partition == "train")].copy()
    values = train.loc[:, SIGNAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    median = values.median().fillna(0.0)
    lower, upper = values.quantile(0.01), values.quantile(0.99)
    clean = values.fillna(median).clip(lower=lower, upper=upper, axis="columns")
    scaler = StandardScaler().fit(clean)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=RNG_SEED + fold).fit(scaler.transform(clean))
    points = pca.transform(scaler.transform(clean))
    clusterer = KMeans(n_clusters=selected_k, n_init=50, random_state=20260821 + fold).fit(points)
    return median, lower, upper, scaler, pca, clusterer


def assign_phenotype(frame: pd.DataFrame, fitted) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy(); result["phenotype"] = pd.Series(dtype=int); return result
    median, lower, upper, scaler, pca, clusterer = fitted
    values = frame.loc[:, SIGNAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    clean = values.fillna(median).clip(lower=lower, upper=upper, axis="columns")
    result = frame.copy()
    result["phenotype"] = clusterer.predict(pca.transform(scaler.transform(clean))).astype(int)
    return result


def interval_meta(prob: pd.DataFrame, records: list[str], calibration: bool, duration_map: dict[str, float]):
    starts, ends, patients = {}, {}, {}
    boundary = records[-1] if calibration else records[0]
    for recording in records:
        group = prob[prob.recording.astype(str) == recording]
        if group.empty:
            continue
        starts[recording] = float(group.end.min())
        ends[recording] = float(duration_map[recording])
        patients[recording] = str(group.patient.iloc[0])
    if calibration and boundary in ends:
        ends[boundary] = max(starts[boundary], ends[boundary] - BUFFER_S)
    if not calibration and boundary in starts:
        starts[boundary] = min(ends[boundary], starts[boundary] + BUFFER_S)
    return starts, ends, patients


def filter_events(frame: pd.DataFrame, starts: dict[str, float], ends: dict[str, float]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keep = [
        str(row.recording) in starts and float(row.start_s) >= starts[str(row.recording)] and float(row.end_s) <= ends[str(row.recording)]
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[keep].copy()


def truths_for(seizures: pd.DataFrame, recording: str, start: float, end: float):
    rows = seizures[seizures.recording_id.astype(str) == recording]
    return [(max(float(r.start_s), start), min(float(r.end_s), end)) for r in rows.itertuples(index=False) if float(r.end_s) > start and float(r.start_s) < end]


def score(events: pd.DataFrame, starts, ends, patients, seizures):
    tp = fp = truth_n = 0; delays=[]; nonseizure_h=0.0
    if events.empty or "event_start" not in events.columns:
        events = pd.DataFrame(columns=["patient", "recording", "event_start", "event_end", "peak_probability"])
    for recording in starts:
        truth = truths_for(seizures, recording, starts[recording], ends[recording])
        subset = events[events.recording.astype(str) == recording] if not events.empty else events
        preds=[Event(float(r.event_start),float(r.event_end),float(r.peak_probability)) for r in subset.sort_values("event_start").itertuples(index=False)]
        matched=match_events(preds,truth); tp += len(matched.pairs); fp += len(matched.unmatched_predictions); truth_n += len(truth)
        delays += [max(0.0,preds[p.prediction_index].start_s-truth[p.truth_index][0]) for p in matched.pairs]
        seizure_s=sum(max(0,b-a) for a,b in truth); nonseizure_h += max(0,ends[recording]-starts[recording]-seizure_s)/3600
    return {"tp":tp,"fp":fp,"truth":truth_n,"sensitivity":tp/truth_n if truth_n else 0.0,"fa_per_24h":fp*24/nonseizure_h if nonseizure_h else math.nan,"mean_delay_s":float(np.mean(delays)) if delays else math.nan,"nonseizure_h":nonseizure_h}


def model_events(frame: pd.DataFrame, model: str):
    return frame[frame.model == model].rename(columns={"start_s":"event_start","end_s":"event_end"})[["patient","recording","event_start","event_end","peak_probability"]].copy()


def choose_patient_model(calibration: pd.DataFrame, starts, ends, patients, seizures):
    scores={m:score(model_events(calibration,m),starts,ends,patients,seizures) for m in MODELS}
    max_s=max(v["sensitivity"] for v in scores.values())
    eligible=[m for m,v in scores.items() if v["sensitivity"] >= max_s-SENSITIVITY_TOLERANCE]
    selected=min(eligible,key=lambda m:(scores[m]["fa_per_24h"],-scores[m]["sensitivity"]))
    return selected,scores


def phenotype_map(calibration: pd.DataFrame, eligible: list[str], global_fallback: str, has_seizure: bool, k: int):
    mapping={}; detail=[]
    for phenotype in range(k):
        counts={m:int(len(calibration[(calibration.model==m)&(calibration.event_type=="fp")&(calibration.phenotype==phenotype)])) for m in MODELS}
        support=sum(counts.values())
        if has_seizure and support >= MIN_PHENOTYPE_CALIBRATION_FP:
            chosen=min(eligible,key=lambda m:(counts[m],m)); fallback=False; reason="patient_calibration_fp_minimization"
        else:
            chosen=global_fallback; fallback=True; reason="insufficient_support_or_no_calibration_seizure"
        mapping[phenotype]=chosen
        detail.append({"phenotype":phenotype,"selected_model":chosen,"fallback":fallback,"reason":reason,"support_fp_events":support,**{f"fp_{m}":counts[m] for m in MODELS}})
    return mapping,detail


def select_by_map(candidates: pd.DataFrame, mapping: dict[int,str], global_fallback: str):
    rows=[]
    for group_id,group in candidates.groupby("group_id",sort=True):
        preferred=[]
        for r in group.itertuples(index=False):
            if str(r.source_model)==mapping.get(int(r.phenotype),global_fallback): preferred.append(r)
        if not preferred:
            preferred=[r for r in group.itertuples(index=False) if str(r.source_model)==global_fallback]
        if not preferred: continue
        r=max(preferred,key=lambda x:float(x.peak_probability))
        rows.append({"patient":str(r.patient),"recording":str(r.recording),"event_start":float(r.start_s),"event_end":float(r.group_decision_end),"peak_probability":float(r.peak_probability),"selected_model":str(r.source_model),"phenotype":int(r.phenotype),"group_id":str(group_id)})
    columns = ["patient", "recording", "event_start", "event_end", "peak_probability", "selected_model", "phenotype", "group_id"]
    return pd.DataFrame(rows, columns=columns)


def combined_events(prob: pd.DataFrame, records: list[str], fold: int, seed: int, threshold: float, starts, ends):
    subset=prob[(prob.fold==fold)&(prob.seed==seed)&(prob.recording.isin(records))]
    wide=subset.pivot_table(index=["patient","recording","end"],columns="model",values="probability").dropna(subset=list(MODELS)).reset_index()
    wide["score"] = wide[list(MODELS)].mean(axis=1)
    rows=[]
    for recording,g in wide.groupby("recording"):
        g=g[(g.end>=starts[str(recording)])&(g.end<=ends[str(recording)])].sort_values("end")
        if g.empty: continue
        for e in eventize(g.end.to_numpy(float),g.score.to_numpy(float),threshold=threshold):
            rows.append({"patient":str(g.patient.iloc[0]),"recording":str(recording),"event_start":e.start_s,"event_end":e.end_s,"peak_probability":e.peak_probability})
    return pd.DataFrame(rows, columns=["patient", "recording", "event_start", "event_end", "peak_probability"])


def bootstrap(waterfall: pd.DataFrame):
    rng=np.random.default_rng(RNG_SEED); rows=[]
    patients=sorted(waterfall.patient.unique())
    for method in sorted(waterfall.method.unique()):
        d=waterfall[waterfall.method==method]
        for rep in range(BOOTSTRAP_REPLICATES):
            chosen=rng.choice(patients,len(patients),replace=True)
            sample=pd.concat([d[d.patient==p] for p in chosen],ignore_index=True)
            rows.append({"method":method,"replicate":rep,"sensitivity":sample.sensitivity.mean(),"fa_per_24h":sample.fa_per_24h.mean(),"mean_delay_s":sample.mean_delay_s.fillna(0).mean()})
    return pd.DataFrame(rows)


def main(smoke: bool):
    OUT.mkdir(parents=True,exist_ok=True)
    recordings=pd.read_parquet(ROOT/"manifests/recordings.parquet")
    seizures=pd.read_parquet(ROOT/"manifests/seizures.parquet")
    cv=json.loads((ROOT/"manifests/groupkfold_cv_v1/cv_manifest.json").read_text())
    manifest=build_chronological_manifest(recordings,cv)
    split_path=OUT/"chronological_split_manifest.json"
    split_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    features=pd.read_parquet(FEATURES_PATH); oof=pd.read_parquet(OOF_PATH)
    h2sel=pd.read_csv(SOURCE/"h2_groupkfold_cluster_selection.csv").set_index("fold")
    h4val=pd.read_csv(SOURCE/"h4_groupkfold_router_validation.csv")
    h4=load_h4_helpers(); duration_map=recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    if smoke:
        patient=manifest["patients"][0]; fold=int(patient["fold"]); k=int(h2sel.loc[fold,"selected_k"])
        fitted=fit_phenotypes(features,fold,k)
        subset=features[(features.split==fold)&(features.patient==patient["patient"])&features.recording.isin(patient["calibration_recordings"])]
        assigned=assign_phenotype(subset,fitted)
        result={"status":"smoke_passed_without_holdout_scoring","patient":patient["patient"],"fold":fold,"selected_k":k,"calibration_events":len(assigned),"phenotypes_present":sorted(assigned.phenotype.unique().tolist()),"split_manifest_sha256":sha256(split_path)}
        (OUT/"smoke.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2)); return
    result_rows=[]; map_rows=[]; patient_rows=[]
    phenotype_fits={f:fit_phenotypes(features,f,int(h2sel.loc[f,"selected_k"])) for f in FOLDS}
    for pinfo in manifest["patients"]:
        patient=str(pinfo["patient"]); fold=int(pinfo["fold"]); k=int(h2sel.loc[fold,"selected_k"])
        cal_records=pinfo["calibration_recordings"]; hold_records=pinfo["holdout_recordings"]
        # Process each seed independently; the chronological split is shared.
        for seed in SEEDS:
            base_prob=oof[(oof.patient==patient)&(oof.fold==fold)&(oof.seed==seed)]
            cal_starts,cal_ends,cal_patients=interval_meta(base_prob,cal_records,True,duration_map)
            hold_starts,hold_ends,hold_patients=interval_meta(base_prob,hold_records,False,duration_map)
            pf=features[(features.split==fold)&(features.seed==seed)&(features.patient==patient)]
            cal=assign_phenotype(filter_events(pf[pf.recording.isin(cal_records)],cal_starts,cal_ends),phenotype_fits[fold])
            hold=assign_phenotype(filter_events(pf[pf.recording.isin(hold_records)],hold_starts,hold_ends),phenotype_fits[fold])
            patient_model,cal_scores=choose_patient_model(cal,cal_starts,cal_ends,cal_patients,seizures)
            max_s=max(v["sensitivity"] for v in cal_scores.values()); eligible=[m for m,v in cal_scores.items() if v["sensitivity"]>=max_s-SENSITIVITY_TOLERANCE]
            global_row=h4val[(h4val.fold==fold)&(h4val.seed==seed)&(h4val.method=="best_single")].iloc[0]
            global_model=str(global_row.selected_model)
            has_cal_seizure=sum(len(truths_for(seizures,r,cal_starts[r],cal_ends[r])) for r in cal_starts)>0
            pmap,details=phenotype_map(cal,eligible,global_model,has_cal_seizure,k)
            for d in details: map_rows.append({"patient":patient,"fold":fold,"seed":seed,"patient_selected_model":patient_model,"global_fallback":global_model,"calibration_has_seizure":has_cal_seizure,**d})
            hold_candidates=h4._candidate_groups(hold)
            methods={
                "global_best_single":model_events(hold,global_model),
                "patient_specific_model":model_events(hold,patient_model),
                "patient_phenotype_router":select_by_map(hold_candidates,pmap,global_model),
            }
            equal_row=h4val[(h4val.fold==fold)&(h4val.seed==seed)&(h4val.method=="equal_weight_average")].iloc[0]
            methods["equal_weight_average"]=combined_events(oof,hold_records,fold,seed,float(equal_row.threshold),hold_starts,hold_ends)
            generic_row=h4val[(h4val.fold==fold)&(h4val.seed==seed)&(h4val.method=="router_model_aware")].iloc[0]
            generic_model=joblib.load(SOURCE/f"h4_router_model_aware_fold{fold}_seed{seed}.joblib")
            matrix,_=h4._router_matrix(hold_candidates,True)
            methods["generic_router"]=h4._router_select(hold_candidates,generic_model.predict_proba(matrix)[:,1],float(generic_row.threshold))
            for method,events in methods.items():
                metrics=score(events,hold_starts,hold_ends,hold_patients,seizures)
                result_rows.append({"patient":patient,"fold":fold,"seed":seed,"method":method,"test_evaluation_count":1,"holdout_labels_used_for_selection":False,**metrics})
            patient_rows.append({"patient":patient,"fold":fold,"seed":seed,"calibration_has_seizure":has_cal_seizure,"patient_selected_model":patient_model,"global_model":global_model,"eligible_models":"+".join(eligible),"calibration_events":len(cal),"holdout_events":len(hold),"fallback_phenotypes":sum(d["fallback"] for d in details),"phenotypes":k})
        print(f"H6_PATIENT_COMPLETE patient={patient} fold={fold}",flush=True)
    results=pd.DataFrame(result_rows); maps=pd.DataFrame(map_rows); coverage=pd.DataFrame(patient_rows)
    results.to_csv(OUT/"holdout_patient_metrics.csv",index=False); maps.to_csv(OUT/"phenotype_routing_maps.csv",index=False); coverage.to_csv(OUT/"calibration_coverage.csv",index=False)
    boot=bootstrap(results); boot.to_parquet(OUT/"patient_bootstrap.parquet",index=False)
    summary=results.groupby("method").agg(patients=("patient","nunique"),rows=("patient","size"),sensitivity_mean=("sensitivity","mean"),fa_per_24h_mean=("fa_per_24h","mean"),mean_delay_s=("mean_delay_s","mean")).reset_index()
    for metric in ("sensitivity","fa_per_24h","mean_delay_s"):
        q=boot.groupby("method")[metric].quantile([.025,.975]).unstack(); q.columns=[f"{metric}_q025",f"{metric}_q975"]; summary=summary.merge(q,on="method")
    summary.to_csv(OUT/"method_summary.csv",index=False)
    baseline=summary[summary.method=="global_best_single"].iloc[0]; primary=summary[summary.method=="patient_phenotype_router"].iloc[0]
    fa_reduction=1-float(primary.fa_per_24h_mean)/float(baseline.fa_per_24h_mean); sens_loss=float(baseline.sensitivity_mean)-float(primary.sensitivity_mean)
    gate={"fa_reduction_fraction":fa_reduction,"sensitivity_loss_pp":100*sens_loss,"pass":fa_reduction>=.25 and sens_loss<=.03,"criterion":"FA/24h reduction >=25% and sensitivity loss <=3pp versus global best single"}
    final={"h6_id":H6_ID,"status":"complete_exploratory_within_patient_personalization","source_release":SOURCE_RELEASE,"split_manifest_sha256":sha256(split_path),"source_hashes":{"features":sha256(FEATURES_PATH),"oof":sha256(OOF_PATH),"h4_validation":sha256(SOURCE/"h4_groupkfold_router_validation.csv")},"patients":manifest["patient_count"],"seeds":list(SEEDS),"holdout_evaluations_per_method":len(results)//results.method.nunique(),"primary_gate":gate,"parent_test_previously_inspected":True,"claim_boundary":"exploratory proof-of-concept; not a pristine confirmatory holdout and not patient-independent generalization","outputs":{"summary":sha256(OUT/"method_summary.csv"),"metrics":sha256(OUT/"holdout_patient_metrics.csv"),"routing_maps":sha256(OUT/"phenotype_routing_maps.csv"),"coverage":sha256(OUT/"calibration_coverage.csv"),"bootstrap":sha256(OUT/"patient_bootstrap.parquet")}}
    (OUT/"release_manifest.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print(summary.to_string(index=False)); print(json.dumps(final,indent=2,sort_keys=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); main(args.smoke)
