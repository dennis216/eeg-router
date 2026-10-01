"""Build immutable H6 causal-policy labels and Train/Dev audit statistics.

This is a labels-only pre-gate.  The router input is the last 10 s ending at
the decision time; frozen expert outcomes describe the following 60 s.  Eval
is intentionally not read.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("/root/b_false_alarm_atlas")
SOURCE_NS="h6-tusz-raw-pre-router-v1"
NS="h6-causal-policy-router-v2"
SOURCE_OUT=ROOT/"outputs/reports"/SOURCE_NS
OUT=ROOT/"outputs/reports"/NS
MODELS=("singlem","cbramod","tcn_gat")
PARTITIONS=("train","dev")
TAU_OUTCOME=.25
TAU_CONF=.25
CONTEXT_SECONDS=10.0
FUTURE_SECONDS=60.0

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def softmax(values:np.ndarray,tau:float)->np.ndarray:
    x=np.asarray(values,dtype=float)
    if x.size==1:return np.ones(1)
    x=x/tau; x-=float(np.nanmax(x)); e=np.exp(np.clip(x,-60,60))
    return e/e.sum() if e.sum()>0 else np.full(x.size,1/x.size)

def target(row:pd.Series):
    has=bool(int(row.future_has_seizure))
    c=np.asarray([int(row[f"correct_{m}"]) for m in MODELS],dtype=int)
    n=int(c.sum()); mask=str(row.correct_mask)
    if has:
        group="unroutable_seizure" if n==0 else ("seizure_all_correct" if n==3 else "seizure_selective")
    else:
        group="background_all_fp" if mask=="000" else ("background_all_safe" if mask=="111" else "background_selective")
    reject=group=="background_all_fp"; unroutable=group=="unroutable_seizure"
    eligible=group in {"seizure_selective","background_selective"}
    q=np.zeros(3,dtype=float)
    if eligible:
        ix=np.flatnonzero(c==1)
        outcome=np.asarray([1-min(max(float(row[f"delay_s_{m}"]),0),FUTURE_SECONDS)/FUTURE_SECONDS for m in MODELS])[ix] if has else np.ones(len(ix))
        conf=np.asarray([float(row[f"confidence_{m}"]) for m in MODELS])[ix]
        q[ix]=.90*softmax(outcome,TAU_OUTCOME)+.10*softmax(conf,TAU_CONF)
        q/=q.sum()
    return group,reject,unroutable,eligible,q

def build(partition:str, source_path:Path):
    src=pd.read_parquet(source_path)
    if src.empty or set(src.partition.astype(str))!={partition}: raise RuntimeError(f"bad source {source_path}")
    # v1 label parquet intentionally did not carry the EDF hash; its source
    # manifest hash and probability hashes below remain the provenance anchor.
    need={"sample_id","partition","patient_id","recording","cache_path","cache_row","future_start_s","future_end_s","future_has_seizure","correct_mask"}
    need|={f"{x}_{m}" for x in ("correct","confidence","delay_s","event_count","threshold","probability_sha256") for m in MODELS}
    missing=sorted(need-set(src.columns))
    if missing: raise RuntimeError(f"missing columns: {missing}")
    rows=[]
    for r in src.itertuples(index=False):
        s=pd.Series(r._asdict()); group,reject,unroutable,eligible,q=target(s)
        t=float(r.future_start_s); rec=str(r.recording)
        o={"sample_id":f"{partition}|{r.patient_id}|{rec}|{int(r.cache_row)}|causal10","source_sample_id_v1":str(r.sample_id),"partition":partition,"patient_id":str(r.patient_id),"recording":rec,"edf_sha256":str(s.get("edf_sha256", "")),"decision_index_v1":int(r.cache_row),"decision_time_s":t,"context_start_s":t-CONTEXT_SECONDS,"context_end_s":t,"future_start_s":t,"future_end_s":float(r.future_end_s),"future_has_seizure":int(bool(r.future_has_seizure)),"raw_cache_path_v2":str(Path("raw_cache_10s")/partition/Path(rec).with_suffix(".npz")),"source_cache_path_v1":str(r.cache_path),"correct_mask":str(r.correct_mask),"correct_count":int(sum(int(s[f"correct_{m}"]) for m in MODELS)),"outcome_group":group,"policy_eligible":int(eligible),"reject_label":int(reject),"unroutable":int(unroutable),"risk_label":int(bool(r.future_has_seizure))}
        for i,m in enumerate(MODELS):
            o.update({f"correct_{m}":int(s[f"correct_{m}"]),f"confidence_{m}":float(s[f"confidence_{m}"]),f"delay_s_{m}":float(s[f"delay_s_{m}"]),f"event_count_{m}":int(s[f"event_count_{m}"]),f"threshold_{m}":float(s[f"threshold_{m}"]),f"probability_sha256_{m}":str(s[f"probability_sha256_{m}"]),f"policy_target_{m}":float(q[i])})
        o["policy_target_argmax"]=MODELS[int(np.argmax(q))] if eligible else "none"
        o["policy_target_entropy"]=float(-np.sum(q[q>0]*np.log(q[q>0]))) if eligible else 0.0
        rows.append(o)
    f=pd.DataFrame(rows)
    if f.sample_id.duplicated().any(): raise RuntimeError(f"duplicate ids in {partition}")
    elig=f[f.policy_eligible==1]
    ps=f.groupby("patient_id").agg(rows=("sample_id","size"),policy_rows=("policy_eligible","sum"),reject_rows=("reject_label","sum"))
    audit={"partition":partition,"source_label":str(source_path.relative_to(ROOT)).replace("\\","/"),"source_label_sha256":sha256(source_path),"rows":int(len(f)),"patients":int(f.patient_id.nunique()),"records":int(f.recording.nunique()),"context_contract":{"seconds":CONTEXT_SECONDS,"sampling_hz":128,"channels":16,"samples":1280,"causal_end":"decision_time_s"},"future_horizon_seconds":FUTURE_SECONDS,"mask_counts":{str(k):int(v) for k,v in f.correct_mask.value_counts().sort_index().items()},"outcome_group_counts":{str(k):int(v) for k,v in f.outcome_group.value_counts().sort_index().items()},"policy_eligible_rows":int(f.policy_eligible.sum()),"reject_positive_rows":int(f.reject_label.sum()),"unroutable_rows":int(f.unroutable.sum()),"correct_counts":{m:int(f[f"correct_{m}"].sum()) for m in MODELS},"policy_target_mass_on_eligible":{m:float(elig[f"policy_target_{m}"].sum()) for m in MODELS},"policy_target_argmax_counts":{str(k):int(v) for k,v in elig.policy_target_argmax.value_counts().sort_index().items()},"policy_target_entropy_mean":float(elig.policy_target_entropy.mean()) if len(elig) else 0.0,"patient_support":{"patients_with_policy_rows":int((ps.policy_rows>0).sum()),"patients_with_reject_rows":int((ps.reject_rows>0).sum()),"rows_per_patient_median":float(ps.rows.median()),"policy_rows_per_patient_median":float(ps.policy_rows.median()),"policy_rows_per_patient_min":int(ps.policy_rows.min()),"policy_rows_per_patient_max":int(ps.policy_rows.max())},"router_input_excludes_expert_outputs":True,"eval_read":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    return f,audit

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--partitions",default="train,dev"); ap.add_argument("--source-dir",type=Path,default=SOURCE_OUT/"labels"); a=ap.parse_args()
    parts=tuple(x.strip() for x in a.partitions.split(",") if x.strip())
    if any(x not in PARTITIONS for x in parts): raise ValueError(parts)
    OUT.mkdir(parents=True,exist_ok=True); mp=OUT/"label_manifest.json"
    if mp.exists(): raise RuntimeError(f"refusing overwrite {mp}")
    audits={}; hashes={}
    for p in parts:
        sp=(a.source_dir/f"{p}_formal_fold0_seed17.parquet").resolve()
        op=OUT/"labels"/f"{p}_causal_policy.parquet"; auditp=OUT/"labels"/f"{p}_causal_policy_audit.json"
        if not sp.exists(): raise FileNotFoundError(sp)
        if op.exists() or auditp.exists(): raise RuntimeError(f"refusing overwrite {op}")
        frame,audit=build(p,sp); op.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(op,index=False); op.with_suffix(".sha256").write_text(sha256(op)+"\n"); auditp.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); auditp.with_suffix(".sha256").write_text(sha256(auditp)+"\n"); audits[p]=audit; hashes[p]=sha256(sp)
    manifest={"release_id":NS,"source_release_id":SOURCE_NS,"dataset":"TUSZ_v2.0.6 official train/dev","status":"labels_only_pre_gate","partitions":list(parts),"labels":{p:f"outputs/reports/{NS}/labels/{p}_causal_policy.parquet" for p in parts},"audits":{p:f"outputs/reports/{NS}/labels/{p}_causal_policy_audit.json" for p in parts},"source_labels":{p:f"outputs/reports/{SOURCE_NS}/labels/{p}_formal_fold0_seed17.parquet" for p in parts},"source_label_sha256":hashes,"decision_contract":{"input":"raw EEG only","channels":16,"sampling_hz":128,"context_seconds":CONTEXT_SECONDS,"context_samples":1280,"context_end":"decision_time_s","future_horizon_seconds":FUTURE_SECONDS,"stride_source":"frozen v1 decision anchors (60 seconds) pending gate","expert_outputs_in_router_input":False},"label_contract":{"policy":"seizure selective or background selective only","all_safe_background":"risk=0; no policy ranking loss","all_fp_background":"reject=1; no policy ranking loss","all_missed_seizure":"unroutable=1; policy target masked","multi_correct":"90% outcome utility + 10% calibrated confidence tie-breaker","confidence_is_offline_only":True},"eval_read":False,"old_results_preserved":True,"audits":audits,"created_utc":datetime.now(timezone.utc).isoformat()}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); mp.with_suffix(".sha256").write_text(sha256(mp)+"\n"); print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__": main()
