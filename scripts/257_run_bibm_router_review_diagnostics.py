"""Reviewer-requested posthoc diagnostics for the frozen standard H4 protocol.

No detector or router is fitted here.  Validation defines every threshold grid
and selected work point; the corresponding frozen threshold is then evaluated
on test once.  Test curve points are descriptive.  The component experiment
keeps either the learned or hindsight retained-group set fixed and varies only
the member-selection rule.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os
from dataclasses import dataclass
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ.get("BFA_ROOT", "/root/b_false_alarm_atlas"))
WORKSPACE_SCRIPTS = Path(os.environ.get("BFA_WORKSPACE_SCRIPTS", "/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/scripts"))
RUN = "v3-groupkfold-confirmatory-v1"
SOURCE = ROOT / "outputs/reports" / RUN
OUT = ROOT / "outputs/reports/bibm-router-review-diagnostics-v1"
ORACLE_OUT = ROOT / "outputs/reports/h4-protocol-matched-oracle-v2"
FINAL_AUDIT = ROOT / "outputs/reports/h4-final-linkage-expert-audit-v1"
MODELS = ("singlem", "cbramod", "tcn_gat")
METHODS = (*MODELS, "best_single", "equal_weight_average", "validation_fixed_weights",
           "router_model_aware", "router_no_model_id")
FA_BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SENS_TARGETS = (0.60, 0.70, 0.80)
SEEDS = (17, 42, 3407)
N_BOOT = 2000

@dataclass(frozen=True)
class FrozenMatchPair:
    prediction_index: int
    truth_index: int
    overlap_s: float

@dataclass(frozen=True)
class FrozenMatchResult:
    pairs: tuple[FrozenMatchPair, ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_truths: tuple[int, ...]

def frozen_h4_match_events(predictions, truths, minimum_overlap_s=1.0):
    """Exact maximum-total-overlap matcher used by the frozen H4 run."""
    if not predictions or not truths:
        return FrozenMatchResult((), tuple(range(len(predictions))), tuple(range(len(truths))))
    overlaps=np.zeros((len(predictions),len(truths)),dtype=float)
    for pi,prediction in enumerate(predictions):
        for ti,truth in enumerate(truths):
            overlaps[pi,ti]=max(0.0,min(prediction.end_s,truth[1])-max(prediction.start_s,truth[0]))
    rows,cols=linear_sum_assignment(-overlaps)
    pairs=tuple(FrozenMatchPair(int(r),int(c),float(overlaps[r,c])) for r,c in zip(rows,cols,strict=True)
                if overlaps[r,c] >= minimum_overlap_s)
    mp={p.prediction_index for p in pairs}; mt={p.truth_index for p in pairs}
    return FrozenMatchResult(pairs,
        tuple(i for i in range(len(predictions)) if i not in mp),
        tuple(i for i in range(len(truths)) if i not in mt))

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def script_path(name: str) -> Path:
    primary=ROOT/"scripts"/name
    fallback=WORKSPACE_SCRIPTS/name
    if primary.exists(): return primary
    if fallback.exists(): return fallback
    raise FileNotFoundError(f"script unavailable at {primary} or {fallback}")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def merge_intervals(xs):
    out=[]
    for a,b in sorted((max(0.,float(a)),float(b)) for a,b in xs if float(b)>max(0.,float(a))):
        if out and a <= out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return [(a,b) for a,b in out]

def interval_len(xs): return sum(b-a for a,b in merge_intervals(xs))

def intersection_len(a,b):
    a,b=merge_intervals(a),merge_intervals(b); i=j=0; total=0.
    while i<len(a) and j<len(b):
        total += max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] <= b[j][1]: i+=1
        else: j+=1
    return total

def empty_events():
    return pd.DataFrame(columns=["patient","recording","event_start","event_end","peak_probability","selected_model","group_id"])

def patient_metrics(events, starts, patients, seizures, durations, h4):
    rows=[]
    for patient in sorted(set(patients.values())):
        tp=fp=truth_n=0; bg_h=monitor_s=false_s=alarm_s=delay_sum=0.; delay_n=0
        for rec,p in patients.items():
            if p != patient: continue
            start=float(starts[rec]); end=float(durations[rec]); monitor_s += max(0.,end-start)
            truth=h4._truths(seizures,rec,start); truth_n += len(truth)
            tints=[(max(start,a),min(end,b)) for a,b in truth if min(end,b)>max(start,a)]
            bg_h += max(0.,end-start-interval_len(tints))/3600.
            g=events[events.recording.astype(str)==rec] if not events.empty else events
            pints=[(max(start,float(r.event_start)),min(end,float(r.event_end))) for r in g.itertuples(index=False)
                   if min(end,float(r.event_end))>max(start,float(r.event_start))]
            alarm_s += interval_len(pints); false_s += interval_len(pints)-intersection_len(pints,tints)
            preds=[h4.Event(float(r.event_start),float(r.event_end),float(r.peak_probability)) for r in g.itertuples(index=False)]
            m=h4.match_events(preds,truth); tp+=len(m.pairs); fp+=len(m.unmatched_predictions)
            for q in m.pairs: delay_sum+=max(0.,preds[q.prediction_index].start_s-truth[q.truth_index][0]); delay_n+=1
        rows.append(dict(patient=patient,tp=tp,fp=fp,truth=truth_n,nonseizure_hours=bg_h,monitoring_s=monitor_s,
                         false_alarm_s=false_s,alarm_s=alarm_s,delay_sum_s=delay_sum,delay_n=delay_n,
                         sensitivity=tp/truth_n if truth_n else 0.,fa_per_24h=fp*24/bg_h if bg_h else math.inf,
                         false_alarm_minutes_per_24h=false_s/monitor_s*1440 if monitor_s else math.inf,
                         alarm_minutes_per_24h=alarm_s/monitor_s*1440 if monitor_s else math.inf,
                         mean_delay_s=delay_sum/delay_n if delay_n else math.nan))
    return pd.DataFrame(rows)

def score(events, starts, patients, seizures, durations, h4, with_patients=False):
    # Keep the frozen matcher but avoid repeated pandas filtering inside the
    # original scorer. This is numerically identical to h4._score_events.
    grouped={str(rec):g for rec,g in events.groupby("recording",sort=False)} if not events.empty else {}
    true_positive=false_alarm=truth_count=0; nonseizure_hours=0.0; delays=[]
    for rec in patients:
        g=grouped.get(str(rec))
        pred=[h4.Event(float(r.event_start),float(r.event_end),float(r.peak_probability))
              for r in g.sort_values("event_start",kind="stable").itertuples(index=False)] if g is not None else []
        truth=h4._truths(seizures,str(rec),float(starts[rec])); matched=h4.match_events(pred,truth)
        true_positive += len(matched.pairs); false_alarm += len(matched.unmatched_predictions); truth_count += len(truth)
        delays.extend(max(0.0,pred[q.prediction_index].start_s-truth[q.truth_index][0]) for q in matched.pairs)
        nonseizure_hours += max(0.0,float(durations[rec])-float(starts[rec])-sum(max(0.0,b-a) for a,b in truth))/3600.0
    base=dict(true_positive_events=int(true_positive),false_alarm_events=int(false_alarm),truth_events=int(truth_count),
              sensitivity=float(true_positive/truth_count) if truth_count else 0.0,
              fa_per_24h=float(false_alarm*24.0/nonseizure_hours) if nonseizure_hours else math.inf,
              mean_delay_s=float(np.mean(delays)) if delays else math.nan,nonseizure_hours=nonseizure_hours)
    # Curve points only need pooled alarm-time metrics.  Patient-level matching
    # is reserved for frozen work points and the LL/LO/OL/OO diagnostic.
    pm=patient_metrics(events,starts,patients,seizures,durations,h4) if with_patients else None
    durations_s=(events.event_end-events.event_start).clip(lower=0).to_numpy(float) if not events.empty else np.array([])
    monitor=false=alarm=0.0
    for rec in patients:
        start=float(starts[rec]); end=float(durations[rec]); monitor += max(0.0,end-start)
        truth=h4._truths(seizures,rec,start)
        tints=[(max(start,a),min(end,b)) for a,b in truth if min(end,b)>max(start,a)]
        g=events[events.recording.astype(str)==rec] if not events.empty else events
        pints=[(max(start,float(r.event_start)),min(end,float(r.event_end))) for r in g.itertuples(index=False)
               if min(end,float(r.event_end))>max(start,float(r.event_start))]
        a=interval_len(pints); alarm += a; false += a-intersection_len(pints,tints)
    base.update(fa_per_h=base["fa_per_24h"]/24., monitoring_hours=monitor/3600.,
                false_alarm_minutes_per_24h=false/monitor*1440 if monitor else math.inf,
                alarm_minutes_per_24h=alarm/monitor*1440 if monitor else math.inf,
                median_alarm_duration_s=float(np.median(durations_s)) if len(durations_s) else 0.,
                p95_alarm_duration_s=float(np.quantile(durations_s,.95)) if len(durations_s) else 0.)
    return base,pm

def grid(values, original=None):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if not len(v): return np.array([-math.inf,math.inf])
    q=np.quantile(v,np.linspace(0,1,201)); eps=max(1e-12,np.finfo(float).eps*max(1.,np.max(np.abs(v)))*8)
    extra=[float(v.min()-eps),float(v.max()+eps)]
    if original is not None and math.isfinite(float(original)): extra.append(float(original))
    return np.unique(np.r_[q,extra])

def probability_table(fold,seed,model,partition):
    run=ROOT/f"runs/{RUN}/{model}/split{fold}_seed{seed}_main"
    p=run/("validation/probabilities_step_05000.parquet" if partition=="validation" else "test/probabilities.parquet")
    return pd.read_parquet(p)[["patient","recording","end","probability"]]

def single_events(table, threshold, h4):
    rows=[]
    for rec,g in table.groupby("recording",sort=False):
        g=g.sort_values("end",kind="stable")
        for e in h4.eventize(g.end.to_numpy(float),g.probability.to_numpy(float),threshold=float(threshold)):
            rows.append(dict(patient=str(g.patient.iloc[0]),recording=str(rec),event_start=e.start_s,event_end=e.end_s,peak_probability=e.peak_probability))
    return pd.DataFrame(rows) if rows else empty_events()

def router_material(candidates, fold, seed, method, h4):
    aware=method=="router_model_aware"; model=joblib.load(SOURCE/f"h4_{method}_fold{fold}_seed{seed}.joblib")
    x,_=h4._router_matrix(candidates,aware); member=model.predict_proba(x)[:,1]
    tmp=candidates[["group_id"]].copy(); tmp["s"]=member
    maxima=tmp.groupby("group_id",sort=True).s.max().to_numpy(float)
    return member,maxima

def original_row(validation,fold,seed,method):
    x=validation[(validation.fold==fold)&(validation.seed==seed)&(validation.method==method)]
    if len(x)!=1: raise RuntimeError(f"missing original row {fold=} {seed=} {method=}")
    return x.iloc[0]

def model_threshold(fold,seed,model):
    p=ROOT/f"runs/{RUN}/{model}/split{fold}_seed{seed}_main/validation/metrics_step_05000.json"
    return float(json.loads(p.read_text())["selected_event_operating_point"]["threshold"])

def selected_map(events):
    if events.empty or "group_id" not in events: return {}
    return {str(r.group_id):str(r.selected_model) for r in events.itertuples(index=False)}

def choose(curve, kind, target):
    if kind=="fa_budget":
        pool=curve[curve.fa_per_h<=target+1e-12]
        if pool.empty: return None
        return pool.sort_values(["sensitivity","false_alarm_minutes_per_24h","fa_per_h","threshold"],ascending=[False,True,True,False],kind="stable").iloc[0]
    pool=curve[curve.sensitivity>=target-1e-12]
    if pool.empty: return None
    return pool.sort_values(["fa_per_h","false_alarm_minutes_per_24h","sensitivity","threshold"],ascending=[True,True,False,False],kind="stable").iloc[0]

def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); q=np.empty(n); prev=1.
    for rank,idx in reversed(list(enumerate(order,1))): prev=min(prev,p[idx]*n/rank); q[idx]=prev
    return q

def learned_choice(candidates,scores,group_ids=None):
    f=candidates.copy(); f["router_score"]=scores
    if group_ids is not None: f=f[f.group_id.astype(str).isin(set(map(str,group_ids)))]
    rows=[]
    for gid,g in f.groupby("group_id",sort=True):
        r=g.sort_values(["router_score","peak_probability"],ascending=[False,False],kind="stable").iloc[0]
        rows.append(dict(patient=str(r.patient),recording=str(r.recording),event_start=float(r.start_s),event_end=float(r.group_decision_end),
                         peak_probability=float(r.peak_probability),selected_model=str(r.source_model),group_id=str(gid)))
    return pd.DataFrame(rows) if rows else empty_events()

def forced_selection(actions, oracle_selected):
    chosen={str(r.group_id):r for r in oracle_selected.itertuples(index=False)}; rows=[]
    for gid,g in actions.groupby("group_id",sort=True):
        gid=str(gid)
        r=chosen.get(gid)
        if r is None: r=g.sort_values("peak_probability",ascending=False,kind="stable").iloc[0]
        rows.append(dict(patient=str(r.patient),recording=str(r.recording),event_start=float(r.event_start),event_end=float(r.event_end),
                         peak_probability=float(r.peak_probability),selected_model=str(r.selected_model),group_id=gid))
    return pd.DataFrame(rows) if rows else empty_events()

def hindsight_fixed(actions, retained, starts, seizures, h4, oracle):
    selected=[]; assigned=[]; retained_ids=set(retained.group_id.astype(str))
    for rec,start in starts.items():
        a=actions[(actions.recording.astype(str)==rec)&actions.group_id.astype(str).isin(retained_ids)]
        o,aa=oracle.oracle_recording(a,h4._truths(seizures,rec,start)); selected.append(o); assigned+=aa
    out=pd.concat(selected,ignore_index=True) if selected else empty_events()
    used=set(out.group_id.astype(str)); fill=retained[~retained.group_id.astype(str).isin(used)]
    return pd.concat([out,fill],ignore_index=True),pd.DataFrame(assigned)

def bootstrap(primary_patients):
    rng=np.random.default_rng(20260925); rows=[]
    for budget,g in primary_patients.groupby("target"):
        roster=sorted(g.patient.unique()); vals=[]
        for _ in range(N_BOOT):
            sampled=rng.choice(roster,len(roster),replace=True); parts=[]
            for method in ("router_model_aware","validation_fixed_weights"):
                m=g[g.method==method]; run=[]
                for (fold,seed),r in m.groupby(["fold","seed"]):
                    rr=pd.concat([r[r.patient==p] for p in sampled],ignore_index=True)
                    run.append(rr.tp.sum()/rr.truth.sum() if rr.truth.sum() else 0.)
                parts.append(float(np.mean(run)))
            vals.append(parts[0]-parts[1])
        a=np.asarray(vals); point=float(a.mean()); p=2*min(float(np.mean(a<=0)),float(np.mean(a>=0)))
        rows.append(dict(target=budget,delta_sensitivity=point,ci025=np.quantile(a,.025),ci975=np.quantile(a,.975),p_value=min(1.,p)))
    out=pd.DataFrame(rows); out["q_bh"]=bh(out.p_value); return out

def main(smoke=False):
    OUT.mkdir(parents=True,exist_ok=True)
    prior=OUT/"manifest.json"
    if prior.exists():
        old=json.loads(prior.read_text())
        if not smoke and old.get("status")=="complete_posthoc_review_diagnostics":
            print(json.dumps({"status":"SKIP_EXACT_COMPLETE","manifest":str(prior)},indent=2)); return
    h4=load("h4",script_path("84_run_groupkfold_h4_router.py"))
    # The shared package changed to cardinality-first matching on 2026-09-19.
    # H4 predates that revision, so keep this analysis on its frozen scorer.
    h4.match_events=frozen_h4_match_events
    oracle=load("oracle",script_path("240_run_h4_protocol_matched_oracle.py"))
    required=[SOURCE/"h4_groupkfold_event_features.parquet",SOURCE/"h4_groupkfold_router_validation.csv",SOURCE/"h4_groupkfold_router_test.csv",
              ROOT/"manifests/seizures.parquet",ROOT/"manifests/recordings.parquet"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("missing formal inputs:\n"+"\n".join(missing))
    features=pd.read_parquet(required[0]); validation=pd.read_csv(required[1]); frozen=pd.read_csv(required[2])
    oracle_frozen=pd.read_csv(ORACLE_OUT/"policy_per_run.csv")
    forced_frozen=pd.read_csv(FINAL_AUDIT/"forced_and_full_oracle_per_run.csv")
    seizures=pd.read_parquet(required[3]); recordings=pd.read_parquet(required[4]); durations=recordings.set_index("recording_id").duration_s.astype(float).to_dict()
    runs=[(0,17)] if smoke else [(f,s) for f in range(5) for s in SEEDS]
    val_curves=[]; test_curves=[]; locks=[]; selected_rows=[]; patient_rows=[]; parity=[]; component=[]; component_patients=[]
    for fold,seed in runs:
        ff=features[(features.split==fold)&(features.seed==seed)]
        frames={p:ff[ff.partition.astype(str)==p].copy() for p in ("validation","test")}
        metas={p:h4._evaluation_meta(frames[p]) for p in frames}; candidates={p:h4._candidate_groups(frames[p]) for p in frames}
        tables={p:{m:probability_table(fold,seed,m,p) for m in MODELS} for p in frames}
        best=str(original_row(validation,fold,seed,"best_single").selected_model)
        # Abort this run before any sweep if the exact frozen policies diverge.
        policies=oracle.materialize_frozen_policies(frames["test"],fold,seed,validation,h4)
        for method,ev in policies.items():
            sc,_=score(ev,*metas["test"],seizures,durations,h4,with_patients=True); old=frozen[(frozen.fold==fold)&(frozen.seed==seed)&(frozen.method==method)].iloc[0]
            row=dict(fold=fold,seed=seed,method=method,tp_error=sc["true_positive_events"]-old.true_positive_events,
                     fp_error=sc["false_alarm_events"]-old.false_alarm_events,truth_error=sc["truth_events"]-old.truth_events,
                     sensitivity_abs_error=abs(sc["sensitivity"]-old.sensitivity),fa_abs_error=abs(sc["fa_per_24h"]-old.fa_per_24h))
            parity.append(row)
            if any(abs(float(row[k]))>1e-8 for k in ("tp_error","fp_error","truth_error","sensitivity_abs_error","fa_abs_error")): raise RuntimeError(f"headline parity failed: {row}")
        full,actions,assign=oracle.protocol_oracle(frames["test"],h4,seizures,metas["test"][0],MODELS)
        forced=forced_selection(actions,full)
        for method,ev,table,name in (("oracle_3",full,oracle_frozen,"oracle_3"),("forced_selection_oracle",forced,forced_frozen,"forced_selection_oracle")):
            sc,_=score(ev,*metas["test"],seizures,durations,h4,with_patients=True); old=table[(table.fold==fold)&(table.seed==seed)&((table.policy==name) if "policy" in table else True)].iloc[0]
            row=dict(fold=fold,seed=seed,method=method,tp_error=sc["true_positive_events"]-old.true_positive_events,
                     fp_error=sc["false_alarm_events"]-old.false_alarm_events,truth_error=sc["truth_events"]-old.truth_events,
                     sensitivity_abs_error=abs(sc["sensitivity"]-old.sensitivity),fa_abs_error=abs(sc["fa_per_24h"]-old.fa_per_24h))
            parity.append(row)
            if any(abs(float(row[k]))>1e-8 for k in ("tp_error","fp_error","truth_error","sensitivity_abs_error","fa_abs_error")): raise RuntimeError(f"oracle parity failed: {row}")
        specs={m:("single",m,None) for m in MODELS}; specs["best_single"]=("single",best,None)
        for method in ("equal_weight_average","validation_fixed_weights"):
            r=original_row(validation,fold,seed,method); specs[method]=("fusion",None,tuple(float(r[f"weight_{m}"]) for m in MODELS))
        router_data={p:{m:router_material(candidates[p],fold,seed,m,h4) for m in ("router_model_aware","router_no_model_id")} for p in frames}
        for method in METHODS:
            original=float(original_row(validation,fold,seed,method).threshold) if method not in MODELS else model_threshold(fold,seed,method)
            typ,model,weights=specs.get(method,("router",None,None))
            values=tables["validation"][model].probability if typ=="single" else (sum(w*tables["validation"][m].probability for w,m in zip(weights,MODELS)) if typ=="fusion" else router_data["validation"][method][1])
            thresholds=grid(values,original)
            partition_curves={}
            for part in ("validation","test"):
                starts,patients=metas[part]; rows=[]
                for threshold in thresholds:
                    if typ=="single": ev=single_events(tables[part][model],threshold,h4)
                    elif typ=="fusion": ev=h4._combined_events(tables[part],weights,threshold)
                    else: ev=h4._router_select(candidates[part],router_data[part][method][0],threshold)
                    sc,_=score(ev,starts,patients,seizures,durations,h4); rows.append(dict(fold=fold,seed=seed,partition=part,method=method,threshold=threshold,**sc))
                partition_curves[part]=pd.DataFrame(rows); (val_curves if part=="validation" else test_curves).append(partition_curves[part])
            for kind,targets in (("fa_budget",FA_BUDGETS),("sensitivity",SENS_TARGETS)):
                for target in targets:
                    pick=choose(partition_curves["validation"],kind,target)
                    if pick is None:
                        locks.append(dict(fold=fold,seed=seed,method=method,kind=kind,target=target,feasible=False,threshold=math.nan)); continue
                    threshold=float(pick.threshold); locks.append(dict(fold=fold,seed=seed,method=method,kind=kind,target=target,feasible=True,threshold=threshold))
                    row=partition_curves["test"].iloc[int(np.argmin(np.abs(partition_curves["test"].threshold-threshold)))].to_dict(); row.update(kind=kind,target=target)
                    selected_rows.append(row)
                    if kind=="fa_budget" and method in ("router_model_aware","validation_fixed_weights"):
                        if typ=="fusion": ev=h4._combined_events(tables["test"],weights,threshold)
                        else: ev=h4._router_select(candidates["test"],router_data["test"][method][0],threshold)
                        pm=patient_metrics(ev,*metas["test"],seizures,durations,h4); pm[["fold","seed","method","target"]]=fold,seed,method,target; patient_rows.append(pm)
        # LL/LO/OL/OO at the frozen model-aware point.
        c=candidates["test"]; member,_=router_data["test"]["router_model_aware"]; threshold=float(original_row(validation,fold,seed,"router_model_aware").threshold)
        ll=h4._router_select(c,member,threshold)
        lo,lo_assign=hindsight_fixed(actions,ll,metas["test"][0],seizures,h4,oracle); ol=learned_choice(c,member,full.group_id.astype(str))
        for condition,ev,a in (("LL",ll,None),("LO",lo,lo_assign),("OL",ol,None),("OO",full,assign)):
            ref=ll if condition in ("LL","LO") else full; sm,rm=selected_map(ev),selected_map(ref)
            changed=sum(sm.get(k)!=v for k,v in rm.items())
            sc,pm=score(ev,*metas["test"],seizures,durations,h4,with_patients=True); component.append(dict(fold=fold,seed=seed,condition=condition,assignments=len(a) if a is not None else math.nan,member_changes_vs_retention_reference=changed,**sc))
            pm[["fold","seed","condition"]]=fold,seed,condition; component_patients.append(pm)
    vc=pd.concat(val_curves,ignore_index=True); tc=pd.concat(test_curves,ignore_index=True); sr=pd.DataFrame(selected_rows); pr=pd.concat(patient_rows,ignore_index=True) if patient_rows else pd.DataFrame()
    pd.DataFrame(parity).to_csv(OUT/"headline_parity.csv",index=False); vc.to_csv(OUT/"validation_curve_points.csv",index=False); tc.to_csv(OUT/"test_curve_points_descriptive.csv",index=False)
    pd.DataFrame(locks).to_csv(OUT/"validation_workpoint_locks.csv",index=False); sr.to_csv(OUT/"test_workpoint_metrics.csv",index=False); pr.to_csv(OUT/"selected_workpoint_patient_metrics.csv",index=False)
    pd.DataFrame(component).to_csv(OUT/"component_metrics.csv",index=False); pd.concat(component_patients,ignore_index=True).to_csv(OUT/"component_patient_metrics.csv",index=False)
    if not smoke:
        boot=bootstrap(pr); boot.to_csv(OUT/"primary_patient_bootstrap.csv",index=False)
        summary=sr.groupby(["method","kind","target"],as_index=False)[["sensitivity","fa_per_h","false_alarm_minutes_per_24h"]].mean(); summary.to_csv(OUT/"workpoint_summary.csv",index=False)
        budgets=sr[sr.kind=="fa_budget"].copy(); budgets["budget_violated_on_test"]=budgets.fa_per_h>budgets.target+1e-12
        budgets.groupby(["method","target"],as_index=False).agg(runs=("fold","size"),violations=("budget_violated_on_test","sum"),violation_fraction=("budget_violated_on_test","mean"),mean_test_fa_h=("fa_per_h","mean"),mean_test_sensitivity=("sensitivity","mean")).to_csv(OUT/"budget_violation_table.csv",index=False)
        cs=pd.DataFrame(component).groupby("condition",as_index=False)[["sensitivity","fa_per_h","false_alarm_minutes_per_24h"]].mean(); cs.to_csv(OUT/"component_summary.csv",index=False)
        fig,ax=plt.subplots(figsize=(6.4,4.2));
        for method in ("best_single","validation_fixed_weights","router_model_aware","router_no_model_id"):
            g=tc[tc.method==method].groupby("threshold",as_index=False)[["fa_per_h","sensitivity"]].mean(); ax.plot(g.fa_per_h,g.sensitivity,".-",ms=2,label=method)
        ax.set(xlabel="False alarms per hour",ylabel="Event sensitivity"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/"froc_descriptive.png",dpi=220); plt.close(fig)
        b=summary[summary.kind=="fa_budget"]
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for method in ("best_single","validation_fixed_weights","router_model_aware","router_no_model_id"):
            g=b[b.method==method]; ax.plot(g.target,g.sensitivity,"o-",label=method)
        ax.set(xscale="log",xlabel="Validation FA/h budget",ylabel="Held-out event sensitivity"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/"validation_selected_budget_performance.png",dpi=220); plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        for method in ("validation_fixed_weights","router_model_aware","router_no_model_id"):
            g=tc[tc.method==method].groupby("threshold",as_index=False)[["false_alarm_minutes_per_24h","sensitivity"]].mean(); ax.plot(g.false_alarm_minutes_per_24h,g.sensitivity,".-",ms=2,label=method)
        ax.set(xlabel="False-alarm minutes per 24 h",ylabel="Event sensitivity"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/"alarm_time_tradeoff_descriptive.png",dpi=220); plt.close(fig)
        fig,ax=plt.subplots(figsize=(5.8,4.1)); ax.bar(cs.condition,cs.sensitivity,color=["#4C78A8","#72B7B2","#F58518","#54A24B"])
        ax.set(xlabel="Retained-group / member-selection condition",ylabel="Event sensitivity",ylim=(0,1)); fig.tight_layout(); fig.savefig(OUT/"component_diagnostic.png",dpi=220); plt.close(fig)
    else:
        summary=pd.DataFrame(); boot=pd.DataFrame(); cs=pd.DataFrame()
    report=("# BIBM router reviewer diagnostics\n\n"
            "This is a posthoc analysis of frozen held-out outputs. No detector or router was fitted. "
            "Validation selected thresholds; test curves are descriptive.\n\n"
            "## Reviewer mapping\n\n"
            "- The FROC and validation-selected work points address matched sensitivity/false-alarm comparisons.\n"
            "- LL/LO/OL/OO hold retained groups or member ranking fixed and quantify their conditional effects.\n"
            "- Forced-selection and full-oracle parity preserve the original interpretation of label-informed abstention.\n\n"
            "## Validation-selected work-point summary\n\n````text\n"+summary.to_string(index=False)+"\n````\n\n"
            "## Fixed-retention component summary\n\n````text\n"+cs.to_string(index=False)+"\n````\n\n"
            "## Primary patient bootstrap\n\n````text\n"+boot.to_string(index=False)+"\n````\n")
    (OUT/"reviewer_evidence_report.md").write_text(report)
    router_hashes={p.name:sha(p) for p in sorted(SOURCE.glob("h4_router_*_fold*_seed*.joblib"))}
    manifest=dict(analysis_id=OUT.name,status="smoke_complete" if smoke else "complete_posthoc_review_diagnostics",source_protocol=RUN,
                  detector_training=False,router_training=False,test_threshold_selection=False,runs=len(runs),bootstrap=N_BOOT if not smoke else 0,
                  input_hashes={p.name:sha(p) for p in required},router_artifact_hashes=router_hashes,
                  output_units={"fa_per_24h":"events/24 background hours","fa_per_h":"events/background hour","false_alarm_minutes_per_24h":"minutes/24 effective monitoring hours"},
                  outputs=sorted(p.name for p in OUT.iterdir() if p.is_file()))
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2)); print(pd.DataFrame(parity).groupby("method")[["sensitivity_abs_error","fa_abs_error"]].max())

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args(); main(args.smoke)
