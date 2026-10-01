"""H7-v2 exploratory gate using a literature-aligned ±10 s onset collar.

This is a new label definition after the strict-overlap feasibility audit had
insufficient patient support.  It reads fit/development rows only; no
chronological holdout is read, selected, or scored.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/root/b_false_alarm_atlas')
SOURCE=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1'
OUT=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle'
MODELS=('singlem','cbramod','tcn_gat'); PAIRS=tuple(combinations(MODELS,2))
TOL=10.; MIN_EVENTS=20; MIN_RECORDS=3; MIN_DEV_TRUTHS=2; MIN_DEV_FP=5; MIN_DEV_ORACLE=.20; N_NULL=1000

def f1(a,b):
    i=j=m=0; a=sorted(a); b=sorted(b)
    while i<len(a) and j<len(b):
        d=a[i]-b[j]
        if abs(d)<=TOL: m+=1;i+=1;j+=1
        elif d<0:i+=1
        else:j+=1
    return 2*m/(len(a)+len(b)) if a or b else 0.
def bh(v):
    order=np.argsort(v);out=np.empty(len(v));run=1.
    for rank in range(len(v),0,-1):
        i=int(order[rank-1]);run=min(run,v[i]*len(v)/rank);out[i]=run
    return out.tolist()
def truth_counts():
    out=[]
    for p in ('train','dev','eval'):
        m=ROOT/'outputs/reports/h6-tusz-raw-pre-router-v1/manifests'/f'{p}_manifest.json'
        by={}
        for r in json.loads(m.read_text())['records']:by.setdefault(str(r['patient_id']),[]).append(r)
        for patient,rs in by.items():
            rs=sorted(rs,key=lambda x:str(x['relative_edf'])); total=sum(float(x['duration_s']) for x in rs);offset=0.
            for r in rs:
                for e in r['seizure_events']:
                    mid=offset+(float(e['start_s'])+float(e['end_s']))/2; phase='fit' if mid<.6*total else 'dev' if mid<.8*total else 'holdout'
                    out.append({'partition':p,'patient_id':patient,'phase':phase})
                offset+=float(r['duration_s'])
    return pd.DataFrame(out).groupby(['partition','patient_id','phase']).size().rename('n').reset_index()
def main():
    if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    all_events=pd.read_parquet(SOURCE/'frozen_expert_events.parquet')
    events=all_events[all_events.phase.isin(['fit','dev'])].copy() # invariant: no holdout access below
    fp=events[events.is_fp==1].copy(); rows=[]
    # One-to-one pair matching within each EDF under an onset collar.  This
    # tolerates detector timing differences while retaining a causal 20-s raw
    # context ending at the later onset for the eventual specialist.
    for (part,patient,rec,phase),g in fp.groupby(['partition','patient_id','recording','phase'],sort=False):
        intervals=[]
        for a,b in PAIRS:
            la=sorted(g[g.model==a].start_s.tolist());lb=sorted(g[g.model==b].start_s.tolist());i=j=0
            while i<len(la) and j<len(lb):
                d=la[i]-lb[j]
                if abs(d)<=TOL:
                    t=max(la[i],lb[j]);intervals.append((t,{a,b}));i+=1;j+=1
                elif d<0:i+=1
                else:j+=1
        intervals.sort(key=lambda x:x[0])
        for t,mask in intervals:
            rows.append({'partition':part,'patient_id':patient,'recording':rec,'phase':phase,'decision_time_s':t,'context_start_s':max(0.,t-20.),'context_end_s':t,'model_mask':'|'.join(sorted(mask)),'label':'2of3_shared_fp_onset_pm10s'})
    shared=pd.DataFrame(rows); shared.to_parquet(OUT/'fit_dev_shared_fp_labels.parquet',index=False)
    tc=truth_counts(); summary=[]
    for (part,patient),g in events.groupby(['partition','patient_id'],sort=False):
        sh=shared[(shared.partition==part)&(shared.patient_id==patient)]; fit=sh[sh.phase=='fit'];dev=sh[sh.phase=='dev']; de=g[g.phase=='dev']
        nt=int(tc[(tc.partition==part)&(tc.patient_id==patient)&(tc.phase=='dev')].n.sum())
        scores=[]
        for m in MODELS:
            x=de[de.model==m];scores.append((m,int(x.matched_seizure.sum()),int(x.is_fp.sum())))
        mx=max(x[1] for x in scores); allow=[x for x in scores if nt and x[1]/nt>=mx/nt-.03]; best=min(allow,key=lambda x:(x[2],-x[1],x[0])) if allow else ('none',0,0)
        bfp=de[(de.model==best[0])&(de.is_fp==1)] if best[0]!='none' else de.iloc[:0];supp=0
        # Count a best-detector FP as oracle-removable only when *one same*
        # dev shared-FP label covers that FP and includes that detector.  The
        # former implementation used two independent ``any`` calls, which
        # could accidentally combine different labels and overstate the
        # oracle.  This is a development-only feasibility statistic.
        for x in bfp.itertuples(index=False):
            matched_label = dev[
                (dev.decision_time_s >= x.start_s - TOL)
                & (dev.decision_time_s <= x.end_s + TOL)
                & dev.model_mask.str.contains(best[0], regex=False)
            ] if len(dev) else dev
            supp += int(len(matched_label) > 0)
        summary.append({'partition':part,'patient_id':patient,'fit_shared_events':len(fit),'fit_shared_records':fit.recording.nunique(),'dev_truth_events':nt,'best_fixed_detector_dev':best[0],'dev_fp_best_fixed':best[2],'dev_oracle_removed':supp,'dev_oracle_fa_reduction':supp/best[2] if best[2] else 0.})
    s=pd.DataFrame(summary)
    # Circular-shift null is restricted to patients that already meet support
    # and development-oracle gates.  This is a label feasibility test, not a
    # model-selection test, and reads no holdout labels.
    candidates=s[(s.fit_shared_events>=MIN_EVENTS)&(s.fit_shared_records>=MIN_RECORDS)&(s.dev_truth_events>=MIN_DEV_TRUTHS)&(s.dev_fp_best_fixed>=MIN_DEV_FP)&(s.dev_oracle_fa_reduction>=MIN_DEV_ORACLE)]
    nullrows=[]
    for r in candidates.itertuples(index=False):
        for a,b in PAIRS:
            vals=[];obs=[];rng=np.random.default_rng(abs(hash((r.partition,r.patient_id,a,b,20260830)))%(2**32))
            g=fp[(fp.partition==r.partition)&(fp.patient_id==r.patient_id)&(fp.phase=='fit')]
            for rec,x in g.groupby('recording'):
                la=sorted(x[x.model==a].start_s.tolist());lb=sorted(x[x.model==b].start_s.tolist())
                if not la and not lb:continue
                dur=max(max(la,default=0),max(lb,default=0))+60.; m=min(300.,.45*dur); offsets=rng.uniform(m,dur-m,N_NULL) if dur>2*m else np.full(N_NULL,dur/2)
                obs.append(f1(la,lb));vals.append(np.array([f1(la,[(v+o)%dur for v in lb]) for o in offsets]))
            observed=float(np.mean(obs)) if obs else 0.; null=np.mean(np.vstack(vals),axis=0) if vals else np.zeros(N_NULL);p=(1+float((null>=observed).sum()))/(N_NULL+1)
            nullrows.append({'partition':r.partition,'patient_id':r.patient_id,'pair':f'{a}|{b}','observed_f1':observed,'null_median_f1':float(np.median(null)),'null_q975_f1':float(np.quantile(null,.975)),'p_raw':p})
    n=pd.DataFrame(nullrows)
    if len(n):
        n['q_fdr_3pairs']=n.groupby(['partition','patient_id'],group_keys=False).p_raw.transform(lambda x:bh(x.tolist()));n['significant']=(n.q_fdr_3pairs<.05)&(n.observed_f1>n.null_q975_f1);n.to_csv(OUT/'fit_pairwise_onset_collar_null.csv',index=False)
        sig=set(zip(n[n.significant].partition,n[n.significant].patient_id))
    else:sig=set()
    s['fit_pair_sharing_significant']=[(p,q) in sig for p,q in zip(s.partition,s.patient_id,strict=True)]
    s['gate_train']=(s.fit_shared_events>=MIN_EVENTS)&(s.fit_shared_records>=MIN_RECORDS)&(s.dev_truth_events>=MIN_DEV_TRUTHS)&(s.dev_fp_best_fixed>=MIN_DEV_FP)&(s.dev_oracle_fa_reduction>=MIN_DEV_ORACLE)&s.fit_pair_sharing_significant
    s.to_csv(OUT/'patient_dev_only_gate.csv',index=False)
    report={'release_id':OUT.name,'status':'exploratory_fit_dev_gate_no_training','source':'frozen H7 expert events','holdout_accessed':False,'shared_label':'two frozen expert FP event onsets within ±10 seconds; no seizure overlap','raw_input_contract':'20 s causal EEG ending at later FP onset; planned only if gate passes','gate':{'fit_events_ge':MIN_EVENTS,'fit_records_ge':MIN_RECORDS,'dev_truths_ge':MIN_DEV_TRUTHS,'dev_fp_ge':MIN_DEV_FP,'dev_oracle_reduction_ge':MIN_DEV_ORACLE,'pair_q_fdr_lt':.05},'counts':{'patients':len(s),'shared_fit_dev_events':len(shared),'eligible':int(s.gate_train.sum())},'created_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
