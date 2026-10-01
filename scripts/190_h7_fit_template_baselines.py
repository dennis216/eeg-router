"""Fit-only recurrent-FP templates and calibrate dev suppressor gates.

This is the first non-neural H7 baseline.  Templates are fit independently
for each eligible patient-detector unit from early FP windows only.  A second
patient-level template uses the frozen 2-of-3 shared-FP candidate flag.  The
only dev operation is selection of a suppression cutoff subject to <=3pp
detector true-positive loss.  Holdout is never read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/root/b_false_alarm_atlas')
INDEX_DIR = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v2'
FEAT_DIR = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-features-v2'
ELIG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-template-baselines-v1'
FEATURES = ['rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log','relative_delta_power','relative_theta_power','relative_alpha_power','relative_beta_power','relative_gamma_power','high_low_ratio']
GRID = np.asarray([.5,.75,1.,1.25,1.5,2.,2.5,3.,4.,5.,6.,8.,10.], dtype=float)


def dist_to(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    if len(X) == 0 or len(P) == 0: return np.full(len(X), np.inf)
    return np.sqrt(((X[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def fit_template(pos: pd.DataFrame, neg: pd.DataFrame, scale_source: pd.DataFrame, prefix: str) -> dict:
    """Fit scaler, select eps by fit-only leave-one-EDF-out, and form prototypes."""
    if len(pos) < 3 or pos.recording.nunique() < 2:
        return {'status': 'not_enough_recurrent_candidates', 'prefix': prefix, 'n_pos': int(len(pos)), 'n_pos_recordings': int(pos.recording.nunique())}
    scaler = StandardScaler().fit(scale_source[FEATURES].to_numpy(dtype=float))
    xp = scaler.transform(pos[FEATURES].to_numpy(dtype=float)); xn = scaler.transform(neg[FEATURES].to_numpy(dtype=float)) if len(neg) else np.empty((0, len(FEATURES)))
    # Leave one EDF out: the held-out positive is compared only with FP
    # examples from other EDFs, and matched negatives are never used as refs.
    scores=[]; ys=[]
    for rec in sorted(pos.recording.unique()):
        ref = xp[pos.recording.to_numpy() != rec]
        if len(ref) == 0: continue
        pp = xp[pos.recording.to_numpy() == rec]
        nn = xn[neg.recording.to_numpy() == rec] if len(neg) else np.empty((0, len(FEATURES)))
        scores.extend(dist_to(pp, ref).tolist()); ys.extend([1] * len(pp))
        scores.extend(dist_to(nn, ref).tolist()); ys.extend([0] * len(nn))
    if len(set(ys)) == 2:
        best = None
        for eps in GRID:
            pred = (np.asarray(scores) <= eps).astype(int)
            f = float(f1_score(ys, pred, zero_division=0))
            candidate = (f, -float(eps))
            if best is None or candidate > best[0]: best = (candidate, eps)
        eps = float(best[1])
    else:
        eps = 3.0
    # DBSCAN is only a convenient connected-component implementation; it is
    # not a K-selection step.  Components must recur across >=2 EDFs and >=3
    # candidate events to become a template.
    labels = DBSCAN(eps=eps, min_samples=1, metric='euclidean').fit_predict(xp)
    prototypes=[]; components=[]
    for label in sorted(set(labels)):
        idx=np.flatnonzero(labels == label)
        recs=sorted(set(pos.iloc[idx].recording.astype(str)))
        if len(idx) >= 3 and len(recs) >= 2:
            prototypes.append(xp[idx].mean(axis=0)); components.append({'component': int(label), 'n_events': int(len(idx)), 'n_recordings': int(len(recs)), 'recordings': recs})
    prot=np.asarray(prototypes, dtype=float) if prototypes else np.empty((0, len(FEATURES)))
    return {'status': 'ok' if len(prototypes) else 'no_qualifying_component', 'prefix': prefix, 'scaler': scaler, 'prototypes': prot, 'eps': eps, 'n_pos': int(len(pos)), 'n_pos_recordings': int(pos.recording.nunique()), 'components': components, 'fit_loo_n': int(len(scores)), 'fit_loo_f1': float(max((f1_score(ys, np.asarray(scores) <= eps, zero_division=0), 0.0), default=0.0)) if len(ys) else None}


def score(template: dict, frame: pd.DataFrame, scaler_override=None) -> np.ndarray:
    if len(frame) == 0 or template.get('status') != 'ok': return np.full(len(frame), np.inf)
    scaler = scaler_override or template['scaler']; return dist_to(scaler.transform(frame[FEATURES].to_numpy(dtype=float)), template['prototypes'])


def single_gate(fp: np.ndarray, tp: np.ndarray) -> tuple[float | None, dict]:
    finite = np.unique(fp[np.isfinite(fp)]); finite = np.unique(np.concatenate([finite, GRID]))
    best = None
    for cut in finite:
        sf = int(np.sum(fp <= cut)); st = int(np.sum(tp <= cut)); loss = st / max(len(tp), 1); red = sf / max(len(fp), 1)
        if loss <= .03 + 1e-12:
            cand=(red, -loss, -float(cut))
            if best is None or cand > best[0]: best=(cand,float(cut),sf,st,loss,red)
    if best is None: return None, {'status':'no_safe_dev_cutoff','n_fp':int(len(fp)),'n_tp':int(len(tp))}
    _,cut,sf,st,loss,red=best
    return cut, {'status':'safe','cutoff':cut,'n_fp':int(len(fp)),'n_tp':int(len(tp)),'suppressed_fp':sf,'suppressed_tp':st,'fp_fraction_suppressed':red,'tp_loss':loss}


def dual_gate(fp_s: np.ndarray, fp_h: np.ndarray, tp_s: np.ndarray, tp_h: np.ndarray) -> tuple[tuple[float|None,float|None],dict]:
    cs=np.unique(fp_s[np.isfinite(fp_s)]); ch=np.unique(fp_h[np.isfinite(fp_h)]); cs=np.unique(np.concatenate([cs,GRID])); ch=np.unique(np.concatenate([ch,GRID])); best=None
    for a in cs:
        for b in ch:
            sf=(fp_s<=a)|(fp_h<=b); st=(tp_s<=a)|(tp_h<=b); loss=float(st.mean()) if len(st) else 0.; red=float(sf.mean()) if len(sf) else 0.
            if loss<=.03+1e-12:
                cand=(red,-loss,-float(a),-float(b))
                if best is None or cand>best[0]: best=(cand,float(a),float(b),int(sf.sum()),int(st.sum()),loss,red)
    if best is None: return (None,None), {'status':'no_safe_dev_cutoff','n_fp':int(len(fp_s)),'n_tp':int(len(tp_s))}
    _,a,b,sf,st,loss,red=best; return (a,b), {'status':'safe','cutoff_specific':a,'cutoff_shared':b,'n_fp':int(len(fp_s)),'n_tp':int(len(tp_s)),'suppressed_fp':sf,'suppressed_tp':st,'fp_fraction_suppressed':red,'tp_loss':loss}


def main() -> None:
    if (OUT/'manifest.json').exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    labels=pd.read_parquet(INDEX_DIR/'complete_context_labels.parquet'); feats=pd.read_parquet(FEAT_DIR/'context_eeg10_features.parquet'); labels=labels.merge(feats[['context_id']+FEATURES],on='context_id',how='inner')
    elig=pd.read_csv(ELIG); elig=elig[elig.fit_eligible_standard.astype(bool)][['partition','patient_id','model']]
    labels=labels[labels.partition.isin(elig.partition.unique())]
    units=[]; templates=[]; scores=[]; gates=[]
    # Per patient scaler for the shared branch, fit only on early contexts.
    shared_models={}
    for (part,pid), allg in labels.groupby(['partition','patient_id'], sort=True):
        f=allg[allg.phase.eq('fit')]
        pos=f[(f.kind=='fp_m') & f.shared_fp.astype(bool)].drop_duplicates('context_id')
        neg=f[f.kind=='tn_m'].drop_duplicates('context_id')
        src=f[f.kind.isin(['fp_m','tn_m','tp_m','seizure'])].drop_duplicates('context_id')
        shared_models[(part,pid)]=fit_template(pos,neg,src,'shared')
    # detector-specific branch and dev calibration.
    for u in elig.itertuples(index=False):
        part,pid,model=str(u.partition),str(u.patient_id),str(u.model)
        g=labels[(labels.partition==part)&(labels.patient_id==pid)&(labels.model==model)]
        fit=g[g.phase.eq('fit')]; dev=g[g.phase.eq('dev')]
        pos=fit[fit.kind.eq('fp_m')].drop_duplicates('context_id'); neg=fit[fit.kind.eq('tn_m')].drop_duplicates('context_id'); src=fit[fit.kind.isin(['fp_m','tn_m','tp_m'])].drop_duplicates('context_id')
        specific=fit_template(pos,neg,src,'specific')
        # A patient-level shared template uses the same fit-only scaler and is
        # intentionally independent of current detector identity.
        shared=shared_models.get((part,pid), {'status':'not_found'})
        d_fp=dev[dev.kind.eq('fp_m')].drop_duplicates('context_id'); d_tp=dev[dev.kind.eq('tp_m')].drop_duplicates('context_id')
        sf=score(specific,d_fp); sh=score(shared,d_fp); st=score(specific,d_tp); sth=score(shared,d_tp)
        cs, gs=single_gate(sf,st); ch, gh=single_gate(sh,sth); (cds,cdh), gd=dual_gate(sf,sh,st,sth)
        uid=f'{part}|{pid}|{model}'; units.append({'unit_id':uid,'partition':part,'patient_id':pid,'model':model,'fit_fp':int(len(pos)),'fit_fp_edfs':int(pos.recording.nunique()),'specific_status':specific.get('status'),'specific_eps':specific.get('eps'),'specific_components':len(specific.get('components',[])),'shared_status':shared.get('status'),'shared_components':len(shared.get('components',[])),'dev_fp':int(len(d_fp)),'dev_tp':int(len(d_tp))})
        templates.append({'unit_id':uid,'branch':'specific','status':specific.get('status'),'eps':specific.get('eps'),'n_pos':specific.get('n_pos'),'n_pos_recordings':specific.get('n_pos_recordings'),'components':json.dumps(specific.get('components',[]))})
        templates.append({'unit_id':uid,'branch':'shared','status':shared.get('status'),'eps':shared.get('eps'),'n_pos':shared.get('n_pos'),'n_pos_recordings':shared.get('n_pos_recordings'),'components':json.dumps(shared.get('components',[]))})
        gates += [{'unit_id':uid,'method':'specific','cutoff':cs,**gs},{'unit_id':uid,'method':'shared','cutoff':ch,**gh},{'unit_id':uid,'method':'dual','cutoff':None if cds is None else json.dumps({'specific':cds,'shared':cdh}),**gd}]
        for kind,fr,ss,hh in [('fp_m',d_fp,sf,sh),('tp_m',d_tp,st,sth)]:
            for i,row in enumerate(fr.itertuples(index=False)):
                scores.append({'unit_id':uid,'phase':'dev','kind':kind,'context_id':row.context_id,'recording':row.recording,'specific_distance':float(ss[i]),'shared_distance':float(hh[i]),'shared_fp_candidate':bool(getattr(row,'shared_fp',False))})
    pd.DataFrame(units).to_csv(OUT/'unit_summary.csv',index=False); pd.DataFrame(templates).to_csv(OUT/'template_summary.csv',index=False); pd.DataFrame(gates).to_csv(OUT/'dev_gates.csv',index=False); pd.DataFrame(scores).to_parquet(OUT/'dev_scores.parquet',index=False)
    m={'release_id':OUT.name,'status':'fit_templates_and_dev_gates_complete','holdout_read':False,'source_labels_sha256':hashlib.sha256((INDEX_DIR/'complete_context_labels.parquet').read_bytes()).hexdigest(),'source_features_sha256':hashlib.sha256((FEAT_DIR/'context_eeg10_features.parquet').read_bytes()).hexdigest(),'eligible_units':int(len(units)),'eligible_patients':int(len(set(x['patient_id'] for x in units))),'template_branches':['detector_specific','patient_shared'],'specific_definition':'fit FP component >=3 events across >=2 EDF; scaler and leave-one-EDF-out distance cutoff fit only','shared_definition':'fit 2-of-3 candidate component >=3 events across >=2 EDF; current shared flag is candidate only','dev_gate':'maximize FP suppression subject to detector TP loss <=3 percentage points','holdout_used_for_selection':False,'test_evaluation_count':0,'patient_bootstrap':0,'files':{'unit_summary':'unit_summary.csv','template_summary':'template_summary.csv','dev_gates':'dev_gates.csv','dev_scores':'dev_scores.parquet'}}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n'); print(json.dumps(m,indent=2,sort_keys=True))


if __name__=='__main__': main()
