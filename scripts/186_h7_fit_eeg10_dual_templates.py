"""Fit patient-local dual recurrent-FP templates using FIT EEG10 only.

Specific templates are fit separately for every patient-detector unit. Shared
templates are fit per patient from the frozen 2-of-3 candidate flag. A template
component must contain >=3 FP contexts from >=2 EDF files. Thresholds are
chosen within fit by leave-one-EDF-out FP-vs-matched-TN balanced accuracy.
DEV is scored only after all fit objects are frozen; no holdout is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/root/b_false_alarm_atlas')
FEATURE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-context-features-v1/context_eeg10.parquet'
ELIG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1'
FEATURES = ('rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log',
            'relative_delta_power','relative_theta_power','relative_alpha_power','relative_beta_power','relative_gamma_power','high_low_ratio')
MODELS = ('singlem','cbramod','tcn_gat')


def pick_cutoff(fp: pd.DataFrame, tn: pd.DataFrame, scaler: StandardScaler) -> tuple[float, float, int]:
    """Fit-only LOEO choice. Each held EDF is scored vs all other EDF FPs."""
    xfp = scaler.transform(fp.loc[:, FEATURES]); xtn = scaler.transform(tn.loc[:, FEATURES])
    # Cross-EDF nearest-positive distances determine an interpretable candidate grid.
    distances = []
    for i, row in enumerate(fp.itertuples(index=False)):
        other = fp.recording.to_numpy() != row.recording
        if other.any(): distances.append(float(cdist(xfp[i:i+1], xfp[other]).min()))
    if len(distances) < 5:
        return float('nan'), float('nan'), 0
    cuts = np.unique(np.quantile(np.asarray(distances), [.10,.25,.40,.55,.70,.85]))
    fold_scores = {float(c): [] for c in cuts}
    valid_folds = 0
    for record in sorted(fp.recording.unique()):
        train_fp = xfp[fp.recording.to_numpy() != record]
        test_fp = xfp[fp.recording.to_numpy() == record]
        test_tn = xtn[tn.recording.to_numpy() == record]
        if len(train_fp) == 0 or len(test_fp) == 0 or len(test_tn) == 0: continue
        valid_folds += 1
        scores = np.r_[cdist(test_fp, train_fp).min(axis=1), cdist(test_tn, train_fp).min(axis=1)]
        y = np.r_[np.ones(len(test_fp),int),np.zeros(len(test_tn),int)]
        for cut in cuts: fold_scores[float(cut)].append(balanced_accuracy_score(y, scores <= cut))
    if not valid_folds:
        return float('nan'),float('nan'),0
    best = max(cuts,key=lambda c:(np.mean(fold_scores[float(c)]),-float(c)))
    return float(best),float(np.mean(fold_scores[float(best)])),valid_folds


def components(fp: pd.DataFrame, scaler: StandardScaler, cutoff: float) -> tuple[np.ndarray, pd.DataFrame]:
    if not np.isfinite(cutoff) or len(fp) < 3: return np.full(len(fp),-1,dtype=int),pd.DataFrame()
    z=scaler.transform(fp.loc[:,FEATURES]); d=cdist(z,z)
    records=fp.recording.to_numpy(); edge=(d<=cutoff)&(records[:,None]!=records[None,:]);np.fill_diagonal(edge,False)
    _, labels=connected_components(csr_matrix(edge),directed=False,return_labels=True)
    keep=set()
    for k in np.unique(labels):
        m=labels==k
        if m.sum()>=3 and fp.loc[m,'recording'].nunique()>=2: keep.add(int(k))
    labels=np.array([int(k) if int(k) in keep else -1 for k in labels],dtype=int)
    rows=[]
    for k in sorted(keep):
        m=labels==k; center=np.median(z[m],axis=0); rows.append({'component':k,'n_events':int(m.sum()),'n_edfs':int(fp.loc[m,'recording'].nunique()),'center':center})
    return labels,pd.DataFrame(rows)


def score(frame: pd.DataFrame, scaler: StandardScaler, centers: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    if centers.empty: return np.full(len(frame),np.inf),np.zeros(len(frame),dtype=bool)
    z=scaler.transform(frame.loc[:,FEATURES]); c=np.stack(centers.center.to_list())
    s=cdist(z,c).min(axis=1); return s,np.ones(len(frame),dtype=bool)


def make_object(fp: pd.DataFrame, tn: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scaler=StandardScaler().fit(pd.concat([fp.loc[:,FEATURES],tn.loc[:,FEATURES]],ignore_index=True))
    cut,loo,nfold=pick_cutoff(fp,tn,scaler)
    labels,centers=components(fp,scaler,cut)
    obj={'scaler':scaler,'cutoff':cut,'loeo_balanced_accuracy':loo,'loeo_edfs':nfold,
         'centers':centers.center.to_list() if not centers.empty else [],
         'component_metadata':centers.drop(columns='center',errors='ignore').to_dict('records')}
    return obj,centers


def main() -> None:
    if (OUT/'manifest.json').exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True); (OUT/'objects').mkdir()
    x=pd.read_parquet(FEATURE); elig=pd.read_csv(ELIG); elig=elig[elig.fit_eligible_standard]
    all_scores=[]; summaries=[]; objects={}
    # detector-specific templates
    for unit in elig.itertuples(index=False):
        base=x[(x.partition==unit.partition)&(x.patient_id==unit.patient_id)&(x.model==unit.model)]
        fp=base[(base.phase=='fit')&(base.kind=='fp_m')]; tn=base[(base.phase=='fit')&(base.kind=='tn_m')]
        obj,centers=make_object(fp,tn); key=f'{unit.partition}|{unit.patient_id}|{unit.model}'; objects[key]=obj
        joblib.dump(obj,OUT/'objects'/f'specific_{hashlib.sha256(key.encode()).hexdigest()[:16]}.joblib')
        observed=base[base.kind.isin(['fp_m','tn_m'])].copy(); s,has=score(observed,obj['scaler'],centers)
        observed['specific_score']=s;observed['specific_has_template']=has;observed['specific_match']=s<=obj['cutoff']
        all_scores.append(observed);summaries.append({'branch':'specific','partition':unit.partition,'patient_id':unit.patient_id,'model':unit.model,'fit_fp':len(fp),'fit_tn':len(tn),'cutoff':obj['cutoff'],'loeo_balanced_accuracy':obj['loeo_balanced_accuracy'],'loeo_edfs':obj['loeo_edfs'],'components':len(centers),'template_events':int(sum(z['n_events'] for z in obj['component_metadata']))})
    # shared templates: candidate FPs across detector labels are de-duplicated by raw context ID.
    for (partition, patient_id), patient_frame in x.groupby(['partition','patient_id'],sort=True):
        unitmask=(elig.partition==partition)&(elig.patient_id==patient_id)
        if not unitmask.any(): continue
        fp_all=patient_frame[(patient_frame.phase=='fit')&(patient_frame.kind=='fp_m')]
        shared=fp_all[fp_all.shared_fp].drop_duplicates('context_id')
        controls=patient_frame[(patient_frame.phase=='fit')&(patient_frame.kind=='tn_m')].drop_duplicates('context_id')
        # Require same recurrence standard as specific; a patient with no valid shared set gets an explicit empty object.
        if len(shared)>=3 and shared.recording.nunique()>=2 and len(controls)>=3:
            obj,centers=make_object(shared,controls)
        else:
            scaler=StandardScaler().fit(pd.concat([fp_all.loc[:,FEATURES],controls.loc[:,FEATURES]],ignore_index=True));obj={'scaler':scaler,'cutoff':float('nan'),'loeo_balanced_accuracy':float('nan'),'loeo_edfs':0,'centers':[],'component_metadata':[]};centers=pd.DataFrame()
        key=f'{partition}|{patient_id}'; objects['shared|'+key]=obj;joblib.dump(obj,OUT/'objects'/f'shared_{hashlib.sha256(key.encode()).hexdigest()[:16]}.joblib')
        observed=patient_frame[patient_frame.kind.isin(['fp_m','tn_m','seizure'])].copy(); s,has=score(observed,obj['scaler'],centers)
        observed['shared_score']=s;observed['shared_has_template']=has;observed['shared_match']=s<=obj['cutoff']
        observed['shared_template_cutoff']=obj['cutoff'];all_scores.append(observed[['partition','patient_id','phase','model','recording','context_end_s','kind','shared_fp','context_id','shared_score','shared_has_template','shared_match','shared_template_cutoff']]);summaries.append({'branch':'shared','partition':partition,'patient_id':patient_id,'model':'__patient__','fit_fp':len(shared),'fit_tn':len(controls),'cutoff':obj['cutoff'],'loeo_balanced_accuracy':obj['loeo_balanced_accuracy'],'loeo_edfs':obj['loeo_edfs'],'components':len(centers),'template_events':int(sum(z['n_events'] for z in obj['component_metadata']))})
    # Merge branch outputs using a key. specific scores apply only detector rows; shared applies to all rows.
    specific=pd.concat([z for z in all_scores if 'specific_score' in z],ignore_index=True)
    shared=pd.concat([z for z in all_scores if 'shared_score' in z],ignore_index=True)
    out=shared.merge(specific[['partition','patient_id','phase','model','recording','context_end_s','kind','context_id','specific_score','specific_has_template','specific_match']],on=['partition','patient_id','phase','model','recording','context_end_s','kind','context_id'],how='left')
    out.to_parquet(OUT/'fit_dev_template_scores.parquet',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'template_summary.csv',index=False)
    manifest={'release_id':OUT.name,'status':'fit_only_templates_dev_scored_no_holdout','holdout_read':False,
              'specific_rule':'fit FP_m components >=3 events across >=2 EDF; StandardScaler fit on fit FP_m + matched TN_m; cutoff selected by leave-one-EDF-out fit balanced accuracy',
              'shared_rule':'fit frozen 2-of-3 shared FP candidates, de-duplicated by raw context; same recurrence and fit-only selection rule',
              'features':list(FEATURES),'eligible_units':int(len(elig)),'template_objects':int(len(objects)),
              'score_rows':int(len(out)),'source_feature_sha256':hashlib.sha256(FEATURE.read_bytes()).hexdigest(),
              'sha256':hashlib.sha256((OUT/'fit_dev_template_scores.parquet').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(summary.groupby('branch').agg(units=('branch','size'),components=('components','sum'),template_events=('template_events','sum')).to_string());print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__':main()
