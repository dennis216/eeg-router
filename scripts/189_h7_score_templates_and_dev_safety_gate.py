"""Score frozen H7 templates on TP/FP contexts and select a DEV-only safety gate.

No template is refit here.  A single global multiplier of each fit-selected
template cutoff is selected per method on DEV, maximising FP removal subject to
at most a 3 percentage-point event-sensitivity loss.  Holdout is not read.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT=Path('/root/b_false_alarm_atlas')
DATA=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-with-tp-v1/context_eeg10_with_tp.parquet'
FIT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1'
ELIG=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
MAN=ROOT/'outputs/reports/h6-tusz-raw-pre-router-v1/manifests'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-template-safety-gate-v1'
FEATURES=('rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log','relative_delta_power','relative_theta_power','relative_alpha_power','relative_beta_power','relative_gamma_power','high_low_ratio')
ALPHAS=(.25,.5,.75,1.,1.25,1.5,2.)

def h(key:str)->str:return hashlib.sha256(key.encode()).hexdigest()[:16]
def score(frame,obj):
 c=np.asarray(obj['centers'],dtype=float)
 if not len(c) or not np.isfinite(obj['cutoff']):return np.full(len(frame),np.inf),np.full(len(frame),np.inf)
 d=cdist(obj['scaler'].transform(frame.loc[:,FEATURES]),c).min(1);return d,d/float(obj['cutoff'])
def truths()->pd.DataFrame:
 rows=[]
 for part in ('train','dev','eval'):
  by={}
  for r in json.loads((MAN/f'{part}_manifest.json').read_text())['records']:by.setdefault(str(r['patient_id']),[]).append(r)
  for patient,rs in by.items():
   rs=sorted(rs,key=lambda x:str(x['relative_edf']));total=sum(float(r['duration_s']) for r in rs);off=0.
   for r in rs:
    for e in r['seizure_events']:
     mid=off+(float(e['start_s'])+float(e['end_s']))/2;phase='fit' if mid<.6*total else 'dev' if mid<.8*total else 'holdout'
     rows.append({'partition':part,'patient_id':patient,'phase':phase,'truth':1})
    off+=float(r['duration_s'])
 return pd.DataFrame(rows).groupby(['partition','patient_id','phase']).size().rename('truth_events').reset_index()
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);x=pd.read_parquet(DATA);e=pd.read_csv(ELIG);e=e[e.fit_eligible_standard]
 # template scores for detector-associated rows
 spec=[]
 for u in e.itertuples(index=False):
  q=x[(x.partition==u.partition)&(x.patient_id==u.patient_id)&(x.model==u.model)&x.kind.isin(['fp_m','tn_m','tp_m'])].copy();key=f'{u.partition}|{u.patient_id}|{u.model}';o=joblib.load(FIT/'objects'/f'specific_{h(key)}.joblib');d,r=score(q,o);q['specific_distance']=d;q['specific_ratio']=r;q['specific_cutoff']=o['cutoff'];spec.append(q)
 spec=pd.concat(spec,ignore_index=True)
 # patient-level shared scores, including safety annotations for a diagnostic only.
 shared=[]
 for (part,patient),q in x.groupby(['partition','patient_id'],sort=True):
  if not ((e.partition==part)&(e.patient_id==patient)).any():continue
  key=f'{part}|{patient}';o=joblib.load(FIT/'objects'/f'shared_{h(key)}.joblib');d,r=score(q,o);z=q[['partition','patient_id','phase','model','recording','context_end_s','kind','context_id']].copy();z['shared_distance']=d;z['shared_ratio']=r;z['shared_cutoff']=o['cutoff'];shared.append(z)
 shared=pd.concat(shared,ignore_index=True)
 cols=['partition','patient_id','phase','model','recording','context_end_s','kind','context_id','shared_distance','shared_ratio','shared_cutoff']
 z=spec.merge(shared[cols],on=['partition','patient_id','phase','model','recording','context_end_s','kind','context_id'],how='left')
 z.to_parquet(OUT/'fit_dev_event_template_scores.parquet',index=False)
 # DEV-only action simulation. TP labels come from frozen one-to-one matching,
 # so suppressing a TP removes exactly one frozen seizure detection.
 t=truths();dev=z[z.phase.eq('dev')].copy();rows=[]
 for method in ('specific','shared','dual_or'):
  for alpha in ALPHAS:
   if method=='specific': sup=dev.specific_ratio<=alpha
   elif method=='shared': sup=dev.shared_ratio<=alpha
   else:sup=(dev.specific_ratio<=alpha)|(dev.shared_ratio<=alpha)
   d=dev.assign(suppressed=sup.fillna(False))
   for (part,patient,model),g in d.groupby(['partition','patient_id','model'],sort=True):
    ntruth=int(t[(t.partition==part)&(t.patient_id==patient)&(t.phase=='dev')].truth_events.sum())
    fp=g[g.kind.eq('fp_m')];tp=g[g.kind.eq('tp_m')]
    rows.append({'method':method,'alpha':alpha,'partition':part,'patient_id':patient,'model':model,'dev_truth_events':ntruth,'baseline_fp_events':len(fp),'suppressed_fp_events':int(fp.suppressed.sum()),'baseline_matched_events':len(tp),'suppressed_matched_events':int(tp.suppressed.sum())})
 r=pd.DataFrame(rows);r['fa_reduction_fraction']=r.suppressed_fp_events/r.baseline_fp_events.replace(0,np.nan);r['sensitivity_loss_pp']=100*r.suppressed_matched_events/r.dev_truth_events.replace(0,np.nan)
 agg=r.groupby(['method','alpha'],as_index=False).agg(units=('model','size'),baseline_fp_events=('baseline_fp_events','sum'),suppressed_fp_events=('suppressed_fp_events','sum'),baseline_matched_events=('baseline_matched_events','sum'),suppressed_matched_events=('suppressed_matched_events','sum'),dev_truth_events=('dev_truth_events','sum'))
 agg['fa_reduction_fraction']=agg.suppressed_fp_events/agg.baseline_fp_events.replace(0,np.nan);agg['sensitivity_loss_pp']=100*agg.suppressed_matched_events/agg.dev_truth_events.replace(0,np.nan);agg['safe']=agg.sensitivity_loss_pp<=3.
 agg['selected']=False
 for m in agg.method.unique():
  candidates=agg[(agg.method==m)&agg.safe]
  if len(candidates):agg.loc[candidates.fa_reduction_fraction.idxmax(),'selected']=True
 r.to_csv(OUT/'dev_patient_unit_action_curve.csv',index=False);agg.to_csv(OUT/'dev_method_gate_curve.csv',index=False)
 # Diagnostic: seizure context score is deliberately not used to select alpha.
 safety=shared[(shared.phase=='dev')&(shared.kind=='seizure')].copy();safety.to_parquet(OUT/'dev_annotation_seizure_shared_diagnostic.parquet',index=False)
 m={'release_id':OUT.name,'status':'dev_only_template_safety_gate_complete_no_holdout','holdout_read':False,'source_feature_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'source_template_manifest_sha256':hashlib.sha256((FIT/'manifest.json').read_bytes()).hexdigest(),'threshold_contract':'one global alpha per method selected on DEV for maximum FP-event removal subject to <=3pp loss relative to frozen detector matches; alpha grid fixed before selection','methods':['specific','shared','dual_or'],'alphas':list(ALPHAS),'eligible_units':len(e),'event_score_rows':len(z),'sha256':hashlib.sha256((OUT/'dev_method_gate_curve.csv').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(agg.to_string(index=False));print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
