"""Audit whether fit-only recurrent-template labels support patient-local raw CNNs.

No model is fit. A positive is a FIT FP mapped to a frozen recurrent component.
Negatives are matched TN and seizure-matched TP; for the shared branch, private
FPs are additionally hard negatives.  This prevents a CNN from merely learning
that an event was an FP and records data sufficiency before any training.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path('/root/b_false_alarm_atlas')
DATA=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-template-safety-gate-v1/fit_dev_event_template_scores.parquet'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cnn-label-support-audit-v1'
MIN_POS=30;MIN_NEG=30;MIN_EDFS=3
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);x=pd.read_parquet(DATA);x=x[x.phase.eq('fit')].copy()
 # Specific: only recurring component FP vs background/TN and matched seizure.
 x['specific_target']=pd.NA;x.loc[(x.kind=='fp_m')&(x.specific_ratio<=1),'specific_target']=1;x.loc[x.kind.isin(['tn_m','tp_m']),'specific_target']=0
 # Shared: template-member shared FP vs background, true seizure detections, and private FP.
 x['shared_target']=pd.NA;x.loc[(x.kind=='fp_m')&x.shared_fp&(x.shared_ratio<=1),'shared_target']=1;x.loc[x.kind.isin(['tn_m','tp_m']),'shared_target']=0;x.loc[(x.kind=='fp_m')&(~x.shared_fp),'shared_target']=0
 rows=[]
 for branch,target in [('specific','specific_target'),('shared','shared_target')]:
  groupcols=['partition','patient_id','model'] if branch=='specific' else ['partition','patient_id']
  for key,g in x[x[target].notna()].groupby(groupcols,sort=True):
   pos=g[g[target].eq(1)];neg=g[g[target].eq(0)];rec=pos.recording.nunique();rows.append({'branch':branch,**dict(zip(groupcols,key if isinstance(key,tuple) else (key,))), 'fit_positive':len(pos),'fit_negative':len(neg),'positive_edfs':rec,'cnn_data_supported':bool(len(pos)>=MIN_POS and len(neg)>=MIN_NEG and rec>=MIN_EDFS)})
 s=pd.DataFrame(rows);s.to_csv(OUT/'per_unit_label_support.csv',index=False);summary=s.groupby('branch').agg(units=('branch','size'),supported=('cnn_data_supported','sum'),median_positive=('fit_positive','median'),median_negative=('fit_negative','median'),median_positive_edfs=('positive_edfs','median')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 labels=x[['partition','patient_id','phase','model','recording','context_end_s','kind','context_id','specific_target','shared_target']];labels.to_parquet(OUT/'fit_cnn_candidate_labels.parquet',index=False)
 m={'release_id':OUT.name,'status':'audit_only_no_cnn_fit','holdout_read':False,'source_scores_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'specific_positive':'FIT FP_m that maps to a frozen specific recurrent component','shared_positive':'FIT 2-of-3 shared FP that maps to a frozen shared recurrent component','negative_controls':'specific: TN_m + TP_m; shared: TN_m + TP_m + private FP_m','cnn_data_gate':{'min_positive':MIN_POS,'min_negative':MIN_NEG,'min_positive_edfs':MIN_EDFS},'counts':summary.to_dict('records'),'sha256':hashlib.sha256((OUT/'per_unit_label_support.csv').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(summary.to_string(index=False));print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
