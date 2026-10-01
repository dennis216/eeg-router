"""Build the frozen H7 exploratory chronological-holdout event catalog.

Policy, templates, and alpha values were frozen before this script.  This
catalog is explicitly exploratory because TUSZ chronological holdout was seen
in earlier H7 work; it is not used for any choice or retraining.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path('/root/b_false_alarm_atlas');EVENTS=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet';ELIG=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv';POLICY=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1/manifest.json';OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-catalog-v1';CTX=20.
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);e=pd.read_parquet(EVENTS);u=pd.read_csv(ELIG);u=u[u.fit_eligible_standard][['partition','patient_id','model']];e=e[(e.phase=='holdout')&(e.start_s>=CTX)].merge(u,on=['partition','patient_id','model'],how='inner').copy();e['kind']=e.is_fp.map({1:'fp_m',0:'tp_m'});e['context_end_s']=e.start_s.astype(float);e['source_start_s']=e.start_s.astype(float);e['source_end_s']=e.end_s.astype(float);keep=['partition','patient_id','phase','model','recording','context_end_s','kind','source_start_s','source_end_s','is_fp','matched_seizure'];e=e[keep].drop_duplicates(['partition','patient_id','model','recording','context_end_s','kind']).sort_values(keep[:6]);e.to_parquet(OUT/'holdout_event_catalog.parquet',index=False);s=e.groupby(['kind','model']).size().rename('n').reset_index();s.to_csv(OUT/'summary.csv',index=False);m={'release_id':OUT.name,'status':'exploratory_holdout_catalog_only','exploratory_reason':'TUSZ chronological holdout was previously examined in earlier H7 work; this evaluation cannot be confirmatory','policy_sha256':hashlib.sha256(POLICY.read_bytes()).hexdigest(),'source_events_sha256':hashlib.sha256(EVENTS.read_bytes()).hexdigest(),'holdout_accessed':True,'event_contexts':len(e),'unique_edfs':e.recording.nunique(),'sha256':hashlib.sha256((OUT/'holdout_event_catalog.parquet').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(s.to_string(index=False));print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
