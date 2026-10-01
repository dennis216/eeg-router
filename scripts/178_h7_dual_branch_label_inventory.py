"""Fit/dev-only data-support inventory for H7 dual-branch suppression."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path('/root/b_false_alarm_atlas')
EVENTS=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet'
SHARED=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle/fit_dev_shared_fp_labels.parquet'
TRUTH=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle/patient_dev_only_gate.csv'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-label-inventory'
MODELS=['singlem','cbramod','tcn_gat']
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True);e=pd.read_parquet(EVENTS);s=pd.read_parquet(SHARED);t=pd.read_csv(TRUTH)[['partition','patient_id','dev_truth_events']]
 rows=[]
 for z in t.itertuples(index=False):
  for m in MODELS:
   g=e[(e.partition==z.partition)&(e.patient_id==z.patient_id)&(e.model==m)]
   f=g[(g.phase=='fit')&(g.is_fp==1)];d=g[(g.phase=='dev')&(g.is_fp==1)]
   sh=s[(s.partition==z.partition)&(s.patient_id==z.patient_id)]
   fs=sh[(sh.phase=='fit')&sh.model_mask.str.contains(m,regex=False)]
   ds=sh[(sh.phase=='dev')&sh.model_mask.str.contains(m,regex=False)]
   eligible=(len(f)>=20 and f.recording.nunique()>=3 and len(d)>=5 and z.dev_truth_events>=2)
   rows.append({'partition':z.partition,'patient_id':z.patient_id,'model':m,'fit_specific_fp_events':len(f),'fit_specific_fp_edfs':f.recording.nunique(),'fit_shared_fp_events':len(fs),'fit_shared_fp_edfs':fs.recording.nunique(),'dev_specific_fp_events':len(d),'dev_shared_fp_events':len(ds),'dev_truth_events':z.dev_truth_events,'data_eligible':eligible})
 x=pd.DataFrame(rows);x.to_csv(OUT/'patient_detector_label_inventory.csv',index=False)
 summary=x.groupby('model').agg(units=('model','size'),data_eligible=('data_eligible','sum'),fit_fp_median=('fit_specific_fp_events','median'),dev_fp_median=('dev_specific_fp_events','median'),fit_shared_median=('fit_shared_fp_events','median')).reset_index();summary.to_csv(OUT/'summary_by_detector.csv',index=False)
 m={'release_id':OUT.name,'status':'fit_dev_label_inventory_only','holdout_read':False,'sources':[str(EVENTS.relative_to(ROOT)),str(SHARED.relative_to(ROOT)),str(TRUTH.relative_to(ROOT))],'eligibility':'fit FP>=20 across >=3 EDF; dev FP>=5; dev seizure truth>=2','counts':{'patient_detector_units':len(x),'data_eligible_units':int(x.data_eligible.sum()),'patients_with_any_eligible_unit':int(x[x.data_eligible].patient_id.nunique())},'sha256':hashlib.sha256((OUT/'patient_detector_label_inventory.csv').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(summary.to_string(index=False));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
