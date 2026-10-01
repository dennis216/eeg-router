"""Create deterministic fit/dev context catalog for H7 template baselines.

Catalog-only stage: no EDF signal is loaded. All contexts are causal 20-s
windows ending at the alarm onset (FP) or 10 s after annotated seizure onset
(safety control). A later extractor materializes the catalog unchanged.
"""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path('/root/b_false_alarm_atlas'); DATA=Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')
ELIG=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
EVENTS=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet'
SHARED=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle/fit_dev_shared_fp_labels.parquet'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-v1';TOL=10.;CTX=20.
def ann(rec):
 p=DATA/rec; q=Path(str(p.with_suffix('.csv'))+'_bi');ls=[z.strip()for z in q.read_text(encoding='utf-8-sig').splitlines()if z.strip()and not z.startswith('#')]
 return [(float(r['start_time']),float(r['stop_time']))for r in csv.DictReader(ls)if str(r.get('label','')).lower()=='seiz']
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True);u=pd.read_csv(ELIG);u=u[u.fit_eligible_standard];e=pd.read_parquet(EVENTS);s=pd.read_parquet(SHARED);rows=[]
 for unit in u.itertuples(index=False):
  g=e[(e.partition==unit.partition)&(e.patient_id==unit.patient_id)&(e.model==unit.model)&e.phase.isin(['fit','dev'])]
  for x in g[g.is_fp==1].itertuples(index=False):
   if x.start_s<CTX:continue
   sh=s[(s.partition==x.partition)&(s.patient_id==x.patient_id)&(s.phase==x.phase)&(s.recording==x.recording)&s.model_mask.str.contains(x.model,regex=False)&(abs(s.decision_time_s-x.start_s)<=TOL)]
   rows.append({'partition':x.partition,'patient_id':x.patient_id,'phase':x.phase,'model':x.model,'recording':x.recording,'context_end_s':float(x.start_s),'kind':'fp_m','shared_fp':bool(len(sh)),'source_start_s':float(x.start_s),'source_end_s':float(x.end_s)})
 # Safety controls are catalogued per patient/phase/record, then reused by all
 # detector units from that patient.  Annotation source is independent of any
 # detector's success or failure.
 patient_phase=u[['partition','patient_id']].drop_duplicates()
 recs=e[e.phase.isin(['fit','dev'])][['partition','patient_id','phase','recording']].drop_duplicates()
 for pp in patient_phase.itertuples(index=False):
  for r in recs[(recs.partition==pp.partition)&(recs.patient_id==pp.patient_id)].itertuples(index=False):
   for a,b in ann(r.recording):
    end=a+10.
    if end>=CTX:rows.append({'partition':r.partition,'patient_id':r.patient_id,'phase':r.phase,'model':'__safety__','recording':r.recording,'context_end_s':end,'kind':'seizure','shared_fp':False,'source_start_s':a,'source_end_s':b})
 x=pd.DataFrame(rows).drop_duplicates(['partition','patient_id','phase','model','recording','context_end_s','kind']).sort_values(['partition','patient_id','phase','model','recording','context_end_s'])
 x.to_parquet(OUT/'context_catalog.parquet',index=False)
 summ=x.groupby(['model','phase','kind','shared_fp']).size().rename('n_contexts').reset_index();summ.to_csv(OUT/'context_summary.csv',index=False)
 m={'release_id':OUT.name,'status':'catalog_only_no_raw_eeg_loaded','holdout_read':False,'eligible_units':int(len(u)),'eligible_patients':int(u.patient_id.nunique()),'contexts':int(len(x)),'window':'20-s causal window ending at FP onset; seizure safety control ends 10 s after seizure onset','shared_definition':'existing frozen 2-of-3 FP onset collar ±10 seconds, supplementary only','sha256':hashlib.sha256((OUT/'context_catalog.parquet').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(summ.to_string(index=False));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
