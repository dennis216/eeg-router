"""Development-only event simulation for the frozen H7-v4 prototypes.

No probabilities, thresholds, or prototypes are refit here.  A dev event is
marked suppressible only when it is near a mapped recurrent shared-FP sample;
the script reports this as a feasibility/upper-bound analysis, not a final
performance result.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/root/b_false_alarm_atlas');DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data';SRC=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1';MAP=ROOT/'outputs/reports/h7-personalized-consensus-phenotype-suppressor-v4-dev-gate';OUT=ROOT/'outputs/reports/h7-personalized-consensus-phenotype-suppressor-v4-template-simulation';TOL=10.
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True);lab=pd.read_parquet(DATA/'labels.parquet');mp=pd.read_parquet(MAP/'dev_all_sample_mappings.parquet');ev=pd.read_parquet(SRC/'frozen_expert_events.parquet');gate=pd.read_csv(ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle/patient_dev_only_gate.csv').set_index('patient_id');rows=[]
 for pat in sorted(lab.patient_id.unique()):
  g=lab[(lab.patient_id==pat)&(lab.phase=='dev')];m=mp[(mp.patient_id==pat)&mp.mapped];cand=m[m.kind=='shared_fp'];e=ev[(ev.patient_id==pat)&(ev.phase=='dev')]
  if pat not in gate.index:continue
  truth=int(gate.loc[pat,'dev_truth_events']);scores=[]
  for model in ['singlem','cbramod','tcn_gat']:
   q=e[e.model==model];scores.append((model,int(q.matched_seizure.sum()),int(q.is_fp.sum())))
  mx=max(z[1]for z in scores);allow=[z for z in scores if z[1]/max(1,truth)>=mx/max(1,truth)-.03];best=min(allow,key=lambda z:(z[2],-z[1],z[0]));fp=e[(e.model==best[0])&(e.is_fp==1)];sz=e[e.matched_seizure==1]
  covered=0
  for r in fp.itertuples(index=False):covered+=int(((cand.recording==r.recording)&(cand.context_end_s>=r.start_s-TOL)&(cand.context_end_s<=r.end_s+TOL)).any()) if len(cand) else 0
  # Safety audit: an event-level candidate is risky if a seizure detection
  # lies in the same collar. This is intentionally conservative.
  risky=0
  for r in sz.itertuples(index=False):risky+=int(((cand.recording==r.recording)&(cand.context_end_s>=r.start_s-TOL)&(cand.context_end_s<=r.end_s+TOL)).any()) if len(cand) else 0
  rows.append({'patient_id':pat,'dev_shared_candidates':int(len(cand)),'best_fixed_detector':best[0],'best_fixed_dev_fp':best[2],'best_fixed_dev_detected_seizures':best[1],'mapped_candidate_fp_count':covered,'mapped_candidate_fp_fraction':covered/best[2] if best[2] else None,'mapped_candidate_seizure_event_count':risky,'sensitivity_loss_if_all_mapped_candidates_suppressed':risky/max(1,truth)})
 out=pd.DataFrame(rows);out.to_csv(OUT/'template_dev_simulation.csv',index=False);m={'release_id':OUT.name,'status':'dev_only_template_feasibility','source_prototypes':str(MAP.relative_to(ROOT)),'holdout_read':False,'threshold_source':'fit-only candidate-background q05; no dev tuning','action_simulated':'remove frozen best-detector events within ±10 s of mapped recurrent shared-FP prototype','interpretation':'upper bound; does not rewrite probabilities or claim clinical performance','safety_gate':'no mapped seizure events and >=20% best-fixed FP coverage in >=3 patients','patients_with_zero_mapped_seizure_events':int((out.mapped_candidate_seizure_event_count==0).sum()),'patients_meeting_fp_gate':int((out.mapped_candidate_fp_fraction>=.2).sum()),'patients_meeting_both':int(((out.mapped_candidate_seizure_event_count==0)&(out.mapped_candidate_fp_fraction>=.2)).sum()),'sha256':hashlib.sha256((OUT/'template_dev_simulation.csv').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(out.to_string(index=False));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
