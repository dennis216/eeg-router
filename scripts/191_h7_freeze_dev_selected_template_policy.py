"""Freeze H7 template policy choices selected on development data only."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path('/root/b_false_alarm_atlas')
GATE=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-template-safety-gate-v1/dev_method_gate_curve.csv'
CNN=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cnn-label-support-audit-v1/manifest.json'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1'
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);x=pd.read_csv(GATE);selected=x[x.selected].copy();selected.to_csv(OUT/'selected_dev_policies.csv',index=False)
 primary=selected[selected.method.eq('shared')].iloc[0].to_dict()
 m={'release_id':OUT.name,'status':'development_policy_frozen','holdout_read':False,'primary_policy':{'method':'shared','alpha':float(primary['alpha']),'selection':'maximum DEV FP-event removal subject to <=3pp frozen matched-event sensitivity loss'},'prespecified_secondary_ablation_policies':selected[['method','alpha']].to_dict('records'),'dev_results_at_selection':selected.to_dict('records'),'cnn_decision':'not_started: no specific unit and only 3 shared patients reached predeclared fit data support gate; raw CNN would be underpowered/overfit-prone','source_gate_sha256':hashlib.sha256(GATE.read_bytes()).hexdigest(),'source_cnn_audit_sha256':hashlib.sha256(CNN.read_bytes()).hexdigest(),'next_evaluation':'external independent cohort; TUSZ chronological holdout was previously examined in earlier H7 work and must only be labelled exploratory if later scored'}
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
