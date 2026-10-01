"""Frozen-prototype development gate for H7-v4 (no training, no holdout)."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT=Path('/root/b_false_alarm_atlas'); DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'; SRC=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1'; GATE=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle'; OUT=ROOT/'outputs/reports/h7-personalized-consensus-phenotype-suppressor-v4-dev-gate';FS=128.;TOL=10.;MIN_SIZE=3;MIN_EDFS=2
def ftr(rel):
 x=np.load(DATA/rel)['x'].astype(np.float32);r=np.sqrt((x*x).mean(1));f,p=welch(x,fs=FS,nperseg=256,axis=-1);p=p.mean(0)
 def q(lo,hi):
  z=(f>=lo)&(f<hi);return float(np.trapezoid(p[z],f[z]))
 d,t,a,b,g=[q(*z) for z in[(.5,4),(4,8),(8,13),(13,30),(30,45)]]
 return np.array([np.median(r),np.subtract(*np.percentile(r,[75,25])),np.std(r),np.ptp(r),*(np.log10(v+1e-8) for v in[d,t,a,b,g]),np.log10((b+g+1e-8)/(d+t+1e-8))])
def comp(adj):
 seen=set();z=[]
 for s in range(len(adj)):
  if s in seen:continue
  q=[s];seen.add(s);c=[]
  while q:
   i=q.pop();c.append(i)
   for j in np.flatnonzero(adj[i]):
    if int(j)not in seen:seen.add(int(j));q.append(int(j))
  z.append(c)
 return z
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True);x=pd.read_parquet(DATA/'labels.parquet');ev=pd.read_parquet(SRC/'frozen_expert_events.parquet');truth_by_patient=pd.read_csv(GATE/'patient_dev_only_gate.csv').set_index('patient_id').dev_truth_events.to_dict();rows=[];maps=[]
 for patient,g in x.groupby('patient_id',sort=True):
  fit=g[(g.phase=='fit')&(g.kind=='shared_fp')].reset_index(drop=True);bg=g[(g.phase=='fit')&(g.kind=='background')].reset_index(drop=True);dev=g[g.phase=='dev'].reset_index(drop=True)
  if len(fit)<3 or len(bg)<3:continue
  F=np.stack([ftr(v)for v in fit.x_path]);B=np.stack([ftr(v)for v in bg.x_path]);V=np.stack([ftr(v)for v in dev.x_path]);sc=StandardScaler().fit(np.vstack([F,B]));F=sc.transform(F);B=sc.transform(B);V=sc.transform(V)
  threshold=float(np.quantile(np.sqrt(((F[:,None]-B[None,:])**2).sum(2)).ravel(),.05));dist=np.sqrt(((F[:,None]-F[None,:])**2).sum(2));same=fit.recording.to_numpy()[:,None]==fit.recording.to_numpy()[None,:];cs=[c for c in comp((dist<=threshold)&(~same)&(~np.eye(len(F),dtype=bool)))if len(c)>=MIN_SIZE and fit.iloc[c].recording.nunique()>=MIN_EDFS]
  if not cs:
   rows.append({'patient_id':patient,'status':'no_fit_phenotype'});continue
  P=np.stack([F[c].mean(0)for c in cs]);D=np.sqrt(((V[:,None]-P[None,:])**2).sum(2));near=D.min(1);mapped=near<=threshold
  for r,m,d in zip(dev.itertuples(index=False),mapped,near):maps.append({'patient_id':patient,'sample_id':r.sample_id,'recording':r.recording,'context_end_s':r.context_end_s,'kind':r.kind,'mapped':bool(m),'distance':float(d),'threshold':threshold})
  y=(dev.kind=='shared_fp').astype(int).to_numpy();auc=float(roc_auc_score(y,-near)) if y.min()!=y.max() else None
  # Development-only upper bound: which FPs of the personal dev-best frozen
  # detector occur at a mapped shared candidate? No alarm is modified here.
  e=ev[(ev.patient_id==patient)&(ev.phase=='dev')];truth=max(1,int(truth_by_patient.get(patient,1)))
  score=[]
  for m in ['singlem','cbramod','tcn_gat']:
   q=e[e.model==m];score.append((m,int(q.matched_seizure.sum()),int(q.is_fp.sum())))
  mx=max(z[1]for z in score);allow=[z for z in score if z[1]/truth>=mx/truth-.03];best=min(allow,key=lambda z:(z[2],-z[1],z[0]))
  fp=e[(e.model==best[0])&(e.is_fp==1)]; sharedmapped=pd.DataFrame(maps)[lambda z:(z.patient_id==patient)&(z.kind=='shared_fp')&(z.mapped)]
  covered=0
  for r in fp.itertuples(index=False):
   covered+=int(((sharedmapped.recording==r.recording)&(sharedmapped.context_end_s>=r.start_s-TOL)&(sharedmapped.context_end_s<=r.end_s+TOL)).any()) if len(sharedmapped) else 0
  rows.append({'patient_id':patient,'status':'complete','dev_n':len(dev),'dev_shared':int(y.sum()),'dev_mapped_shared':int(((dev.kind=='shared_fp')&mapped).sum()),'dev_mapped_nonshared':int(((dev.kind!='shared_fp')&mapped).sum()),'dev_shared_vs_rest_auc':auc,'best_fixed':best[0],'best_fixed_dev_fp':best[2],'mapped_candidate_fp_coverage_upper_bound':covered/best[2] if best[2] else None})
 out=pd.DataFrame(rows);out.to_csv(OUT/'dev_frozen_phenotype_gate.csv',index=False);pd.DataFrame(maps).to_parquet(OUT/'dev_all_sample_mappings.parquet',index=False)
 passed=out[(out.status=='complete')&(out.dev_shared_vs_rest_auc>=.65)&(out.mapped_candidate_fp_coverage_upper_bound>=.2)]
 m={'release_id':OUT.name,'status':'dev_only_frozen_prototype_gate','holdout_read':False,'threshold_source':'fit candidate-to-background q05 only','gate':'requires at least 3 patients with dev phenotype AUC >=0.65 and mapped FP coverage upper bound >=0.20 before any suppressor training','patients_passing':int(len(passed)),'training_authorized':bool(len(passed)>=3),'sha256':hashlib.sha256((OUT/'dev_frozen_phenotype_gate.csv').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(out.to_string(index=False));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
