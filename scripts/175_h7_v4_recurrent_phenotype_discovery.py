"""Fit-only recurrent shared-FP phenotype discovery for H7-v4.

Temporal 2-of-3 co-alarm makes an event a *candidate*, not a training label.
This script uses only each patient's early ``fit`` raw EEG contexts to define
recurrent phenotypes.  A positive prototype must recur in >=3 candidates from
>=2 EDFs.  Dev samples are assigned to frozen prototypes only; they never
alter scaling, thresholds, graph edges or component membership.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

ROOT=Path('/root/b_false_alarm_atlas')
DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'
OUT=ROOT/'outputs/reports/h7-personalized-consensus-phenotype-suppressor-v4-fit-only-discovery'
FS=128.; MIN_SIZE=3; MIN_EDFS=2

def feat(rel):
 x=np.load(DATA/rel)['x'].astype(np.float32);r=np.sqrt((x*x).mean(1));f,p=welch(x,fs=FS,nperseg=256,axis=-1);p=p.mean(0)
 def power(lo,hi):
  z=(f>=lo)&(f<hi);return float(np.trapezoid(p[z],f[z]))
 d,t,a,b,g=[power(*z) for z in [(0.5,4),(4,8),(8,13),(13,30),(30,45)]]
 return np.array([np.median(r),np.subtract(*np.percentile(r,[75,25])),np.std(r),np.ptp(r),*(np.log10(v+1e-8) for v in [d,t,a,b,g]),np.log10((b+g+1e-8)/(d+t+1e-8))],dtype=float)

def components(adj):
 out=[];seen=set()
 for start in range(len(adj)):
  if start in seen:continue
  q=[start];seen.add(start);c=[]
  while q:
   i=q.pop();c.append(i)
   for j in np.flatnonzero(adj[i]):
    if int(j) not in seen:seen.add(int(j));q.append(int(j))
  out.append(c)
 return out

def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite: {OUT}')
 OUT.mkdir(parents=True)
 x=pd.read_parquet(DATA/'labels.parquet');cache={};report=[];assigned=[];prototypes=[]
 for patient,g in x.groupby('patient_id',sort=True):
  fit=g[(g.phase=='fit')&(g.kind=='shared_fp')].copy().reset_index(drop=True)
  bg=g[(g.phase=='fit')&(g.kind=='background')].copy().reset_index(drop=True)
  dev=g[(g.phase=='dev')&(g.kind=='shared_fp')].copy().reset_index(drop=True)
  if len(fit)<MIN_SIZE or len(bg)<3:
   report.append({'patient_id':patient,'status':'insufficient_fit_candidates','fit_candidates':len(fit),'fit_background':len(bg),'dev_candidates':len(dev)});continue
  for r in pd.concat([fit,bg,dev]).itertuples(index=False):cache[r.sample_id]=feat(r.x_path)
  A=np.stack([cache[s] for s in fit.sample_id]);B=np.stack([cache[s] for s in bg.sample_id]);D=np.stack([cache[s] for s in dev.sample_id]) if len(dev) else np.empty((0,10))
  sc=StandardScaler().fit(np.vstack([A,B]));A=sc.transform(A);B=sc.transform(B);D=sc.transform(D) if len(D) else D
  # Patient-local empirical null: distances from a shared candidate to ordinary
  # early background.  The 5th percentile is frozen before looking at dev.
  null=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(2)).ravel(); threshold=float(np.quantile(null,.05))
  dist=np.sqrt(((A[:,None,:]-A[None,:,:])**2).sum(2));same=(fit.recording.to_numpy()[:,None]==fit.recording.to_numpy()[None,:]);adj=(dist<=threshold)&(~same)&(~np.eye(len(fit),dtype=bool))
  comps=[c for c in components(adj) if len(c)>=MIN_SIZE and fit.iloc[c].recording.nunique()>=MIN_EDFS]
  # A dev candidate maps only to the closest frozen component prototype and
  # only if it meets the patient's frozen early-data threshold.
  proto=np.stack([A[c].mean(0) for c in comps]) if comps else np.empty((0,10))
  dmap=np.sqrt(((D[:,None,:]-proto[None,:,:])**2).sum(2)) if len(D) and len(proto) else np.empty((len(D),0))
  for ci,c in enumerate(comps):
   pid=f'{patient}:P{ci+1}'
   prototypes.append({'patient_id':patient,'phenotype_id':pid,'fit_events':len(c),'fit_edfs':int(fit.iloc[c].recording.nunique()),'fit_sample_ids':json.dumps(fit.iloc[c].sample_id.tolist()),'distance_threshold':threshold,'prototype_eeg10':json.dumps(proto[ci].tolist())})
   for ii in c:assigned.append({'patient_id':patient,'phase':'fit','sample_id':fit.iloc[ii].sample_id,'phenotype_id':pid,'is_recurrent_shared_phenotype':True,'distance_to_prototype':float(np.linalg.norm(A[ii]-proto[ci]))})
  for i,r in enumerate(dev.itertuples(index=False)):
   if dmap.shape[1]:
    j=int(np.argmin(dmap[i]));match=bool(dmap[i,j]<=threshold);pid=f'{patient}:P{j+1}' if match else None;dd=float(dmap[i,j])
   else:match=False;pid=None;dd=None
   assigned.append({'patient_id':patient,'phase':'dev','sample_id':r.sample_id,'phenotype_id':pid,'is_recurrent_shared_phenotype':match,'distance_to_prototype':dd})
  report.append({'patient_id':patient,'status':'complete','fit_candidates':len(fit),'fit_background':len(bg),'dev_candidates':len(dev),'feature_distance_threshold_q05_background':threshold,'recurrent_phenotypes':len(comps),'fit_phenotype_events':sum(map(len,comps)),'dev_mapped_events':int(sum(a['is_recurrent_shared_phenotype'] for a in assigned if a['patient_id']==patient and a['phase']=='dev') )})
 pd.DataFrame(report).to_csv(OUT/'patient_fit_only_phenotype_audit.csv',index=False)
 pd.DataFrame(prototypes,columns=['patient_id','phenotype_id','fit_events','fit_edfs','fit_sample_ids','distance_threshold','prototype_eeg10']).to_parquet(OUT/'frozen_phenotype_prototypes.parquet',index=False)
 pd.DataFrame(assigned,columns=['patient_id','phase','sample_id','phenotype_id','is_recurrent_shared_phenotype','distance_to_prototype']).to_parquet(OUT/'fit_dev_phenotype_assignments.parquet',index=False)
 manifest={'release_id':OUT.name,'status':'fit_only_discovery_no_suppressor_training','source_dataset':str(DATA.relative_to(ROOT)),'holdout_read':False,'candidate':'2-of-3 frozen expert FP onset collar ±10 s, no seizure','phenotype':'fit-only EEG10 graph component; feature distance <= patient candidate-background q05; cross-EDF edge; >=3 events and >=2 EDFs','dev_role':'frozen prototype mapping only','input':'20-s raw EEG transformed to 10 EEG/QC features; no detector probabilities or model-ID features','feature_names':['rms_median','rms_iqr','spatial_rms_std','spatial_rms_range','delta','theta','alpha','beta','gamma','high_low_ratio'],'sha256_patient_audit':hashlib.sha256((OUT/'patient_fit_only_phenotype_audit.csv').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
