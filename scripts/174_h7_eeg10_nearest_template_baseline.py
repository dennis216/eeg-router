"""H7 patient-local positive-template baseline on the same EEG10 features."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path('/root/b_false_alarm_atlas'); DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'; OUT=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-eeg10-template-baseline-v1'; FS=128.
def feat(rel):
 x=np.load(DATA/rel)['x'].astype(np.float32);r=np.sqrt((x*x).mean(1));f,p=welch(x,fs=FS,nperseg=256,axis=-1);p=p.mean(0)
 def q(lo,hi):
  z=(f>=lo)&(f<hi);return float(np.trapezoid(p[z],f[z]))
 d,t,a,b,g=[q(*z) for z in [(0.5,4),(4,8),(8,13),(13,30),(30,45)]]
 return [np.median(r),np.subtract(*np.percentile(r,[75,25])),np.std(r),np.ptp(r),*(np.log10(z+1e-8) for z in [d,t,a,b,g]),np.log10((b+g+1e-8)/(d+t+1e-8))]
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True);x=pd.read_parquet(DATA/'labels.parquet');cache={};rows=[]
 for pat,g in x.groupby('patient_id',sort=True):
  tr=g[g.phase=='fit'].copy();dv=g[g.phase=='dev'].copy()
  if tr.shared_fp_label.sum()<20 or dv.shared_fp_label.sum()<5:
   rows.append({'patient_id':pat,'status':'insufficient'});continue
  for r in pd.concat([tr,dv]).itertuples(index=False):cache[r.sample_id]=feat(r.x_path)
  A=np.array([cache[z] for z in tr.sample_id]);B=np.array([cache[z] for z in dv.sample_id]);sc=StandardScaler().fit(A);A=sc.transform(A);B=sc.transform(B);P=A[tr.shared_fp_label.to_numpy().astype(bool)]
  # Mean distance to three nearest early shared-FP prototypes.  Unlike a CNN,
  # this has no fitted discriminative parameters beyond fit-only scaling.
  dist=((B[:,None,:]-P[None,:,:])**2).sum(2)**.5;k=min(3,len(P));score=-np.partition(dist,k-1,axis=1)[:,:k].mean(1)
  rows.append({'patient_id':pat,'status':'complete','fit_shared':len(P),'dev_shared':int(dv.shared_fp_label.sum()),'dev_shared_auc':float(roc_auc_score(dv.shared_fp_label,score)),'median_nearest_distance':float(np.median(-score))})
 z=pd.DataFrame(rows);z.to_csv(OUT/'template_dev_results.csv',index=False)
 m={'release_id':OUT.name,'status':'dev_only_patient_template_baseline','source_dataset':str(DATA.relative_to(ROOT)),'holdout_read':False,'method':'fit-only EEG10 scaler; score = negative mean Euclidean distance to 3 nearest fit shared-FP templates','sha256':hashlib.sha256((OUT/'template_dev_results.csv').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(z.to_string(index=False))
if __name__=='__main__':main()
