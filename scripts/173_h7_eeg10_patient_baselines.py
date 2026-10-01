"""Patient-local EEG10 logistic baseline for H7 shared-FP specialist.

This is a low-capacity, raw-EEG-only comparator to the CNN smoke.  Features
are extracted from the same frozen 20-s contexts: two amplitude, two spatial,
five band-power and one high/low features.  Scaling and classifier fit use
only each patient's chronological fit segment; dev is read only for score.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path('/root/b_false_alarm_atlas'); DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'; OUT=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-eeg10-baselines-v2-numpy-compatible'
FS=128.
def eeg10(rel):
 x=np.load(DATA/rel)['x'].astype(np.float32); rms=np.sqrt(np.mean(x*x,axis=1))
 f,p=welch(x,fs=FS,nperseg=256,axis=-1); p=np.mean(p,axis=0)
 def band(a,b): return float(np.trapezoid(p[(f>=a)&(f<b)],f[(f>=a)&(f<b)]))
 d,t,a,b,g=[band(*q) for q in [(0.5,4),(4,8),(8,13),(13,30),(30,45)]]
 return [float(np.median(rms)),float(np.subtract(*np.percentile(rms,[75,25]))),float(np.std(rms)),float(np.max(rms)-np.min(rms)),*(float(np.log10(v+1e-8)) for v in [d,t,a,b,g]),float(np.log10((b+g+1e-8)/(d+t+1e-8)))]
def score(y,s):return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True); x=pd.read_parquet(DATA/'labels.parquet'); cache={};rows=[]
 for patient,g in x.groupby('patient_id',sort=True):
  tr=g[g.phase=='fit'].copy();dv=g[g.phase=='dev'].copy()
  if tr.shared_fp_label.sum()<20 or dv.shared_fp_label.sum()<5 or tr.seizure_risk_label.sum()<1 or dv.seizure_risk_label.sum()<1:
   rows.append({'patient_id':patient,'status':'insufficient_for_baseline','fit_n':len(tr),'dev_n':len(dv),'fit_shared':int(tr.shared_fp_label.sum()),'dev_shared':int(dv.shared_fp_label.sum())});continue
  for r in pd.concat([tr,dv]).itertuples(index=False):cache[r.sample_id]=eeg10(r.x_path)
  Xtr=np.array([cache[v] for v in tr.sample_id]);Xdv=np.array([cache[v] for v in dv.sample_id]);sc=StandardScaler().fit(Xtr);Xtr=sc.transform(Xtr);Xdv=sc.transform(Xdv)
  sh=LogisticRegression(C=.2,class_weight='balanced',max_iter=2000,random_state=20260830).fit(Xtr,tr.shared_fp_label);ri=LogisticRegression(C=.2,class_weight='balanced',max_iter=2000,random_state=20260830).fit(Xtr,tr.seizure_risk_label)
  rows.append({'patient_id':patient,'status':'complete','fit_n':len(tr),'dev_n':len(dv),'fit_shared':int(tr.shared_fp_label.sum()),'dev_shared':int(dv.shared_fp_label.sum()),'dev_shared_auc':score(dv.shared_fp_label,sh.predict_proba(Xdv)[:,1]),'dev_risk_auc':score(dv.seizure_risk_label,ri.predict_proba(Xdv)[:,1]),'shared_coef_l2':float(np.linalg.norm(sh.coef_))})
 z=pd.DataFrame(rows);z.to_csv(OUT/'patient_eeg10_dev_results.csv',index=False)
 manifest={'release_id':OUT.name,'status':'dev_only_baseline','source_dataset':str(DATA.relative_to(ROOT)),'holdout_read':False,'input':'same raw EEG contexts transformed to frozen 10 handcrafted EEG features; no detector outputs','fit':'patient-local chronological fit only','feature_names':['rms_median','rms_iqr','spatial_rms_std','spatial_rms_range','delta','theta','alpha','beta','gamma','high_low_ratio'],'model':'class-balanced logistic regression C=0.2; scaler fit only on fit','n_patients':len(z),'sha256':hashlib.sha256((OUT/'patient_eeg10_dev_results.csv').read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(z.to_string(index=False))
if __name__=='__main__':main()
