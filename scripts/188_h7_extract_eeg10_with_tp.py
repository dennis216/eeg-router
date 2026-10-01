"""Extract EEG10 for the exact H7 FP/TN/TP frozen context catalog."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/root/b_false_alarm_atlas')
CAT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1/context_catalog_with_tn_tp.parquet'
BASE=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1/windows'
TP=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-tp-raw-context-cache-v1/windows'
EXC=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cache-exclusion-audit-v1/excluded_contexts.csv'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-with-tp-v1'
FEATURES=('rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log','relative_delta_power','relative_theta_power','relative_alpha_power','relative_beta_power','relative_gamma_power','high_low_ratio')
def cid(r):return hashlib.sha256('|'.join(str(r[k]) for k in ('partition','patient_id','phase','recording','context_end_s')).encode()).hexdigest()[:24]
def eeg10(x):
 x=np.asarray(x,np.float32)
 if x.shape!=(16,2560):raise ValueError(x.shape)
 q=x.reshape(16,10,256).transpose(1,0,2);q-=q.mean(axis=2,keepdims=True);r=np.sqrt(np.mean(q*q,2)+1e-8);rl=np.log1p(r);f=np.fft.rfftfreq(256,1/128.);v=(f>=.5)&(f<=45.);sp=np.abs(np.fft.rfft(q,axis=2))**2;tot=sp[:,:,v].sum(2)+1e-8;p=[]
 for lo,hi in ((.5,4),(4,8),(8,13),(13,30),(30,45)):
  z=v&(f>=lo)&(f<hi);p.append(np.mean(sp[:,:,z].sum(2)/tot,1))
 a=np.column_stack((np.median(r,1),np.quantile(r,.75,1)-np.quantile(r,.25,1),np.std(rl,1),np.max(rl,1)-np.min(rl,1),*p));o=np.empty(10,np.float32);o[:4]=np.median(a[:,:4],0);o[4:9]=np.mean(a[:,4:9],0);o[9]=(o[7]+o[8])/(o[4]+o[5]+1e-8);return np.nan_to_num(o)
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);x=pd.read_parquet(CAT);x['context_id']=x.apply(cid,axis=1);missing=set(pd.read_csv(EXC).context_id.astype(str));x=x[~x.context_id.isin(missing)].copy();vec={};sources={}
 for i,z in enumerate(x.context_id.drop_duplicates(),1):
  p=BASE/f'{z}.npz';src='base'
  if not p.exists():p=TP/f'{z}.npz';src='tp'
  if not p.exists():raise FileNotFoundError(z)
  vec[z]=eeg10(np.load(p,allow_pickle=False)['x']);sources[z]=src
  if i%500==0:print(f'EEG10 {i}/{x.context_id.nunique()}',flush=True)
 a=np.stack([vec[z] for z in x.context_id]);
 for c,v in zip(FEATURES,a.T):x[c]=v.astype(float)
 x.to_parquet(OUT/'context_eeg10_with_tp.parquet',index=False);s=x.groupby(['phase','kind','model']).size().rename('n').reset_index();s.to_csv(OUT/'summary.csv',index=False)
 m={'release_id':OUT.name,'status':'complete_feature_extraction_no_model_fit','holdout_read':False,'source_catalog_sha256':hashlib.sha256(CAT.read_bytes()).hexdigest(),'excluded_contexts':len(missing),'catalog_rows':len(x),'unique_contexts':x.context_id.nunique(),'tp_cache_contexts':int(sum(v=='tp' for v in sources.values())),'features':list(FEATURES),'sha256':hashlib.sha256((OUT/'context_eeg10_with_tp.parquet').read_bytes()).hexdigest()};(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(s.to_string(index=False));print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
