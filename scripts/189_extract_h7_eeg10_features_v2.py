"""Extract the frozen 10-feature representation for H7 FP/TN/TP contexts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path('/root/b_false_alarm_atlas')
INDEX = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v2'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-features-v2'
FEATURES = ['rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log','relative_delta_power','relative_theta_power','relative_alpha_power','relative_beta_power','relative_gamma_power','high_low_ratio']
def feat(x):
    if x.shape != (16,2560): raise ValueError(f'expected 16x2560, got {x.shape}')
    x=x.astype(np.float32); x=x-x.mean(1,keepdims=True); rms=np.sqrt(np.mean(x*x,1)+1e-8); rl=np.log1p(rms); sp=np.abs(np.fft.rfft(x,axis=1))**2; f=np.fft.rfftfreq(x.shape[1],1/128); valid=(f>=.5)&(f<=45); total=sp[:,valid].sum(1)+1e-8; o={'rms_median_uv':float(np.median(rms)),'rms_iqr_uv':float(np.quantile(rms,.75)-np.quantile(rms,.25)),'spatial_rms_dispersion':float(np.std(rl)),'spatial_rms_max_min_log':float(np.max(rl)-np.min(rl))}
    for n,lo,hi in [('delta',.5,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]: o[f'relative_{n}_power']=float(np.mean(sp[:,valid&(f>=lo)&(f<hi)].sum(1)/total))
    o['high_low_ratio']=float((o['relative_beta_power']+o['relative_gamma_power'])/(o['relative_delta_power']+o['relative_theta_power']+1e-8))
    if not np.isfinite(list(o.values())).all(): raise ValueError('nonfinite feature')
    return o
def main():
    if (OUT/'manifest.json').exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True); idx=pd.read_parquet(INDEX/'complete_context_index.parquet'); rows=[]
    for r in idx.itertuples(index=False):
        with np.load(ROOT/r.cache_path) as z: v=feat(z['x'])
        v.update({'context_id':r.context_id,'partition':r.partition,'patient_id':r.patient_id,'phase':r.phase,'recording':r.recording,'context_end_s':float(r.context_end_s)}); rows.append(v)
    d=pd.DataFrame(rows).sort_values(['partition','patient_id','phase','recording','context_end_s']).reset_index(drop=True); d.to_parquet(OUT/'context_eeg10_features.parquet',index=False)
    m={'release_id':OUT.name,'status':'complete','holdout_read':False,'source_index_sha256':hashlib.sha256((INDEX/'complete_context_index.parquet').read_bytes()).hexdigest(),'source_index_manifest_sha256':hashlib.sha256((INDEX/'manifest.json').read_bytes()).hexdigest(),'contexts':int(len(d)),'patients':int(d.patient_id.nunique()),'recordings':int(d.recording.nunique()),'feature_names':FEATURES,'fit_or_selection_performed':False,'output_sha256':hashlib.sha256((OUT/'context_eeg10_features.parquet').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n'); print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__': main()
