"""Materialize only the additional TP safety contexts missing from the H7 cache.

Existing FP/TN cache is never modified.  Contexts already present in that cache
are linked by identity; only new TP windows are written here.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv

ROOT=Path('/root/b_false_alarm_atlas'); DATA=Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')
CATALOG=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1/context_catalog_with_tn_tp.parquet'
BASE=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-tp-raw-context-cache-v1'
RATE=128; N=2560

def cid(r:pd.Series)->str:
 return hashlib.sha256('|'.join(str(r[k]) for k in ('partition','patient_id','phase','recording','context_end_s')).encode()).hexdigest()[:24]
def save(p:Path,x:np.ndarray)->None:
 p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.tmp_',suffix='.npz',dir=p.parent);os.close(fd)
 try: np.savez_compressed(tmp,x=x.astype(np.float16,copy=False));os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def main()->None:
 if (OUT/'manifest.json').exists():raise RuntimeError(f'Refusing overwrite: {OUT}')
 OUT.mkdir(parents=True);c=pd.read_parquet(CATALOG);c['context_id']=c.apply(cid,axis=1)
 # Only TP contexts that are absent from the immutable base cache need EDF I/O.
 tp=c[c.kind.eq('tp_m')].drop_duplicates('context_id').copy();todo=tp[~tp.context_id.map(lambda z:(BASE/'windows'/f'{z}.npz').exists())].copy()
 todo.to_parquet(OUT/'tp_context_index.parquet',index=False);done=[];errors={}
 for rec,g in todo.groupby('recording',sort=True):
  try:
   raw,sfreq,_=load_tuh_canonical_raw_uv(DATA/rec);ratio=Fraction(RATE/sfreq).limit_denominator(10000);raw=resample_poly(raw,ratio.numerator,ratio.denominator,axis=-1).astype(np.float32)
   for r in g.itertuples(index=False):
    end=int(round(float(r.context_end_s)*RATE));start=end-N
    if start<0 or end>raw.shape[-1]:raise ValueError(f'window out of bounds end={r.context_end_s}s n={raw.shape[-1]}')
    save(OUT/'windows'/f'{r.context_id}.npz',raw[:,start:end])
   done.append(rec)
  except Exception as exc:errors[rec]=f'{type(exc).__name__}: {exc}'
  (OUT/'progress.json').write_text(json.dumps({'completed_records':len(done),'total_records':int(todo.recording.nunique()),'completed_contexts':sum((OUT/'windows'/f'{z}.npz').exists() for z in todo.context_id),'total_contexts':len(todo),'errors':errors,'last_record':rec},indent=2,sort_keys=True)+'\n')
 missing=[z for z in todo.context_id if not(OUT/'windows'/f'{z}.npz').exists()]
 m={'release_id':OUT.name,'status':'complete' if not missing and not errors else 'complete_with_errors','holdout_read':False,'source_catalog_sha256':hashlib.sha256(CATALOG.read_bytes()).hexdigest(),'tp_unique_contexts':int(len(tp)),'already_in_base_cache':int(len(tp)-len(todo)),'new_contexts':int(len(todo)),'completed_new_contexts':int(len(todo)-len(missing)),'missing_context_ids':missing,'errors':errors,'raw_representation':'16 channel bipolar microvolt, 128 Hz, causal 20-s window ending at frozen TP event onset'}
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__':main()
