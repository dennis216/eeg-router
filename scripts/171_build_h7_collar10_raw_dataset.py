"""Create patient-local raw-EEG training/dev samples for H7-v2.

Only patients passing the frozen fit/development gate are included.  Samples
are 20-s causal raw EEG contexts ending at an alarm onset (or matched control
time).  No holdout data are read.
"""
from __future__ import annotations
import hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import resample_poly
from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv

ROOT=Path('/root/b_false_alarm_atlas'); GATE=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-dev-gate-corrected-oracle'; SOURCE=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v1'; OUT=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'
FS=128.; SECONDS=20.; SAMPLES=2560; RNG=20260830
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def records():
 out={}
 for part in ('train','dev','eval'):
  m=ROOT/'outputs/reports/h6-tusz-raw-pre-router-v1/manifests'/f'{part}_manifest.json'
  for r in json.loads(m.read_text())['records']:out[(part,str(r['relative_edf']))]=r
 return out
def main():
 if (OUT/'manifest.json').exists():raise RuntimeError(f'refusing overwrite {OUT}')
 OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(RNG)
 gate=pd.read_csv(GATE/'patient_dev_only_gate.csv'); selected=set(zip(gate[gate.gate_train].partition,gate[gate.gate_train].patient_id))
 event=pd.read_parquet(SOURCE/'frozen_expert_events.parquet'); shared=pd.read_parquet(GATE/'fit_dev_shared_fp_labels.parquet'); recmap=records(); rows=[]
 # The positive set is exactly the new collar label.  Private FP and matched
 # seizure events are hard negatives; clean controls are sampled per EDF.
 for part,patient in sorted(selected):
  pos=shared[(shared.partition==part)&(shared.patient_id==patient)]
  ev=event[(event.partition==part)&(event.patient_id==patient)&(event.phase.isin(['fit','dev']))]
  for phase in ('fit','dev'):
   p=pos[pos.phase==phase]; e=ev[ev.phase==phase]
   samples=[]
   for i,x in enumerate(p.itertuples(index=False)):
    samples.append((x.recording,float(x.context_end_s),'shared_fp',1,0,f'pos{i}'))
   # Deduplicate event times across detector predictions.
   seizure=e[e.matched_seizure==1].drop_duplicates(['recording','start_s'])
   private=e[e.is_fp==1].drop_duplicates(['recording','start_s'])
   for i,x in enumerate(seizure.itertuples(index=False)):
    samples.append((x.recording,min(float(x.end_s),float(x.start_s)+SECONDS),'seizure',0,1,f'sz{i}'))
   for i,x in enumerate(private.itertuples(index=False)):
    nearby=((p.recording==x.recording)&(np.abs(p.decision_time_s-float(x.start_s))<=10)).any() if len(p) else False
    if not nearby:samples.append((x.recording,float(x.start_s),'private_fp',0,0,f'private{i}'))
   # Balanced clean controls.  They are excluded from seizure/expert event
   # neighbourhoods and never carry a suppress-positive label.
   n_bg=max(1,len(p)); by_rec={r:recmap[(part,r)] for r in set(e.recording)}
   existing={r:[] for r in by_rec}
   for x in e.itertuples(index=False):existing[x.recording].append((float(x.start_s)-20,float(x.end_s)+20))
   for i in range(n_bg):
    rec=list(by_rec)[i%len(by_rec)];meta=by_rec[rec];duration=float(meta['duration_s']); end=None
    for _ in range(100):
     t=float(rng.uniform(SECONDS,max(SECONDS+.1,duration)))
     if not any(a<t and t-SECONDS<b for a,b in existing[rec]):end=t;break
    if end is not None:samples.append((rec,end,'background',0,0,f'bg{i}'))
   # Read each EDF once, construct samples, and store only selected windows.
   grouped={}
   for sample in samples:grouped.setdefault(sample[0],[]).append(sample)
   for recording, items in grouped.items():
    path=Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')/recording
    if not path.exists():raise FileNotFoundError(path)
    raw,rate,_=load_tuh_canonical_raw_uv(path); ratio=Fraction(FS/rate).limit_denominator(10000);raw=resample_poly(raw,ratio.numerator,ratio.denominator,axis=-1).astype(np.float32)
    for rec,end,kind,yshared,yrisk,key in items:
     stop=int(round(end*FS));start=stop-SAMPLES
     if start<0 or stop>raw.shape[-1]:continue
     x=raw[:,start:stop].astype(np.float16); sid=f'{part}|{patient}|{phase}|{Path(rec).stem}|{int(round(end*FS))}|{key}'
     target=OUT/'raw'/part/patient/f'{hashlib.sha1(sid.encode()).hexdigest()}.npz';target.parent.mkdir(parents=True,exist_ok=True)
     np.savez_compressed(target,x=x)
     rows.append({'sample_id':sid,'partition':part,'patient_id':patient,'phase':phase,'recording':rec,'context_end_s':end,'kind':kind,'shared_fp_label':yshared,'seizure_risk_label':yrisk,'x_path':str(target.relative_to(OUT))})
 frame=pd.DataFrame(rows)
 if frame.empty:raise RuntimeError('no samples')
 frame.to_parquet(OUT/'labels.parquet',index=False)
 audit=frame.groupby(['patient_id','phase','kind']).size().rename('n').reset_index();audit.to_csv(OUT/'label_counts.csv',index=False)
 manifest={'release_id':OUT.name,'source_gate':str(GATE.relative_to(ROOT)),'selected_patients':sorted([{'partition':p,'patient_id':x} for p,x in selected],key=lambda x:(x['partition'],x['patient_id'])),'input':{'raw_only':True,'channels':16,'sampling_hz':128,'context_seconds':20,'shape':[16,SAMPLES],'expert_outputs_in_input':False},'labels':{'shared_fp_positive':'2-of-3 frozen-expert FP onsets within ±10 s','hard_negatives':['true seizure','private FP','matched clean background']},'holdout_read':False,'samples':len(frame),'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n');print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
