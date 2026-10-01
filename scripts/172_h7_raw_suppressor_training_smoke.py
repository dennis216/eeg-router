"""H7-v3 patient-local shared-FP suppressor: 25-update trainability smoke.

This script is deliberately *not* a final experiment.  It uses only each
patient's chronological ``fit`` samples for fitting and ``dev`` samples for
diagnostics.  There is no holdout read, threshold choice, or alarm-suppression
claim.  Its sole purpose is to reject a collapsing raw-EEG model before an
expensive formal fit.
"""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/root/b_false_alarm_atlas')
DATA=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-collar10-raw-data'
OUT=ROOT/'outputs/reports/h7-personalized-consensus-fp-suppressor-v3-training-smoke'
SEED=20260830; STEPS=25; BATCH=12

class TinySuppressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(16,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.shared=nn.Linear(128,1); self.risk=nn.Linear(128,1)
    def forward(self,x):
        z=self.net(x); return self.shared(z).squeeze(1),self.risk(z).squeeze(1)

def load(row):
    x=np.load(DATA/row.x_path)['x'].astype(np.float32)
    # Causal per-window robust normalization; no corpus or dev statistics.
    med=np.median(x,axis=1,keepdims=True); mad=np.median(np.abs(x-med),axis=1,keepdims=True)
    return np.clip((x-med)/(1.4826*mad+1e-3),-12,12)

def auc(y,s):
    if len(np.unique(y))<2:return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,s))

def main():
    if (OUT/'manifest.json').exists(): raise RuntimeError(f'refusing overwrite: {OUT}')
    OUT.mkdir(parents=True); (OUT/'checkpoints').mkdir()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    labels=pd.read_parquet(DATA/'labels.parquet')
    records=[]
    for patient,g in labels.groupby('patient_id',sort=True):
        tr=g[g.phase=='fit'].reset_index(drop=True); dv=g[g.phase=='dev'].reset_index(drop=True)
        # Formal fit needs enough positives in both chronology-preserving parts.
        if tr.shared_fp_label.sum()<20 or dv.shared_fp_label.sum()<5:
            records.append({'patient_id':patient,'status':'insufficient_for_smoke','fit_n':len(tr),'dev_n':len(dv),'fit_shared':int(tr.shared_fp_label.sum()),'dev_shared':int(dv.shared_fp_label.sum())});continue
        model=TinySuppressor(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
        # Fixed balanced batch composition: 3 shared-FP, 3 seizure, 3 private,
        # 3 background when each class exists.  This stops background from
        # teaching the specialist a trivial all-negative solution.
        pools={k:tr[tr.kind==k].index.to_numpy() for k in ['shared_fp','seizure','private_fp','background']}
        if any(len(v)==0 for v in pools.values()):
            records.append({'patient_id':patient,'status':'missing_training_class','fit_n':len(tr),'dev_n':len(dv)});continue
        first_loss=None; last_loss=None; grad_norm=0.
        for step in range(STEPS):
            ix=np.concatenate([np.random.choice(pools[k],3,replace=len(pools[k])<3) for k in pools])
            b=tr.iloc[ix]; x=torch.from_numpy(np.stack([load(r) for r in b.itertuples(index=False)]))
            ys=torch.tensor(b.shared_fp_label.to_numpy(),dtype=torch.float32); yr=torch.tensor(b.seizure_risk_label.to_numpy(),dtype=torch.float32)
            sh,ri=model(x); loss=F.binary_cross_entropy_with_logits(sh,ys)+0.5*F.binary_cross_entropy_with_logits(ri,yr)
            opt.zero_grad();loss.backward();grad_norm=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)));opt.step()
            if first_loss is None:first_loss=float(loss)
            last_loss=float(loss)
        model.eval()
        with torch.no_grad():
            x=torch.from_numpy(np.stack([load(r) for r in dv.itertuples(index=False)])); sh,ri=model(x)
            ps=torch.sigmoid(sh).numpy(); pr=torch.sigmoid(ri).numpy()
        ck=OUT/'checkpoints'/f'{patient}.pt';torch.save({'state_dict':model.state_dict(),'steps':STEPS,'seed':SEED,'input_shape':[16,2560],'normalization':'per-window channel median/MAD'},ck)
        records.append({'patient_id':patient,'status':'smoke_complete','fit_n':len(tr),'dev_n':len(dv),'fit_shared':int(tr.shared_fp_label.sum()),'dev_shared':int(dv.shared_fp_label.sum()),'loss_first':first_loss,'loss_last':last_loss,'gradient_norm_final':grad_norm,'dev_shared_auc':auc(dv.shared_fp_label.to_numpy(),ps),'dev_risk_auc':auc(dv.seizure_risk_label.to_numpy(),pr),'dev_shared_score_min':float(ps.min()),'dev_shared_score_max':float(ps.max()),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()})
    res=pd.DataFrame(records);res.to_csv(OUT/'smoke_results.csv',index=False)
    manifest={'release_id':OUT.name,'status':'training_smoke_only','source_dataset':str(DATA.relative_to(ROOT)),'holdout_read':False,'evaluation_or_threshold_selection':False,'model':{'raw_only':True,'architecture':'16x2560 -> Conv1D 32/64/128 + BN/ReLU + GAP -> shared-FP sigmoid, seizure-risk sigmoid','steps_per_patient':STEPS,'batch_composition':'3 shared-FP + 3 seizure + 3 private-FP + 3 background'},'patients':len(res),'completed':int((res.status=='smoke_complete').sum()),'sha256_smoke_results':hashlib.sha256((OUT/'smoke_results.csv').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
