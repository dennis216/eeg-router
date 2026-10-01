"""Compute frozen EEG10 features from the H7 raw-context cache.

This is a representation step, not a classifier.  It retains every catalog
row with an extractable 20-s raw context and never opens EDF files or holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1/context_catalog_with_tn.parquet'
CACHE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
EXCLUSIONS = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cache-exclusion-audit-v1/excluded_contexts.csv'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-context-features-v1'
RATE = 128.0
FRAME = 256
FEATURES = ('rms_median_uv', 'rms_iqr_uv', 'spatial_rms_dispersion', 'spatial_rms_max_min_log',
            'relative_delta_power', 'relative_theta_power', 'relative_alpha_power',
            'relative_beta_power', 'relative_gamma_power', 'high_low_ratio')


def cid(row: pd.Series) -> str:
    key = '|'.join(str(row[k]) for k in ('partition', 'patient_id', 'phase', 'recording', 'context_end_s'))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def eeg10(x: np.ndarray) -> np.ndarray:
    """EEG10 on 20 seconds: ten non-overlapping two-second frames."""
    x = np.asarray(x, np.float32)
    if x.shape != (16, 2560):
        raise ValueError(f'Expected 16x2560, got {x.shape}')
    frames = x.reshape(16, 10, FRAME).transpose(1, 0, 2)
    frames -= frames.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=2) + 1e-8)
    rms_log = np.log1p(rms)
    f = np.fft.rfftfreq(FRAME, 1.0 / RATE)
    valid = (f >= .5) & (f <= 45.)
    spectrum = np.abs(np.fft.rfft(frames, axis=2)) ** 2
    total = spectrum[:, :, valid].sum(axis=2) + 1e-8
    powers = []
    for lo, hi in ((.5,4.),(4.,8.),(8.,13.),(13.,30.),(30.,45.)):
        mask = valid & (f >= lo) & (f < hi)
        powers.append(np.mean(spectrum[:, :, mask].sum(axis=2) / total, axis=1))
    vals = np.column_stack((np.median(rms,1), np.quantile(rms,.75,1)-np.quantile(rms,.25,1),
                            np.std(rms_log,1),np.max(rms_log,1)-np.min(rms_log,1),*powers))
    out = np.empty(10,np.float32)
    out[:4] = np.median(vals[:,:4],axis=0)
    out[4:9] = np.mean(vals[:,4:9],axis=0)
    out[9] = (out[7]+out[8])/(out[4]+out[5]+1e-8)
    return np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    catalog = pd.read_parquet(CATALOG).copy()
    catalog['context_id'] = catalog.apply(cid,axis=1)
    missing = set(pd.read_csv(EXCLUSIONS)['context_id'].astype(str))
    catalog = catalog[~catalog.context_id.isin(missing)].copy()
    vectors: dict[str,np.ndarray] = {}
    for index, row in enumerate(catalog.drop_duplicates('context_id').itertuples(index=False),start=1):
        payload = np.load(CACHE/'windows'/f'{row.context_id}.npz',allow_pickle=False)
        vectors[row.context_id] = eeg10(payload['x'].astype(np.float32))
        if index % 500 == 0:
            print(f'EEG10 {index}/{catalog.context_id.nunique()}',flush=True)
    feature = np.stack([vectors[x] for x in catalog.context_id])
    for col, values in zip(FEATURES,feature.T):
        catalog[col]=values.astype(float)
    catalog.to_parquet(OUT/'context_eeg10.parquet',index=False)
    summary=catalog.groupby(['phase','kind','model']).size().rename('n').reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    manifest={'release_id':OUT.name,'status':'complete_feature_extraction_no_model_fit','holdout_read':False,
              'representation':'EEG10: amplitude(2), spatial(2), relative bandpower delta-gamma(5), high/low ratio(1), each from ten 2-s frames in a frozen 20-s causal raw EEG context',
              'source_catalog_sha256':hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
              'excluded_context_ids_sha256':hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
              'catalog_rows':int(len(catalog)),'unique_raw_contexts':int(catalog.context_id.nunique()),'features':list(FEATURES),
              'sha256':hashlib.sha256((OUT/'context_eeg10.parquet').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(summary.to_string(index=False));print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__':main()
