"""Materialize frozen H7 exploratory holdout contexts; no fitting or tuning."""
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

ROOT = Path('/root/b_false_alarm_atlas')
DATA = Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')
CAT = ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-catalog-v1/holdout_event_catalog.parquet'
OUT = ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-raw-v1'
RATE, N = 128, 2560


def cid(row: pd.Series) -> str:
    key = '|'.join(str(row[k]) for k in ('partition','patient_id','phase','recording','context_end_s'))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.tmp_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, x=x.astype(np.float16, copy=False))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    if (OUT/'manifest.json').exists():
        raise RuntimeError(f'Refusing overwrite {OUT}')
    OUT.mkdir(parents=True)
    catalog = pd.read_parquet(CAT)
    catalog['context_id'] = catalog.apply(cid, axis=1)
    unique = catalog.drop_duplicates('context_id').copy()
    unique.to_parquet(OUT/'context_index.parquet', index=False)
    done, errors = [], {}
    for recording, group in unique.groupby('recording', sort=True):
        try:
            raw, sfreq, _ = load_tuh_canonical_raw_uv(DATA/recording)
            ratio = Fraction(RATE/sfreq).limit_denominator(10000)
            raw = resample_poly(raw, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
            for row in group.itertuples(index=False):
                end = int(round(row.context_end_s*RATE)); start = end-N
                if start < 0 or end > raw.shape[-1]:
                    raise ValueError(f'out of bounds {row.context_end_s} {raw.shape[-1]}')
                save(OUT/'windows'/f'{row.context_id}.npz', raw[:,start:end])
            done.append(recording)
        except Exception as exc:
            errors[recording] = f'{type(exc).__name__}: {exc}'
        (OUT/'progress.json').write_text(json.dumps({
            'completed_records':len(done), 'total_records':int(unique.recording.nunique()),
            'completed_contexts':sum((OUT/'windows'/f'{z}.npz').exists() for z in unique.context_id),
            'total_contexts':len(unique), 'errors':errors, 'last_record':recording,
        }, indent=2, sort_keys=True)+'\n')
    missing = [z for z in unique.context_id if not (OUT/'windows'/f'{z}.npz').exists()]
    manifest = {
        'release_id':OUT.name, 'status':'complete' if not missing and not errors else 'complete_with_errors',
        'exploratory':True, 'holdout_accessed':True,
        'source_catalog_sha256':hashlib.sha256(CAT.read_bytes()).hexdigest(),
        'unique_contexts':len(unique), 'edf_records':int(unique.recording.nunique()),
        'missing_context_ids':missing, 'errors':errors,
        'raw_representation':'16 channel bipolar microvolt, 128 Hz, causal 20-s window ending at frozen event onset',
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
