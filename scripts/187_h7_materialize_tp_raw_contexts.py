"""Materialize only the additional TP windows for H7 safety evaluation.

The original raw cache is immutable.  Existing context IDs are reused by
reference; only newly introduced TP contexts are written to this separate
directory.
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

ROOT = Path('/root/b_false_alarm_atlas')
DATA = Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1/context_catalog_with_tn_tp.parquet'
BASE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-tp-raw-context-cache-v1'
RATE = 128
SAMPLES = 20 * RATE


def cid(row: pd.Series) -> str:
    key = '|'.join(str(row[k]) for k in ('partition', 'patient_id', 'phase', 'recording', 'context_end_s'))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp_', suffix='.npz', dir=path.parent); os.close(fd)
    try:
        np.savez_compressed(tmp, x=x.astype(np.float16, copy=False)); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    if (OUT / 'manifest.json').exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    cat = pd.read_parquet(CATALOG)
    uniq = cat.drop_duplicates(['partition', 'patient_id', 'phase', 'recording', 'context_end_s']).copy()
    uniq['context_id'] = uniq.apply(cid, axis=1)
    # Only TP contexts not already present in the immutable base cache need EDF I/O.
    new = uniq[(uniq.kind == 'tp_m') & ~uniq.context_id.map(lambda x: (BASE / 'windows' / f'{x}.npz').exists())].copy()
    jobs = list(new.groupby('recording', sort=True))
    done_records = []
    errors = {}
    for recording, group in jobs:
        try:
            raw, sfreq, _ = load_tuh_canonical_raw_uv(DATA / recording)
            ratio = Fraction(RATE / sfreq).limit_denominator(10_000)
            raw128 = resample_poly(raw, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
            for row in group.itertuples(index=False):
                end = int(round(float(row.context_end_s) * RATE)); start = end - SAMPLES
                target = OUT / 'windows' / f'{row.context_id}.npz'
                if start < 0 or end > raw128.shape[-1]: raise ValueError(f'out-of-bounds end={row.context_end_s} n={raw128.shape[-1]}')
                if not target.exists(): save(target, raw128[:, start:end])
            done_records.append(recording)
        except Exception as exc: errors[recording] = f'{type(exc).__name__}: {exc}'
    missing = [x for x in new.context_id if not (OUT / 'windows' / f'{x}.npz').exists()]
    manifest = {
        'release_id': OUT.name, 'status': 'complete' if not missing and not errors else 'complete_with_extraction_errors',
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(), 'holdout_read': False,
        'new_tp_contexts': int(len(new)), 'edf_records': int(len(jobs)), 'completed_contexts': int(len(new) - len(missing)),
        'missing_context_ids': missing, 'errors': errors, 'raw_representation': '16-channel bipolar microvolt, 128 Hz, 20-s causal window, float16',
        'sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__': main()
