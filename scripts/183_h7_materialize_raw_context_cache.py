"""Materialize immutable 20-s, 16-channel, 128-Hz raw EEG contexts for H7.

The source catalog is frozen and contains only fit/dev records.  Work is
resumable: a completed context file is never overwritten, and ``progress.json``
records every EDF as it is processed.  Windows retain raw bipolar microvolt
signals after montage mapping and resampling; feature extraction happens later.
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
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1/context_catalog_with_tn.parquet'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
RATE = 128
SECONDS = 20
SAMPLES = RATE * SECONDS


def context_id(row: pd.Series) -> str:
    key = '|'.join(str(row[k]) for k in ('partition', 'patient_id', 'phase', 'recording', 'context_end_s'))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def atomic_save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.tmp_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, x=x.astype(np.float16, copy=False))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_progress(payload: dict) -> None:
    (OUT / 'progress.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(CATALOG)
    unique = catalog.drop_duplicates(['partition', 'patient_id', 'phase', 'recording', 'context_end_s']).copy()
    unique['context_id'] = unique.apply(context_id, axis=1)
    unique = unique.sort_values(['recording', 'context_end_s']).reset_index(drop=True)
    unique.to_parquet(OUT / 'frozen_context_index.parquet', index=False)
    by_record = list(unique.groupby('recording', sort=True))
    progress_path = OUT / 'progress.json'
    done_records: set[str] = set()
    errors: dict[str, str] = {}
    if progress_path.exists():
        old = json.loads(progress_path.read_text())
        done_records = set(old.get('completed_records', []))
        errors = dict(old.get('errors', {}))
    for index, (recording, group) in enumerate(by_record, start=1):
        if recording in done_records:
            continue
        try:
            raw, sfreq, _ = load_tuh_canonical_raw_uv(DATA / recording)
            ratio = Fraction(RATE / sfreq).limit_denominator(10_000)
            raw128 = resample_poly(raw, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
            for row in group.itertuples(index=False):
                end = int(round(float(row.context_end_s) * RATE))
                start = end - SAMPLES
                if start < 0 or end > raw128.shape[-1]:
                    raise ValueError(f'window out of bounds end={row.context_end_s:.4f}s n={raw128.shape[-1]}')
                target = OUT / 'windows' / f'{row.context_id}.npz'
                if not target.exists():
                    atomic_save(target, raw128[:, start:end])
            done_records.add(recording)
            errors.pop(recording, None)
        except Exception as exc:  # preserve an auditable failure and continue
            errors[recording] = f'{type(exc).__name__}: {exc}'
        write_progress({
            'release_id': OUT.name, 'status': 'materializing',
            'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            'holdout_read': False, 'raw_representation': '16-channel bipolar microvolt, 128 Hz, 20-s causal window, float16',
            'completed_records': sorted(done_records), 'completed_record_count': len(done_records),
            'total_records': len(by_record), 'completed_contexts': sum((OUT / 'windows' / f'{r.context_id}.npz').exists() for r in unique.itertuples(index=False)),
            'total_contexts': len(unique), 'errors': errors, 'last_record': recording, 'record_index': index,
        })
    missing = [r.context_id for r in unique.itertuples(index=False) if not (OUT / 'windows' / f'{r.context_id}.npz').exists()]
    manifest = {
        'release_id': OUT.name,
        'status': 'complete' if not missing and not errors else 'complete_with_extraction_errors',
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'holdout_read': False,
        'unique_contexts': int(len(unique)), 'edf_records': int(len(by_record)),
        'completed_contexts': int(len(unique) - len(missing)), 'missing_context_ids': missing,
        'errors': errors, 'raw_representation': '16-channel bipolar microvolt, resampled to 128 Hz; 20-s causal window ending at frozen context time; float16 cache',
        'index_sha256': hashlib.sha256((OUT / 'frozen_context_index.parquet').read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    write_progress({**manifest, 'completed_records': sorted(done_records), 'completed_record_count': len(done_records), 'total_records': len(by_record)})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
