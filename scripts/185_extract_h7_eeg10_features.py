"""Extract the prespecified 10 interpretable EEG features from H7 windows.

Only the frozen fit/dev raw-context cache is read.  No labels, thresholds, or
holdout records are used in feature construction.  The output is one row per
unique raw context; FP/TN/shared/seizure roles remain in the separate frozen
label table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
INDEX_DIR = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v1'
CACHE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-features-v1'
FEATURES = [
    'rms_median_uv', 'rms_iqr_uv', 'spatial_rms_dispersion',
    'spatial_rms_max_min_log', 'relative_delta_power',
    'relative_theta_power', 'relative_alpha_power', 'relative_beta_power',
    'relative_gamma_power', 'high_low_ratio',
]


def one(x: np.ndarray) -> dict[str, float]:
    if x.shape != (16, 2560):
        raise ValueError(f'expected 16x2560 window, got {x.shape}')
    x = x.astype(np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-8)
    rms_log = np.log1p(rms)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], d=1 / 128.0)
    valid = (f >= 0.5) & (f <= 45.0)
    total = spec[:, valid].sum(axis=1) + 1e-8
    out = {
        'rms_median_uv': float(np.median(rms)),
        'rms_iqr_uv': float(np.quantile(rms, .75) - np.quantile(rms, .25)),
        'spatial_rms_dispersion': float(np.std(rms_log)),
        'spatial_rms_max_min_log': float(np.max(rms_log) - np.min(rms_log)),
    }
    bands = {'delta': (.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    for name, (low, high) in bands.items():
        mask = valid & (f >= low) & (f < high)
        out[f'relative_{name}_power'] = float(np.mean(spec[:, mask].sum(axis=1) / total))
    out['high_low_ratio'] = float((out['relative_beta_power'] + out['relative_gamma_power']) /
                                  (out['relative_delta_power'] + out['relative_theta_power'] + 1e-8))
    if not np.isfinite(np.asarray(list(out.values()), dtype=np.float64)).all():
        raise ValueError('non-finite EEG10 feature')
    return out


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing to overwrite frozen output: {OUT}')
    OUT.mkdir(parents=True)
    idx = pd.read_parquet(INDEX_DIR / 'complete_context_index.parquet')
    rows = []
    for row in idx.itertuples(index=False):
        path = ROOT / row.cache_path
        with np.load(path) as z:
            values = one(z['x'])
        values.update({'context_id': row.context_id, 'partition': row.partition,
                       'patient_id': row.patient_id, 'phase': row.phase,
                       'recording': row.recording, 'context_end_s': float(row.context_end_s)})
        rows.append(values)
    frame = pd.DataFrame(rows).sort_values(['partition', 'patient_id', 'phase', 'recording', 'context_end_s']).reset_index(drop=True)
    frame.to_parquet(OUT / 'context_eeg10_features.parquet', index=False)
    manifest = {
        'release_id': OUT.name, 'status': 'complete', 'holdout_read': False,
        'source_index_sha256': hashlib.sha256((INDEX_DIR / 'complete_context_index.parquet').read_bytes()).hexdigest(),
        'source_cache_manifest_sha256': hashlib.sha256((CACHE / 'manifest.json').read_bytes()).hexdigest(),
        'contexts': int(len(frame)), 'patients': int(frame.patient_id.nunique()),
        'recordings': int(frame.recording.nunique()), 'feature_names': FEATURES,
        'feature_definition': '20-s causal raw bipolar EEG; channel RMS and spatial summaries; relative 0.5-4/4-8/8-13/13-30/30-45 Hz powers; beta+gamma over delta+theta',
        'fit_or_selection_performed': False, 'output_sha256': hashlib.sha256((OUT / 'context_eeg10_features.parquet').read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
