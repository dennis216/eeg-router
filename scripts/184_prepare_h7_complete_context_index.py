"""Freeze the complete-window subset for H7 without changing the raw cache.

The materializer records three source EDF timing inconsistencies.  This audit
removes only contexts whose full 20-s window is absent; it never truncates,
pads, or silently replaces a window.  All exclusions are retained in the
manifest for transparent denominator accounting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1/context_catalog_with_tn.parquet'
CACHE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v1'


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing to overwrite frozen output: {OUT}')
    OUT.mkdir(parents=True)
    catalog = pd.read_parquet(CATALOG)
    unique = catalog.drop_duplicates(['partition', 'patient_id', 'phase', 'recording', 'context_end_s']).copy()
    # The raw-cache index uses the same deterministic context-id function.
    key = unique[['partition', 'patient_id', 'phase', 'recording', 'context_end_s']].astype(str).agg('|'.join, axis=1)
    unique['context_id'] = key.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:24])
    unique['cache_path'] = unique.context_id.map(lambda x: str((CACHE / 'windows' / f'{x}.npz').relative_to(ROOT)))
    unique['cache_exists'] = unique.context_id.map(lambda x: (CACHE / 'windows' / f'{x}.npz').exists())
    missing = unique[~unique.cache_exists].copy()
    complete = unique[unique.cache_exists].drop(columns=['cache_exists']).copy()
    complete.to_parquet(OUT / 'complete_context_index.parquet', index=False)
    missing.to_csv(OUT / 'excluded_incomplete_contexts.csv', index=False)
    # Preserve all labels/roles for complete contexts; labels are joined to the
    # feature table later, so no event row is lost here.
    labels = catalog.merge(complete[['partition', 'patient_id', 'phase', 'recording', 'context_end_s', 'context_id']],
                           on=['partition', 'patient_id', 'phase', 'recording', 'context_end_s'], how='inner')
    labels.to_parquet(OUT / 'complete_context_labels.parquet', index=False)
    manifest = {
        'release_id': OUT.name,
        'status': 'complete_context_index_frozen',
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'source_raw_cache_manifest_sha256': hashlib.sha256((CACHE / 'manifest.json').read_bytes()).hexdigest(),
        'holdout_read': False,
        'unique_contexts_catalogued': int(len(unique)),
        'unique_contexts_complete': int(len(complete)),
        'unique_contexts_excluded_incomplete': int(len(missing)),
        'label_rows_complete': int(len(labels)),
        'excluded_reason': 'full 20-s context was outside the actual EDF after resampling; no truncation or padding',
        'excluded_by_kind': missing.groupby('kind').size().to_dict(),
        'complete_index_sha256': hashlib.sha256((OUT / 'complete_context_index.parquet').read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256((OUT / 'complete_context_labels.parquet').read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if len(missing):
        print('Excluded incomplete contexts:')
        print(missing[['context_id', 'patient_id', 'phase', 'model', 'recording', 'context_end_s', 'kind']].to_string(index=False))


if __name__ == '__main__':
    main()
