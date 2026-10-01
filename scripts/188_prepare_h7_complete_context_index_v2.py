"""Freeze a complete context index after adding TP safety contexts.

Resolution order is immutable base raw cache, then immutable additional TP
cache.  Incomplete windows remain excluded and explicitly listed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1/context_catalog_with_tn_tp.parquet'
BASE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1/windows'
TP = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-tp-raw-context-cache-v1/windows'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-complete-context-index-v2'


def main() -> None:
    if (OUT / 'manifest.json').exists(): raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    cat = pd.read_parquet(CATALOG)
    uniq = cat.drop_duplicates(['partition', 'patient_id', 'phase', 'recording', 'context_end_s']).copy()
    key = uniq[['partition', 'patient_id', 'phase', 'recording', 'context_end_s']].astype(str).agg('|'.join, axis=1)
    uniq['context_id'] = key.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:24])
    paths = []; sources = []
    for x in uniq.context_id:
        bp = BASE / f'{x}.npz'; tp = TP / f'{x}.npz'
        if bp.exists(): paths.append(str(bp.relative_to(ROOT))); sources.append('base')
        elif tp.exists(): paths.append(str(tp.relative_to(ROOT))); sources.append('tp')
        else: paths.append(''); sources.append('missing')
    uniq['cache_path'] = paths; uniq['cache_source'] = sources
    complete = uniq[uniq.cache_source != 'missing'].drop(columns=['cache_source']).copy()
    missing = uniq[uniq.cache_source == 'missing'].copy()
    complete.to_parquet(OUT / 'complete_context_index.parquet', index=False)
    missing.to_csv(OUT / 'excluded_incomplete_contexts.csv', index=False)
    labels = cat.merge(complete[['partition', 'patient_id', 'phase', 'recording', 'context_end_s', 'context_id']], on=['partition', 'patient_id', 'phase', 'recording', 'context_end_s'], how='inner')
    labels.to_parquet(OUT / 'complete_context_labels.parquet', index=False)
    manifest = {'release_id': OUT.name, 'status': 'complete_context_index_frozen', 'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(), 'holdout_read': False, 'unique_contexts_catalogued': int(len(uniq)), 'unique_contexts_complete': int(len(complete)), 'unique_contexts_excluded_incomplete': int(len(missing)), 'label_rows_complete': int(len(labels)), 'excluded_by_kind': missing.groupby('kind').size().to_dict(), 'cache_sources': complete.cache_path.map(lambda x: 'tp' if 'tp-raw-context' in x else 'base').value_counts().to_dict(), 'complete_index_sha256': hashlib.sha256((OUT / 'complete_context_index.parquet').read_bytes()).hexdigest(), 'labels_sha256': hashlib.sha256((OUT / 'complete_context_labels.parquet').read_bytes()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if len(missing): print(missing[['context_id', 'patient_id', 'phase', 'model', 'recording', 'context_end_s', 'kind']].to_string(index=False))


if __name__ == '__main__': main()
