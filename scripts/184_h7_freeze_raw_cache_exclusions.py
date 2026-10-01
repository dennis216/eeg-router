"""Freeze and explain raw-context extraction exclusions before H7 modeling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
CACHE = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-context-cache-v1'
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1/context_catalog_with_tn.parquet'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cache-exclusion-audit-v1'


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    cache_manifest = json.loads((CACHE / 'manifest.json').read_text())
    index = pd.read_parquet(CACHE / 'frozen_context_index.parquet')
    catalog = pd.read_parquet(CATALOG)
    missing = set(cache_manifest['missing_context_ids'])
    affected = index[index.context_id.isin(missing)].merge(
        catalog, on=['partition', 'patient_id', 'phase', 'recording', 'context_end_s'], how='left', suffixes=('_index', '')
    ).sort_values(['recording', 'context_end_s', 'kind'])
    affected.to_csv(OUT / 'excluded_contexts.csv', index=False)
    manifest = {
        'release_id': OUT.name,
        'status': 'frozen_exclusion_audit',
        'decision': 'exclude only the three frozen context identities that cannot be extracted; do not clip, move, or replace their windows',
        'holdout_read': False,
        'source_cache_manifest_sha256': hashlib.sha256((CACHE / 'manifest.json').read_bytes()).hexdigest(),
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'excluded_unique_contexts': int(len(missing)),
        'excluded_catalog_rows': int(len(affected)),
        'errors': cache_manifest['errors'],
        'retained_unique_contexts': int(cache_manifest['completed_contexts']),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(affected.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
