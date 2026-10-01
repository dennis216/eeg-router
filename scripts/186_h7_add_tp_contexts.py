"""Add detector true-positive contexts to the frozen H7 fit/dev catalog.

TP contexts are safety hard negatives for the suppressor.  They are derived
from the frozen expert event table (``is_fp=0`` means a seizure-matched event)
and are never taken from holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1/context_catalog_with_tn.parquet'
EVENTS = ROOT / 'outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet'
ELIG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-tp-v1'
CTX = 20.0


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing to overwrite frozen output: {OUT}')
    OUT.mkdir(parents=True)
    base = pd.read_parquet(CATALOG)
    eligible = pd.read_csv(ELIG)
    eligible = eligible[eligible.fit_eligible_standard.astype(bool)][['partition', 'patient_id', 'model']]
    events = pd.read_parquet(EVENTS)
    tp = events[(events.phase.isin(['fit', 'dev'])) & (events.is_fp == 0)].merge(
        eligible, on=['partition', 'patient_id', 'model'], how='inner')
    tp = tp[tp.start_s >= CTX].copy()
    tp_rows = tp.drop_duplicates(['partition', 'patient_id', 'phase', 'model', 'recording', 'start_s'])
    extra = pd.DataFrame({
        'partition': tp_rows.partition.astype(str), 'patient_id': tp_rows.patient_id.astype(str),
        'phase': tp_rows.phase.astype(str), 'model': tp_rows.model.astype(str),
        'recording': tp_rows.recording.astype(str), 'context_end_s': tp_rows.start_s.astype(float),
        'kind': 'tp_m', 'shared_fp': False, 'source_start_s': tp_rows.start_s.astype(float),
        'source_end_s': tp_rows.end_s.astype(float), 'matched_fp_end_s': pd.NA,
        'time_stratum': pd.NA,
    })
    full = pd.concat([base, extra], ignore_index=True, sort=False)
    full = full.drop_duplicates(['partition', 'patient_id', 'phase', 'model', 'recording', 'context_end_s', 'kind'])
    full = full.sort_values(['partition', 'patient_id', 'phase', 'model', 'recording', 'context_end_s', 'kind']).reset_index(drop=True)
    full.to_parquet(OUT / 'context_catalog_with_tn_tp.parquet', index=False)
    summ = full.groupby(['phase', 'kind', 'model']).size().rename('n_contexts').reset_index()
    summ.to_csv(OUT / 'context_summary.csv', index=False)
    manifest = {
        'release_id': OUT.name, 'status': 'catalog_with_tn_tp_no_raw_eeg_loaded',
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'source_events_sha256': hashlib.sha256(EVENTS.read_bytes()).hexdigest(),
        'holdout_read': False, 'tp_definition': 'frozen event table is_fp=0, fit/dev only, detector-specific seizure-matched event, causal context ending at detector event onset',
        'tp_contexts_added': int(len(extra)), 'total_rows': int(len(full)),
        'sha256': hashlib.sha256((OUT / 'context_catalog_with_tn_tp.parquet').read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(summ.to_string(index=False)); print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
