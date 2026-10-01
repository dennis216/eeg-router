"""Add deterministic matched true-negative contexts to the frozen H7 catalog.

For every detector-specific FP context, this selects one non-seizure, no-alarm
20-s context from the same patient, EDF, phase, and recording-time tercile.
The selection is deterministic and never reads a holdout record.  TN contexts
are controls for template discovery; they prevent a patient-specific template
from merely learning that patient's ordinary background EEG.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
DATA = Path('/mnt/d/TUH_EEG/TUSZ_v2.0.6')
CATALOG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-v1/context_catalog.parquet'
EVENTS = ROOT / 'outputs/reports/h7-personalized-consensus-fp-suppressor-v1/frozen_expert_events.parquet'
MANIFEST_DIR = ROOT / 'outputs/reports/h6-tusz-raw-pre-router-v1/manifests'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-context-catalog-with-tn-v1'
CTX = 20.0
EXCLUSION_S = 30.0
GRID_S = 10.0


def seizure_intervals(recording: str) -> list[tuple[float, float]]:
    path = DATA / recording
    ann = Path(str(path.with_suffix('.csv')) + '_bi')
    lines = [line.strip() for line in ann.read_text(encoding='utf-8-sig').splitlines()
             if line.strip() and not line.startswith('#')]
    return [(float(row['start_time']), float(row['stop_time']))
            for row in csv.DictReader(lines)
            if str(row.get('label', '')).lower() == 'seiz']


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def stable_choice(values: list[float], key: str) -> float:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return values[int(digest[:16], 16) % len(values)]


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing to overwrite frozen output: {OUT}')
    OUT.mkdir(parents=True)
    catalog = pd.read_parquet(CATALOG)
    events = pd.read_parquet(EVENTS)
    events = events[events.phase.isin(['fit', 'dev'])].copy()

    durations: dict[str, float] = {}
    for name in ('train_manifest.json', 'dev_manifest.json', 'eval_manifest.json'):
        manifest = json.loads((MANIFEST_DIR / name).read_text())
        durations.update({r['relative_edf']: float(r['duration_s']) for r in manifest['records']})

    fp = catalog[catalog.kind.eq('fp_m')].copy()
    rows: list[dict] = []
    cache: dict[tuple[str, str, str, str, str], list[float]] = {}
    fallbacks = 0
    for item in fp.itertuples(index=False):
        key = (item.partition, item.patient_id, item.phase, item.model, item.recording)
        if key not in cache:
            duration = durations.get(item.recording)
            if duration is None:
                raise KeyError(f'Missing manifest duration: {item.recording}')
            alarms = events[(events.partition == item.partition) &
                            (events.patient_id == item.patient_id) &
                            (events.phase == item.phase) &
                            (events.model == item.model) &
                            (events.recording == item.recording)][['start_s', 'end_s']]
            exclusions = [(max(0.0, a - EXCLUSION_S), min(duration, b + EXCLUSION_S))
                          for a, b in seizure_intervals(item.recording)]
            exclusions += [(max(0.0, float(a) - EXCLUSION_S), min(duration, float(b) + EXCLUSION_S))
                           for a, b in alarms.itertuples(index=False)]
            candidates = []
            t = CTX
            while t <= duration:
                if not any(overlaps(t - CTX, t, a, b) for a, b in exclusions):
                    candidates.append(round(t, 4))
                t += GRID_S
            cache[key] = candidates
        candidates = cache[key]
        if not candidates:
            continue
        duration = durations[item.recording]
        tercile = min(2, int(3 * float(item.context_end_s) / duration))
        local = [x for x in candidates if min(2, int(3 * x / duration)) == tercile]
        if not local:
            local = candidates
            fallbacks += 1
        end = stable_choice(local, f'{item.partition}|{item.patient_id}|{item.phase}|{item.model}|{item.recording}|{item.context_end_s:.4f}')
        rows.append({
            'partition': item.partition, 'patient_id': item.patient_id, 'phase': item.phase,
            'model': item.model, 'recording': item.recording, 'context_end_s': end,
            'kind': 'tn_m', 'shared_fp': False, 'source_start_s': end - CTX,
            'source_end_s': end, 'matched_fp_end_s': float(item.context_end_s),
            'time_stratum': tercile,
        })
    tn = pd.DataFrame(rows)
    full = pd.concat([catalog.assign(matched_fp_end_s=pd.NA, time_stratum=pd.NA), tn], ignore_index=True)
    full = full.sort_values(['partition', 'patient_id', 'phase', 'model', 'recording', 'context_end_s', 'kind']).reset_index(drop=True)
    full.to_parquet(OUT / 'context_catalog_with_tn.parquet', index=False)
    summary = full.groupby(['phase', 'kind', 'model']).size().rename('n_contexts').reset_index()
    summary.to_csv(OUT / 'context_summary.csv', index=False)
    manifest = {
        'release_id': OUT.name,
        'status': 'catalog_with_deterministic_matched_tn_only_no_raw_eeg_loaded',
        'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        'holdout_read': False,
        'tn_definition': 'same patient, EDF, phase, and recording-time tercile as each FP_m; 20-s causal window; excludes annotated seizure and selected detector alarm intervals by 30 s',
        'selection': f'deterministic SHA-256 choice from {GRID_S:g}-s candidate grid',
        'tercile_fallbacks': int(fallbacks),
        'contexts': int(len(full)),
        'tn_contexts': int(len(tn)),
        'sha256': hashlib.sha256((OUT / 'context_catalog_with_tn.parquet').read_bytes()).hexdigest(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
