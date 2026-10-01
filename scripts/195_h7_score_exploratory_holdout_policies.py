"""Apply pre-frozen H7 template policies to TUSZ exploratory holdout events."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path('/root/b_false_alarm_atlas')
DATA = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-eeg10-v1/holdout_eeg10.parquet'
FIT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-eeg10-dual-template-fit-v1'
ELIG = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-fit-only-eligibility-v2/patient_detector_fit_only_eligibility.csv'
MAN = ROOT / 'outputs/reports/h6-tusz-raw-pre-router-v1/manifests'
POL = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1/manifest.json'
OUT = ROOT / 'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1'
FEATURES = ('rms_median_uv','rms_iqr_uv','spatial_rms_dispersion','spatial_rms_max_min_log',
            'relative_delta_power','relative_theta_power','relative_alpha_power',
            'relative_beta_power','relative_gamma_power','high_low_ratio')


def short_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ratio_score(frame: pd.DataFrame, obj: dict) -> np.ndarray:
    if len(frame) == 0:
        return np.empty(0, dtype=float)
    centers = np.asarray(obj['centers'], dtype=float)
    if not len(centers) or not np.isfinite(obj['cutoff']):
        return np.full(len(frame), np.inf)
    distances = cdist(obj['scaler'].transform(frame.loc[:, FEATURES]), centers).min(axis=1)
    return distances / float(obj['cutoff'])


def holdout_truth_durations() -> pd.DataFrame:
    rows = []
    for partition in ('train', 'dev', 'eval'):
        by_patient = {}
        manifest = json.loads((MAN / f'{partition}_manifest.json').read_text())
        for record in manifest['records']:
            by_patient.setdefault(str(record['patient_id']), []).append(record)
        for patient_id, records in by_patient.items():
            records = sorted(records, key=lambda r: str(r['relative_edf']))
            total = sum(float(r['duration_s']) for r in records)
            offset = 0.0
            truth = 0
            for record in records:
                for event in record['seizure_events']:
                    midpoint = offset + (float(event['start_s']) + float(event['end_s'])) / 2
                    if midpoint >= 0.8 * total:
                        truth += 1
                offset += float(record['duration_s'])
            rows.append({'partition': partition, 'patient_id': patient_id,
                         'holdout_duration_s': 0.2 * total,
                         'holdout_truth_events': truth})
    return pd.DataFrame(rows)


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.read_parquet(DATA)
    eligible = pd.read_csv(ELIG)
    eligible = eligible[eligible.fit_eligible_standard]

    specific = []
    for unit in eligible.itertuples(index=False):
        key = f'{unit.partition}|{unit.patient_id}|{unit.model}'
        obj = joblib.load(FIT / 'objects' / f'specific_{short_hash(key)}.joblib')
        q = x[(x.partition == unit.partition) & (x.patient_id == unit.patient_id) &
              (x.model == unit.model)].copy()
        q['specific_ratio'] = ratio_score(q, obj)
        specific.append(q)
    specific = pd.concat(specific, ignore_index=True)

    shared = []
    for (partition, patient_id), q in x.groupby(['partition', 'patient_id'], sort=True):
        if not ((eligible.partition == partition) & (eligible.patient_id == patient_id)).any():
            continue
        obj = joblib.load(FIT / 'objects' / f'shared_{short_hash(f"{partition}|{patient_id}")}.joblib')
        z = q[['partition','patient_id','phase','model','recording','context_end_s','kind','context_id']].copy()
        z['shared_ratio'] = ratio_score(q, obj)
        shared.append(z)
    shared = pd.concat(shared, ignore_index=True)

    keys = ['partition','patient_id','phase','model','recording','context_end_s','kind','context_id']
    x = specific.merge(shared[keys + ['shared_ratio']], on=keys, how='left')
    policies = {'specific': 1.25, 'shared': 1.5, 'dual_or': 1.0}
    for method, alpha in policies.items():
        if method == 'specific':
            mask = x.specific_ratio <= alpha
        elif method == 'shared':
            mask = x.shared_ratio <= alpha
        else:
            mask = (x.specific_ratio <= alpha) | (x.shared_ratio <= alpha)
        x[f'{method}_suppressed'] = mask.fillna(False)
    x.to_parquet(OUT / 'holdout_event_scores.parquet', index=False)

    durations = holdout_truth_durations()
    events = x[x.kind.isin(['fp_m', 'tp_m'])]
    rows = []
    for (partition, patient_id, model), group in events.groupby(['partition','patient_id','model'], sort=True):
        d = durations[(durations.partition == partition) & (durations.patient_id == patient_id)].iloc[0]
        row = {'partition': partition, 'patient_id': patient_id, 'model': model,
               'holdout_duration_s': float(d.holdout_duration_s),
               'truth_events': int(d.holdout_truth_events),
               'baseline_fp': int((group.kind == 'fp_m').sum()),
               'baseline_tp': int((group.kind == 'tp_m').sum())}
        for method in policies:
            row[f'{method}_fp'] = int(((group.kind == 'fp_m') & ~group[f'{method}_suppressed']).sum())
            row[f'{method}_tp'] = int(((group.kind == 'tp_m') & ~group[f'{method}_suppressed']).sum())
            row[f'{method}_suppressed_fp'] = int(((group.kind == 'fp_m') & group[f'{method}_suppressed']).sum())
            row[f'{method}_suppressed_tp'] = int(((group.kind == 'tp_m') & group[f'{method}_suppressed']).sum())
        rows.append(row)
    patient = pd.DataFrame(rows)
    for method in ['baseline', *policies]:
        patient[f'{method}_fa24'] = patient[f'{method}_fp'] / patient.holdout_duration_s * 86400
        patient[f'{method}_sensitivity'] = patient[f'{method}_tp'] / patient.truth_events.replace(0, np.nan)
    patient.to_csv(OUT / 'patient_waterfall.csv', index=False)

    summary_rows = []
    with_truth = patient[patient.truth_events > 0]
    for method in ['baseline', *policies]:
        summary_rows.append({
            'method': method, 'patients': len(patient),
            'patients_with_truth': len(with_truth),
            'pooled_fp': int(with_truth[f'{method}_fp'].sum()),
            'pooled_tp': int(with_truth[f'{method}_tp'].sum()),
            'pooled_truth': int(with_truth.truth_events.sum()),
            'pooled_sensitivity': float(with_truth[f'{method}_tp'].sum() / with_truth.truth_events.sum()),
            'macro_sensitivity': float(with_truth[f'{method}_sensitivity'].mean()),
            'macro_FA_24h': float(with_truth[f'{method}_fa24'].mean()),
            'pooled_FA_24h': float(with_truth[f'{method}_fp'].sum() / with_truth.holdout_duration_s.sum() * 86400),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'holdout_metrics.csv', index=False)

    rng = np.random.default_rng(20260831)
    bootstrap = []
    for method in policies:
        for replicate in range(2000):
            sample = with_truth.iloc[rng.integers(0, len(with_truth), len(with_truth))]
            bootstrap.append({'method': method, 'replicate': replicate,
                'delta_sensitivity': float(sample[f'{method}_sensitivity'].mean() - sample.baseline_sensitivity.mean()),
                'delta_FA24': float(sample[f'{method}_fa24'].mean() - sample.baseline_fa24.mean())})
    pd.DataFrame(bootstrap).to_csv(OUT / 'patient_bootstrap_delta.csv', index=False)
    manifest = {'release_id': OUT.name, 'status': 'exploratory_holdout_scored_once',
                'exploratory': True, 'holdout_accessed': True,
                'no_selection_after_holdout': True, 'policy_sha256': hashlib.sha256(POL.read_bytes()).hexdigest(),
                'source_feature_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
                'policies': policies, 'event_rows': len(x), 'patient_model_rows': len(patient),
                'bootstrap_replicates': 2000,
                'sha256': hashlib.sha256((OUT / 'holdout_metrics.csv').read_bytes()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
