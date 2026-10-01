"""Correct H7 selection gate: use patient fit + development only.

The preceding H7 feasibility artifact is retained as a diagnostic because it
read chronological holdout while computing an oracle.  This script does not
use that holdout for any gate, threshold, eligibility decision, or training.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path('/root/b_false_alarm_atlas')
SOURCE = ROOT / 'outputs/reports/h7-personalized-consensus-fp-suppressor-v1'
OUT = ROOT / 'outputs/reports/h7-personalized-consensus-fp-suppressor-v1-dev-gate-v2'
SOURCE_NS = 'h6-tusz-raw-pre-router-v1'
MODELS = ('singlem', 'cbramod', 'tcn_gat')
MIN_EVENTS, MIN_RECORDS, MIN_DEV_TRUTHS, MIN_DEV_FP, MIN_DEV_ORACLE = 20, 3, 2, 5, .20


def chronology_truths() -> pd.DataFrame:
    rows = []
    for part in ('train', 'dev', 'eval'):
        path = ROOT / 'outputs/reports' / SOURCE_NS / 'manifests' / f'{part}_manifest.json'
        if not path.exists():
            continue
        by_patient = {}
        for record in json.loads(path.read_text())['records']:
            by_patient.setdefault(str(record['patient_id']), []).append(record)
        for patient, records in by_patient.items():
            ordered = sorted(records, key=lambda x: str(x['relative_edf']))
            total = sum(float(x['duration_s']) for x in ordered); offset = 0.
            for record in ordered:
                for event in record['seizure_events']:
                    midpoint = offset + (float(event['start_s']) + float(event['end_s'])) / 2
                    phase = 'fit' if midpoint < .60 * total else 'dev' if midpoint < .80 * total else 'holdout'
                    rows.append({'partition':part, 'patient_id':patient, 'phase':phase, 'truth_event':1})
                offset += float(record['duration_s'])
    return pd.DataFrame(rows)


def main() -> None:
    if (OUT / 'manifest.json').exists():
        raise RuntimeError(f'refusing overwrite: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    events = pd.read_parquet(SOURCE / 'frozen_expert_events.parquet')
    shared = pd.read_parquet(SOURCE / 'shared_fp_events.parquet')
    nulls = pd.read_csv(SOURCE / 'fit_pairwise_circular_shift.csv')
    truths = chronology_truths()
    truth_counts = truths.groupby(['partition','patient_id','phase']).size().rename('truth_events').reset_index()
    rows = []
    for (part, patient), patient_events in events.groupby(['partition','patient_id'], sort=False):
        patient_shared = shared[(shared.partition == part) & (shared.patient_id == patient)]
        fit = patient_shared[patient_shared.phase == 'fit']
        dev_events = patient_events[patient_events.phase == 'dev']
        dev_shared = patient_shared[patient_shared.phase == 'dev']
        n_truth = int(truth_counts[(truth_counts.partition == part) & (truth_counts.patient_id == patient) & (truth_counts.phase == 'dev')].truth_events.sum())
        scores = []
        for model in MODELS:
            d = dev_events[dev_events.model == model]
            scores.append({'model':model, 'matched':int(d.matched_seizure.sum()), 'fp':int(d.is_fp.sum())})
        max_matched = max(x['matched'] for x in scores)
        # Only compare FA after retaining models within 3pp of the best
        # development sensitivity.  A patient without dev seizures fails the
        # gate below rather than being assigned an arbitrary safe model.
        allowed = [x for x in scores if n_truth and x['matched'] / n_truth >= max_matched / n_truth - .03]
        best = min(allowed, key=lambda x:(x['fp'], -x['matched'], x['model'])) if allowed else {'model':'none','matched':0,'fp':0}
        fp_events = dev_events[(dev_events.model == best['model']) & (dev_events.is_fp == 1)] if best['model'] != 'none' else dev_events.iloc[0:0]
        suppressed = 0
        if len(dev_shared):
            for event in fp_events.itertuples(index=False):
                hit = ((dev_shared.start_s < event.end_s) & (dev_shared.end_s > event.start_s) & dev_shared.model_mask.str.contains(best['model'], regex=False)).any()
                suppressed += int(bool(hit))
        significant = nulls[(nulls.partition == part) & (nulls.patient_id == patient) & (nulls.significant == True)]
        rows.append({
            'partition':part, 'patient_id':patient,
            'fit_shared_events':int(len(fit)), 'fit_shared_records':int(fit.recording.nunique()),
            'fit_significant_pairwise_sharing':int(len(significant) > 0),
            'dev_truth_events':n_truth, 'best_fixed_detector_dev':best['model'],
            'dev_sensitivity_best_fixed':float(best['matched'] / n_truth) if n_truth else None,
            'dev_fp_best_fixed':int(best['fp']), 'dev_shared_fp_oracle_removed':int(suppressed),
            'dev_oracle_fa_reduction':float(suppressed / best['fp']) if best['fp'] else 0.,
            'oracle_sensitivity_loss':0.,
        })
    audit = pd.DataFrame(rows)
    audit['gate_data_support'] = (audit.fit_shared_events >= MIN_EVENTS) & (audit.fit_shared_records >= MIN_RECORDS)
    audit['gate_statistical_sharing'] = audit.fit_significant_pairwise_sharing == 1
    audit['gate_safety_support'] = audit.dev_truth_events >= MIN_DEV_TRUTHS
    audit['gate_oracle'] = (audit.dev_fp_best_fixed >= MIN_DEV_FP) & (audit.dev_oracle_fa_reduction >= MIN_DEV_ORACLE)
    audit['gate_train'] = audit.gate_data_support & audit.gate_statistical_sharing & audit.gate_safety_support & audit.gate_oracle
    audit.to_csv(OUT / 'patient_dev_only_gate.csv', index=False)
    report = {
        'release_id':'h7-personalized-consensus-fp-suppressor-v1-dev-gate-v2',
        'status':'dev_only_training_gate_complete_no_model_training',
        'source_h7_feasibility':str(SOURCE.relative_to(ROOT)),
        'source_h7_holdout_use':'not used by this gate; prior source artifact retained only as diagnostic',
        'gate_contract':{'fit_shared_events_ge':MIN_EVENTS,'fit_shared_records_ge':MIN_RECORDS,'fit_significant_pair':True,'dev_truth_events_ge':MIN_DEV_TRUTHS,'dev_best_fixed_fp_ge':MIN_DEV_FP,'dev_oracle_fa_reduction_ge':MIN_DEV_ORACLE,'dev_oracle_sensitivity_loss':0.},
        'counts':{'patients':int(len(audit)),'eligible_for_training':int(audit.gate_train.sum())},
        'created_utc':datetime.now(timezone.utc).isoformat(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == '__main__': main()
