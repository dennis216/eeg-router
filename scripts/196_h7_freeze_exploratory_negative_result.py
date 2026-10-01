"""Freeze the completed H7 exploratory result without changing any analysis."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path('/root/b_false_alarm_atlas')
DEV=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-dev-policy-freeze-v1/manifest.json'
CNN=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-raw-cnn-label-support-audit-v1/manifest.json'
HOLD=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-exploratory-tusz-holdout-score-v1'
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-final-exploratory-freeze-v1'

def main():
    if (OUT/'manifest.json').exists():
        raise RuntimeError(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    metrics=pd.read_csv(HOLD/'holdout_metrics.csv'); metrics.to_csv(OUT/'holdout_metrics.csv',index=False)
    pd.read_csv(HOLD/'patient_waterfall.csv').to_csv(OUT/'patient_waterfall.csv',index=False)
    manifest={
        'release_id':OUT.name,
        'status':'frozen_exploratory_negative_result',
        'interpretation':'Early patient-specific recurrent-FP templates reduced false alarms on development data but did not generalize safely to later TUSZ recordings; all policies exceeded the intended sensitivity-loss safety boundary on exploratory holdout.',
        'scope':'H7 patient-specific dual-branch template suppressor; not a seizure detector and not a router',
        'holdout_status':'exploratory_only: TUSZ chronological holdout was previously exposed in earlier H7 work',
        'cnn_status':'not_run_by_predeclared_data-support gate',
        'no_post_holdout_selection':True,
        'source_dev_policy_sha256':hashlib.sha256(DEV.read_bytes()).hexdigest(),
        'source_cnn_audit_sha256':hashlib.sha256(CNN.read_bytes()).hexdigest(),
        'source_holdout_manifest_sha256':hashlib.sha256((HOLD/'manifest.json').read_bytes()).hexdigest(),
        'holdout_metrics':metrics.to_dict('records'),
        'next_step':'Only a genuinely untouched external cohort may be used for confirmatory evaluation; otherwise incorporate this as a negative boundary in False-Alarm Atlas.',
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (OUT/'conclusion.txt').write_text('Frozen conclusion: early patient-specific recurrent false-positive templates did not generalize safely to later TUSZ recordings under the prespecified sensitivity constraint. No CNN was trained and no post-holdout tuning was performed.\n')
    print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=='__main__':main()
