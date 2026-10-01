"""Freeze standards-aligned amendments for H7 dual-branch suppressor.

This addendum supersedes only the selection/scoring details in protocol-freeze
v1. It performs no label generation, fitting, threshold selection or testing.
"""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/b_false_alarm_atlas');OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-standardized-addendum-v1'
def main():
 if (OUT/'addendum.json').exists():raise RuntimeError(f'refusing overwrite: {OUT}')
 OUT.mkdir(parents=True)
 x={
  'release_id':OUT.name,'supersedes':'h7-dual-branch-recurrent-fp-suppressor-v1-protocol-freeze-v1 selection/scoring clauses only','status':'protocol_addendum_frozen_before_raw_extraction',
  'literature_informed_components':{
   'patient_adaptation':'Early false detections learn patient-specific repeated non-epileptiform patterns; later records are treated prospectively.',
   'template_method':'Fit templates require recurrence across complete independent EDF records; a nearest-template/DTW comparator is mandatory before CNN.',
   'error_classes':'FP-like patterns are explicit classes; seizure/TP contexts remain hard safety negatives.',
   'event_evaluation':'Clinical primary outcomes are event-level sensitivity, false alarms per time, and delay; window accuracy is supplementary only.'
  },
  'chronology':{
   'unit':'complete EDF record, sorted by recording start time; no EDF is split across fit/dev/holdout','fit':'earliest 60% of records','dev':'next 20% of records','holdout':'latest 20% of records; never read until policy is fully locked','drift_sensitivity':'predeclared optional rolling-history fit analysis, reported separately, never selected using holdout'
  },
  'population_accounting':{
   'fit_only_eligibility':'A patient-detector unit is eligible if fit has >=20 FP events across >=3 EDF records and >=3 candidate seizure events/annotated seizure records for safety-template construction. This is computed before dev.',
   'dev_calibration_status':'Dev FP and seizure counts determine whether a unit can lock a safety threshold, but do not remove it from the fit-eligible denominator.',
   'reporting':'Report every fit-eligible unit as calibrated, not-calibratable, or failed gate; never present only successful patients.'
  },
  'template_discovery':{
   'specific_branch':'For detector m, form fit-only FP_m templates from recurrent events across >=2 EDF; require >=3 events/component.',
   'shared_branch':'Use 2-of-3 FP onset collar ±10 seconds only to form an additional candidate set; it is not a deployment requirement.',
   'representation_primary':'Patient-local EEG10 standardized on fit only; leave-one-EDF-out within fit selects a single similarity cutoff.',
   'representation_comparator':'Multichannel robust-normalized waveform DTW with a fixed 1-second warping band and 32-Hz matching rate; used as a predeclared template comparator, not tuned on dev.',
   'negative_controls':['TN_m from same patient and EDF/time strata','TP/seizure contexts','other-detector private FP'],
   'template_lock':'No fit prototype, scaling parameter, component, feature weight, or distance cutoff may change after dev is opened.'
  },
  'fair_baselines':[
   'frozen detector m with original eventization','same detector m with dev-calibrated threshold/post-processing under identical sensitivity constraint','2-of-3 and 3-of-3 voting, reported as ensemble references only','detector-specific nearest-template','shared-only nearest-template','dual OR template: specific OR shared','EEG10 logistic risk model','raw CNN only if it beats locked template baselines on dev'
  ],
  'threshold_and_safety':{
   'threshold_selection':'Choose suppression/risk thresholds only on dev, minimizing FA/24h subject to event sensitivity no more than 3 percentage points below the same frozen detector.',
   'seizure_veto':'A high seizure-risk score always retains alarm; any baseline true-positive event lost after suppression is counted and shown.',
   'oracle_gate':'Before learning CNN, calculate a template-reachable dev upper bound. CNN is forbidden unless >=3 fit-eligible patient-detector units have >=20% reachable FP reduction and zero direct seizure-template matches under the locked dev definition.'
  },
  'event_scoring':{
   'primary':'One-to-one event matching with the same frozen eventizer and reference annotations for baseline and suppressor; report sensitivity, FA/24h, precision/F1, and delay.',
   'standardized_secondary':'Run SzCORE-compatible event scoring with documented overlap and pre/post tolerance parameters; report alongside primary, not selectively.',
   'supplementary':'Window AUROC/AUPRC only for risk-head diagnostics; never substitute for event-level clinical outcomes.'
  },
  'statistics':{'paired_unit':'patient-detector paired changes vs frozen detector','uncertainty':'2,000 patient bootstrap CIs including all EDFs per sampled patient','tables':'raw event counts, evaluated hours, eligible/calibrated/non-calibratable units, and per-patient waterfall','multiplicity':'BH-FDR only for families of exploratory detector/feature comparisons; primary policy comparison is prespecified.'},
  'scope':'TUSZ work remains exploratory because previous related H6/H7 development has accessed its chronology. A locked policy requires external confirmation on Siena or another untouched cohort with a pre-frozen channel map and no threshold adaptation.',
  'created_utc':datetime.now(timezone.utc).isoformat()
 }
 raw=json.dumps(x,indent=2,sort_keys=True)+'\n';(OUT/'addendum.json').write_text(raw);(OUT/'addendum.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n');print(raw)
if __name__=='__main__':main()
