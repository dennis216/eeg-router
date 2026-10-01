"""Freeze the protocol for H7 dual-branch patient-specific FP suppression.

No model fitting or test/holdout access occurs here.  The output is a
machine-readable contract used by subsequent label, oracle and training code.
"""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/root/b_false_alarm_atlas')
OUT=ROOT/'outputs/reports/h7-dual-branch-recurrent-fp-suppressor-v1-protocol-freeze-v1'
def main():
 if (OUT/'protocol.json').exists():raise RuntimeError(f'refusing overwrite: {OUT}')
 OUT.mkdir(parents=True)
 p={
  'release_id':OUT.name,
  'status':'protocol_frozen_before_label_audit',
  'scope':'exploratory TUSZ development; any final confirmation requires an untouched external cohort (prefer Siena)',
  'experts':{'names':['singlem','cbramod','tcn_gat'],'frozen':True,'training_forbidden':True,'checkpoint_source':'h6-tusz-raw-pre-router-v1 frozen CHB checkpoints','threshold_source':'frozen CHB validation-only thresholds'},
  'patient_time_split':{'fit':'earliest 60% of each patient chronology; phenotype/template fit only','dev':'next 20%; threshold selection and gates only','holdout':'latest 20%;not accessed in this protocol-development stage'},
  'input':{'raw_eeg_only':True,'canonical_channels':16,'sampling_hz':128,'context_seconds':20,'expert_probability_or_model_id_in_input':False},
  'labels':{
   'specific_fp_m':'non-seizure event where frozen detector m alarms; candidate becomes recurrent only if its early EEG pattern has >=3 events across >=2 EDF',
   'shared_fp':'non-seizure candidate where >=2 frozen detector FP onsets fall within a fixed ±10-s collar; must separately satisfy recurrence criterion',
   'tn_m':'non-seizure context with no alarm from detector m; matched by patient/EDF/time of recording',
   'tp_or_seizure':'seizure contexts and detector true-positive contexts; hard safety negatives for any suppressor',
   'other_detector_private_fp':'hard negative for detector-specific branch m'
  },
  'branches':{
   'specific':{'heads':['r_singlem','r_cbramod','r_tcn_gat'],'meaning':'similarity to detector-specific recurrent early FP templates'},
   'shared':{'head':'r_shared','meaning':'similarity to recurrent early shared-FP templates'},
   'safety':{'head':'r_seizure','meaning':'seizure risk veto; high score forbids suppression'}
  },
  'inference':{'per_detector':'For an existing alarm from detector m, suppress only if (r_m >= tau_m OR r_shared >= tau_shared) AND r_seizure < tau_safe. Otherwise retain m.','event_policy':'Primary evaluation is each frozen detector plus its suppressor; no union/average of three detector alarms in the primary result.'},
  'gates':{
   'data_eligibility':'For a detector m: >=20 fit FP candidates across >=3 EDF, >=5 dev FP, >=2 dev seizure events.',
   'recurrent_template':'Cross-EDF phenotype component: >=3 fit FP events across >=2 EDF; similarity threshold fit-only.',
   'oracle':'Before training, template-reachable dev FP reduction >=20% with zero constructed seizure suppression.',
   'training':'At least 3 patient-detector units pass the data, recurrence and oracle gates.',
   'dev_safety':'Choose all thresholds only on dev, with sensitivity loss <=3 percentage points versus same frozen detector.',
   'test':'No holdout tuning; test evaluation count exactly 1 per frozen policy.'
  },
  'baselines':['frozen detector m','threshold-only adjustment of m','patient early FP nearest-template','EEG10 logistic FP-risk head','raw-EEG CNN FP-risk head only if lower-capacity baseline passes'],
  'outcomes':['event sensitivity','FA/24h','detection delay','number of suppressed seizure events','patient-detector waterfall'],
  'prohibited':['expert retraining','using holdout to define labels/templates/thresholds','using detector probabilities or IDs as input','claiming TUSZ development results as independent confirmation'],
  'created_utc':datetime.now(timezone.utc).isoformat()
 }
 raw=json.dumps(p,indent=2,sort_keys=True)+'\n';(OUT/'protocol.json').write_text(raw);(OUT/'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n');print(raw)
if __name__=='__main__':main()
