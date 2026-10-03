# v3-groupkfold-confirmatory-v1

This namespace is the replacement confirmatory layer for the superseded
`v3-standard-confirmatory-v1` queue. The old queue used a custom constrained
greedy assignment and is preserved as a sensitivity artifact; it is not
resumed and is not pooled with this experiment.

## Frozen split rule

- Outer evaluation uses `sklearn.model_selection.GroupKFold(n_splits=5)`.
- The input to the splitter has one row per canonical patient.
- The only grouping/assignment variable is `canonical patient_id`.
- Inner validation uses patient-level `GroupShuffleSplit` with a 20% test size.
- Seizure count, recording duration, signal quality, predictions, and test
  outcomes are forbidden inputs to fold assignment.
- All records, windows, and seizure events for one patient remain together.

## Frozen matrix

- 5 outer folds;
- 3 models: SingLEM, CBraMod, TCN-GAT;
- 3 model seeds: 17, 42, 3407;
- 5,000 optimizer updates per run;
- 45 runs total;
- validation-only threshold selection and one-time outer-test evaluation.

## Required outputs

The primary release must include an out-of-fold probability table and event
table with event sensitivity, FA/24 h, detection delay, patient-level 2,000
replicate bootstrap intervals, patient waterfall summaries, and
sensitivity–FA/24 h curves. H1 and H3 are rerun only after this OOF table is
frozen. H2 must fit discovery features on train patients and use validation
only for specification selection. The frozen H4 preliminary-negative result
is not retuned in this namespace.
