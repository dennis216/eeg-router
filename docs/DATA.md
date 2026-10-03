# External inputs

No patient data or trained model artifacts are included in this code repository.

For frozen posthoc evaluation, provision:

```text
/root/b_false_alarm_atlas/
  manifests/recordings.parquet
  manifests/seizures.parquet
  manifests/groupkfold_cv_v1/...
  runs/v3-groupkfold-confirmatory-v1/<detector>/split<fold>_seed<seed>_main/
    validation/probabilities_step_05000.parquet
    validation/metrics_step_05000.json
    test/probabilities.parquet
  outputs/reports/v3-groupkfold-confirmatory-v1/
    h4_groupkfold_event_features.parquet
    h4_groupkfold_router_validation.csv
    h4_groupkfold_router_test.csv
    h4_groupkfold_router_manifest.json
    h4_router_<variant>_fold<fold>_seed<seed>.joblib
  outputs/reports/h4-protocol-matched-oracle-v2/...
  outputs/reports/h4-final-linkage-expert-audit-v1/...
```

Raw detector training additionally requires CHB-MIT EDFs, generated caches, licensed/downloaded upstream encoders and the applicable split manifests. TUH/TUSZ extensions require separately authorized TUH access. External dependencies must be supplied under their own terms.

Do not fabricate missing probabilities, event features, patient splits or results. Historical `.joblib` files must be trusted and loaded in a compatible scikit-learn environment.

The public repository's Git history identifies the released code snapshot. Detailed source-copy provenance remains in the private backup; it is not an external input needed to run the synthetic example.
