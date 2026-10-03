# Experiment index

[Home](../README.md) · [Status](STATUS.md) · [External inputs](DATA.md)

Use full filenames: numerical prefixes collide across historical projects. These links locate code, not certifications that every experiment completed. No pipeline is launched on installation.

## 1. Original paper route

Read the [historical split specification](../protocols/v3-groupkfold-confirmatory-v1.md), [frozen configuration](../configs/standard/v3_groupkfold_confirmatory_v1.yaml) and [implemented event protocol](PROTOCOL.md) first. The restored split specification matches the private backup text (line endings may differ); its required-output list is a specification, not a completion manifest. Its H2 references describe the broader historical project, outside this repository's standalone router scope.

| Step | Entry points |
|---|---|
| Patient splits and audit | [71_build_groupkfold_confirmatory_v1.py](../scripts/71_build_groupkfold_confirmatory_v1.py), [72_audit_groupkfold_confirmatory_v1.py](../scripts/72_audit_groupkfold_confirmatory_v1.py) |
| OOF assembly and freeze | [73_build_groupkfold_confirmatory_oof.py](../scripts/73_build_groupkfold_confirmatory_oof.py), [74_freeze_groupkfold_preflight.py](../scripts/74_freeze_groupkfold_preflight.py) |
| Temporal sharing | [75_run_groupkfold_h1.py](../scripts/75_run_groupkfold_h1.py) |
| Discovery inference | [76_infer_groupkfold_train_discovery.py](../scripts/76_infer_groupkfold_train_discovery.py) |
| Detector-source features and classification | [77_build_groupkfold_h2h3_events.py](../scripts/77_build_groupkfold_h2h3_events.py), [78_extract_groupkfold_h2h3_features.py](../scripts/78_extract_groupkfold_h2h3_features.py), [79_run_groupkfold_h2_h3.py](../scripts/79_run_groupkfold_h2_h3.py) |
| Router discovery subset | [80_build_groupkfold_discovery_subset.py](../scripts/80_build_groupkfold_discovery_subset.py), [81_infer_groupkfold_discovery_subset.py](../scripts/81_infer_groupkfold_discovery_subset.py) |
| Event router | [82_run_groupkfold_h4_gate.py](../scripts/82_run_groupkfold_h4_gate.py), [83_extract_groupkfold_h4_features.py](../scripts/83_extract_groupkfold_h4_features.py), [84_run_groupkfold_h4_router.py](../scripts/84_run_groupkfold_h4_router.py) |
| Same-protocol oracle | [240_run_h4_protocol_matched_oracle.py](../scripts/240_run_h4_protocol_matched_oracle.py) |
| Linkage and oracle decomposition | [241_run_h1_h4_linkage_and_gap_audit.py](../scripts/241_run_h1_h4_linkage_and_gap_audit.py), [242_run_expert_adequacy_audit.py](../scripts/242_run_expert_adequacy_audit.py), [243_finalize_h4_pre_manuscript_audit.py](../scripts/243_finalize_h4_pre_manuscript_audit.py) |
| Background-group router actions | [248_diagnose_h4_router_group_actions.py](../scripts/248_diagnose_h4_router_group_actions.py) |

Start with [110_preflight_no_duplicate_overlap.py](../scripts/110_preflight_no_duplicate_overlap.py) and verify frozen inputs before launching data-dependent work. Existing detector OOF predictions and router artifacts may be reused where the script expects them. Detector adapters and training support are in [src/bfa](../src/bfa/) and [model configurations](../configs/model/). This table is navigation, not a shell execution recipe: inspect every script's flags and hardcoded paths.

## 2. Follow-up analyses and distinct protocols

| Analysis | Entry points / interpretation |
|---|---|
| Matching-tolerance robustness | [128_run_h1_temporal_tolerance_sensitivity.py](../scripts/128_run_h1_temporal_tolerance_sensitivity.py) |
| Local-context null | [129_run_h1_local_context_null.py](../scripts/129_run_h1_local_context_null.py) |
| Feature ablation | [130_run_h3_feature_ablation.py](../scripts/130_run_h3_feature_ablation.py) |
| Alternative event matcher | [244_h4_cardinality_first_matcher_robustness.py](../scripts/244_h4_cardinality_first_matcher_robustness.py); separate scoring control |
| Main figure | [245_regenerate_h4_main_figure.py](../scripts/245_regenerate_h4_main_figure.py) |
| Earlier router revision | [251_run_h4_router_revision_diagnostics.py](../scripts/251_run_h4_router_revision_diagnostics.py); changed interval/scoring-start rules, not paper-equivalent |
| Checkpoints and alarm time | [252_audit_h4_existing_checkpoints_alarm_time.py](../scripts/252_audit_h4_existing_checkpoints_alarm_time.py), [253_optimize_h4_detector_alarm_time.py](../scripts/253_optimize_h4_detector_alarm_time.py), [254_infer_h4_selected_checkpoint_test.py](../scripts/254_infer_h4_selected_checkpoint_test.py) |
| Cache checks | [255_audit_h4_cache_reconstruction_parity.py](../scripts/255_audit_h4_cache_reconstruction_parity.py), [256_build_h4_revision_tcn_quality_cache.py](../scripts/256_build_h4_revision_tcn_quality_cache.py) |

## 3. Reviewer-requested supplement — development

[257_run_bibm_router_review_diagnostics.py](../scripts/257_run_bibm_router_review_diagnostics.py) contains the FROC, validation-selected operating-point and LL/LO/OL/OO diagnostics. The latest verified artifact is a one-run smoke, not a complete 15-run analysis. Read [outstanding issues](STATUS.md) before using it. This release does not repair or execute that experimental implementation.

## 4. Historical and related code

The [scripts directory](../scripts/) retains compact/legacy implementations and related H6/H7, MoE, BIOT and same-window developments at their original paths. They use separate protocols and should not replace the original paper route merely because a filename is newer. Filename numbers alone do not establish execution order or completion.

Standalone HMM, artifact-ontology and TTT projects are excluded. Shared H2/H3 utilities remain dependencies; their presence does not add those standalone experiments to the router paper. The broad cross-project freeze utility has an intentionally absent HMM dependency, documented in [STATUS.md](STATUS.md).
