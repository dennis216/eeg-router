# Experiment index

Use full filenames; numerical prefixes are historical identifiers and collide.

| Research stage | Entry points / prefixes | Role |
|---|---|---|
| Preprocessing and detector baselines | `01`–`30`, `61`–`67`, `src/bfa`, model configs | Dataset preparation, frozen pretrained encoders, task-specific detector, inference |
| Original patient splits and OOF | `71`–`74`, `run_groupkfold_confirmatory_v1.sh` | Standard patient-level GroupKFold |
| Temporal sharing | `75`, `128`, workspace `129_run_h1_local_context_null.py`, `144`/`145` external H1 | Time-shift nulls and sensitivity analyses |
| Detector-source classification | `77`–`79`, `130_run_h3_feature_ablation.py`, H3 scripts `132`, `146`–`165`, raw-source `245` | Features, classification and controls; shared H2/H3 utilities are dependencies |
| Paper event router | `80`–`84` | Discovery data, gating, candidate features, model-aware/no-ID router |
| External evaluation | `46`–`60`, `85`–`92`, `94` | TUH/TUSZ preparation, frozen inference and descriptive evaluation |
| Protocol-matched oracle | `240_run_h4_protocol_matched_oracle.py` | Same candidate groups and abstention actions |
| Oracle and error diagnostics | `241_run_h1_h4_linkage_and_gap_audit.py`, `242_run_expert_adequacy_audit.py`, `243_finalize_h4_pre_manuscript_audit.py` | Linkage, calibration, assignment/abstention decomposition |
| Scorer robustness and figures | `244_h4_cardinality_first_matcher_robustness.py`, `245_regenerate_h4_main_figure.py` | Explicit alternative scorer and main figure |
| Background router actions | `248_diagnose_h4_router_group_actions.py` | Retained/abstained background and seizure-associated groups |
| Earlier revision diagnostics | `251_run_h4_router_revision_diagnostics.py`, `252`–`256` H4 files | Changed interval/scoring-start and checkpoint/alarm-duration analyses; not paper-equivalent |
| Reviewer-requested supplement | `257_run_bibm_router_review_diagnostics.py` | FROC, validation-selected budgets and LL/LO/OL/OO; incomplete |
| Related developments | Filenames containing `h6`, `h7`, `moe`, `biot`, `same_window`, `phenotype` | Router/suppressor explorations; separate protocols and outcomes |
| Legacy implementation | `31`–`45` compact scripts | Archived experimental lineage; not the main GroupKFold protocol |

## Paper reproduction order (requires external artifacts)

1. Inspect `protocols/v3-groupkfold-confirmatory-v1.md` and `configs/standard/v3_groupkfold_confirmatory_v1.yaml`.
2. Audit frozen input hashes, splits and validation locks. `110_preflight_no_duplicate_overlap.py` is included.
3. Prepare detector OOF artifacts with `71`–`74` and detector training/inference dependencies as required.
4. Use `80`–`84` for original router construction, or reuse the original saved router artifacts for posthoc work.
5. Run `240_run_h4_protocol_matched_oracle.py`, then the full-name `241`, `242`, `243` diagnostics.
6. Use `248_diagnose_h4_router_group_actions.py` for the background action analysis.
7. Run `244_h4_cardinality_first_matcher_robustness.py` only as the explicitly separate scorer control.

Run scripts only after reading their input/output paths. Several training scripts can create checkpoints and modify experiment outputs. This repository does not run them on install.

The full provenance table is machine readable in `source_manifest.json`. Source variants are archived instead of being discarded based on timestamps. HMM, artifact-ontology and TTT standalone projects are outside this router package; joint H2/H3 and broad release utilities remain where they are shared dependencies.
