# Completion and engineering status

Inventory date: 2026-09-30. A script's existence does not imply its experiment finished.

## Existing paper evidence

The prior completed original-protocol oracle and linkage directories are `h4-protocol-matched-oracle-v2` and `h4-final-linkage-expert-audit-v1`. Background action analysis is `h4-router-background-actions-v1`. These are distinct from the reviewer supplement.

## Reviewer supplement

The latest accessible `bibm-router-review-diagnostics-v1/manifest.json` states **smoke_complete / runs=1 / bootstrap=0**. No verified complete 15-run result is packaged. Prior long processes stopped before writing the final artifacts; no retained diagnostic evidence proves a specific reason for termination. It was incorrect to infer that a command lacking `--smoke` had never started merely because only smoke output remained.

The current script is preserved with these known issues to resolve before formal use:

- All-run tables are written at the end; per-run checkpoints, persistent logs and explicit failure status are needed for resumability.
- Smoke and full execution share an output directory. Separate destinations are needed to preserve smoke artifacts.
- The bootstrap substitutes zero sensitivity when a resample contains no eligible seizures in a fold. The accepted plan requires redraw and a recorded redraw count.
- The bootstrap point estimate currently averages bootstrap samples. It should report the original 15-run effect, with the bootstrap used for uncertainty.
- Current bootstrap code covers sensitivity only. FA/h and alarm-time intervals and four-condition uncertainty still need implementation.
- Fusion quantiles are currently calculated by row-index alignment; they must use the same patient/recording/time join as fusion inference.
- Threshold locks are retained in memory until the end. Persist validation-only locks before scoring test.
- Patient-level scoring order and background denominators need explicit parity with the original scorer, including tie cases.
- FROC plotting averages rows by numeric threshold across different runs. This does not define a valid common operating-point curve when validation quantile thresholds differ; show individual run curves or explicitly aligned grid indices without test-based optimization.
- A completed smoke for the pre-optimization implementation does not validate every later optimization. The final optimized scorer must pass numerical parity before a full run.

These issues are documented, not repaired by the packaging task. The original paper's results remain the original frozen results. No training or test evaluation was initiated while packaging.

## Source reconciliation

The Linux shared matcher changed after the paper to a cardinality-first objective. The package restores the original Git implementation for the paper runtime and adjusts its matching regression test accordingly. The later matcher and all differing source copies remain in `archive/`. Other Linux source takes precedence over the mirrored Windows version; differences are retained and inventoried, not claimed to be paper-era byte-identical.

`111_freeze_research_a_v1_1.py` references a separate HMM audit (`107_run_h2_dynamic_confirmatory_audit.py`). That unrelated cross-project audit is intentionally absent. It is listed in `package_audit.json` and is not required by the primary router/oracle route.
