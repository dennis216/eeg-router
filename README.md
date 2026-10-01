# EEG router: BIBM manuscript and follow-up analyses

Research code accompanying **Oracle Decomposition of False-Alarm Control in Routed EEG Seizure Detection** and subsequent router investigations.

Authors: Ziquan Wang and Zixuan Luo (equal contribution).

This repository collects the original analysis, detector training/inference support, temporal-sharing and detector-source analyses, oracle decomposition, and reviewer-requested follow-up code. It also preserves related router/MoE development scripts. See [the experiment index](docs/EXPERIMENTS.md) before running a numbered script: similar numbers do not necessarily refer to the same experiment.

## Current status

The original paper uses `v3-groupkfold-confirmatory-v1`: five patient-held-out folds and three seeds (17, 42, 3407). Its learned-router negative result is preserved.

The reviewer supplement `257_run_bibm_router_review_diagnostics.py` is **unfinished development code**. The latest available completion artifact records `smoke_complete`, one run and zero bootstrap replicates. The complete 15-run FROC, validation-selected comparison and four-condition supplement have **not** been verified as complete. Packaging this code does not change that status. Read [status and known issues](docs/STATUS.md).

## Layout

| Location | Contents |
|---|---|
| `src/bfa/` | Dataset/preprocessing, detector adapters, training and evaluation |
| `scripts/` | Original full-name numbered entry points, including reviewer diagnostics |
| `configs/`, `protocols/` | Training and experiment specifications |
| `tests/` | Unit tests; a subset uses external checkpoints/data |
| `archive/` | Later event matcher, kept separate from the paper protocol |

## Install

Python 3.11 and Linux/WSL are the historical runtime. Install PyTorch appropriate for your hardware, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-analysis.txt
PYTHONPATH=src python -m pytest tests/evaluation -q
```

Additional MoE experiments may require `requirements-extensions.txt` and BIOT source. CBraMod and SingLEM source and pretrained weights must be obtained separately; their repository locations and pinned revisions are in `configs/model/`. No third-party pretrained weights are redistributed here.

## Reproduction

Many original scripts explicitly use `/root/b_false_alarm_atlas`, and some historical fallbacks refer to the author's Windows workspace. These are preserved to avoid silently changing experiment behavior. For the original analyses, prepare an **isolated** Linux checkout at `/root/b_false_alarm_atlas`, install it there, and supply the external artifacts documented in [DATA.md](docs/DATA.md). Do not overwrite an existing experiment directory.

The original sequence is described in [EXPERIMENTS.md](docs/EXPERIMENTS.md). Existing detector probabilities, router joblib files, patient splits, event features and validation locks are required for posthoc reproduction. Source code alone cannot reproduce the published numeric table without those inputs.

The package defaults to the **paper's maximum-total-overlap event matcher**, followed by a minimum one-second overlap filter. The later cardinality-first matcher is preserved in `archive/matcher_after_2026_09_19.py`; it can produce different metrics. Use the dedicated robustness script for that comparison. See [PROTOCOL.md](docs/PROTOCOL.md).

Do not run the unfinished reviewer supplement as a claim of full reproduction. Its completion criteria and outstanding engineering/statistical issues are listed in `docs/STATUS.md`.

## Data and licensing

Raw EEG, per-patient result tables, credentials, private correspondence, manuscript drafts, checkpoints and caches are excluded. Obtain CHB-MIT and TUH datasets under their own access terms. The complete historical backup remains private. No new open-source license is granted by this publication step; third-party source/weights retain their upstream terms. See [THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Code snapshot verification

Python syntax parsing and five evaluation smoke tests passed. Full training and data-dependent reproduction were not rerun for publication. Public source release does not change the completion status of any experiment.
