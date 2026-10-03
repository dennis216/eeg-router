# EEG Router

**Code for evaluating event routing and false alarms in seizure detection.**

[中文说明](README.zh-CN.md) · [Experiment index](docs/EXPERIMENTS.md) · [Protocol](docs/PROTOCOL.md) · [Data requirements](docs/DATA.md) · [Status](docs/STATUS.md)

Authors: Ziquan Wang and Zixuan Luo (equal contribution).

## Research question

Can selecting among existing EEG detector events reduce false alarms on held-out patients? This repository contains code accompanying *Oracle Decomposition of False-Alarm Control in Routed EEG Seizure Detection* and related follow-up analyses.

The study evaluates pretrained and task-specific detectors (SingLEM, CBraMod and TCN-GAT), a learned event router, and retrospective oracle decisions using the same candidate groups. The original router's negative result is retained. Temporal-sharing and detector-source analyses provide additional descriptions of detector errors.

## What is included

| Area | Start here |
|---|---|
| Original paper: patient splits, detectors, router and oracle | [Paper experiment entry points](docs/EXPERIMENTS.md) |
| Follow-up diagnostics and alternative protocols | [Experiment index](docs/EXPERIMENTS.md) |
| Reviewer-requested comparisons in development | [Completion and known issues](docs/STATUS.md) |
| Evaluation primitives and tests | [Evaluation code](src/bfa/evaluation/) · [Tests](tests/evaluation/) |
| Historical alternative event matcher | [Archived matcher](archive/matcher_after_2026_09_19.py) |

## Try it without EEG or a GPU

Requires Python 3.11. From a fresh clone, run:

```bash
git clone https://github.com/dennis216/eeg-router.git
cd eeg-router
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python examples/evaluation_demo.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The demo also runs from another working directory when invoked by its absolute path.

The [synthetic example](examples/evaluation_demo.py) converts fixed scores into alarm events, matches them to synthetic seizure intervals, and calculates TP, FP, event sensitivity and false alarms per 24 h. Expected output includes **TP=1, FP=1, sensitivity=0.5 and FA/24h≈24.1341**. The denominator is 3,580 background seconds within a one-hour synthetic recording.

These are teaching values, not study results. The example exercises evaluation primitives; it does not train or reproduce the full router.

### Run lightweight checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/evaluation tests/examples -q
python scripts/check_public_repository.py
```

[GitHub Actions](https://github.com/dennis216/eeg-router/actions) runs these checks and the demo on Ubuntu / Python 3.11. It does not download EEG, train models or run held-out experiments.

## Running study analyses

Follow the [experiment index](docs/EXPERIMENTS.md), [original split specification](protocols/v3-groupkfold-confirmatory-v1.md) and [implemented event protocol](docs/PROTOCOL.md). The original evaluation uses five patient-held-out folds and three seeds (17, 42, 3407).

Full analyses require external inputs and a larger environment:

```bash
python -m pip install -e .
python -m pip install -r requirements-analysis.txt
```

Install hardware-appropriate PyTorch and obtain upstream encoder sources/weights separately. Related MoE experiments may need [additional dependencies](requirements-extensions.txt). See [upstream information](docs/THIRD_PARTY.md).

Historical scripts retain absolute paths, including `/root/b_false_alarm_atlas`. Read their input/output paths before running them. The lightweight example is portable; the complete research pipeline has not been converted into a one-command reproduction package. Never overwrite an existing experiment directory.

## Data and reproducibility

Raw EEG, saved predictions, patient-level tables, trained checkpoints and validation locks are external inputs, documented in [DATA.md](docs/DATA.md). The code alone does not reproduce the published numerical tables without those artifacts.

The default matcher preserves the paper's maximum-total-overlap assignment followed by a one-second overlap filter. A later cardinality-first matcher is archived separately because it can change event metrics.

## Experiment status

The last verified reviewer-supplement artifact, inventoried on **2026-09-30**, was a one-run smoke with zero bootstrap replicates. The complete 15-run supplement is not verified as complete. [STATUS.md](docs/STATUS.md) lists the outstanding checks and implementation issues. Documentation and demo checks do not change experiment completion status.

## Sharing and licensing

This public repository contains code and technical documentation. Manuscript drafts, private correspondence, patient data, credentials and weights are excluded; the complete historical backup remains private.

No open-source license has been selected for the authors' code. Public visibility is not a general reuse license. Upstream software and datasets retain their own terms. Please discuss reuse requiring additional permission with the authors.
