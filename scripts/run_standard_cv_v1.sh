#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/b_false_alarm_atlas
PYTHON=/root/b_false_alarm_atlas/.venv/bin/python
RUNNER=/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts/11_run_bsn_pilot.py
CONFIG=/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/configs/standard/v3_standard_confirmatory_v1.yaml
RUN_ID=v3-standard-confirmatory-v1

export BFA_ROOT="$ROOT"
export BFA_CONFIG="$CONFIG"
export PYTHONPATH="$ROOT/src"

for fold in 0 1 2 3 4; do
  split_file="$ROOT/manifests/standard_cv_v1/fold_${fold}.json"
  for model in singlem cbramod tcn_gat; do
    for model_seed in 17 42 3407; do
      run_dir="$ROOT/runs/$RUN_ID/$model/split${fold}_seed${model_seed}_main"
      if [[ -f "$run_dir/completed.json" ]]; then
        echo "STANDARD_CV_SKIP completed model=$model fold=$fold seed=$model_seed" >&2
        continue
      fi

      args=(
        "$PYTHON" "$RUNNER"
        --model "$model"
        --model-seed "$model_seed"
        --split-seed "$fold"
        --split-file "$split_file"
        --run-id "$RUN_ID"
      )
      if [[ -d "$run_dir/checkpoints" ]]; then
        latest_checkpoint=$(find "$run_dir/checkpoints" -maxdepth 1 -name 'step_*.pt' -printf '%f\n' | sort | tail -n 1 || true)
        if [[ -n "$latest_checkpoint" ]]; then
          args+=(--resume "$run_dir/checkpoints/$latest_checkpoint")
        fi
      fi
      echo "STANDARD_CV_START model=$model fold=$fold seed=$model_seed" >&2
      "${args[@]}"
      echo "STANDARD_CV_DONE model=$model fold=$fold seed=$model_seed" >&2
    done
  done
done

echo "STANDARD_CV_ALL_RUNS_COMPLETE run_id=$RUN_ID" >&2
