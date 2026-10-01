#!/usr/bin/env bash
set -euo pipefail

cd /root/b_false_alarm_atlas
ROOT="$PWD"
PYTHON="$ROOT/.venv/bin/python"
RUN_ID="v3-lite-2split"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

for split_seed in 42; do
    for model_seed in 3407; do
        for model in singlem cbramod tcn_gat; do
            run_dir="$ROOT/runs/$RUN_ID/$model/split${split_seed}_seed${model_seed}_main"
            log_path="$LOG_DIR/v3-lite-2split-${model}-split${split_seed}-seed${model_seed}.log"
            if [[ -f "$run_dir/completed.json" ]]; then
                echo "V3_LITE_ALREADY_COMPLETE model=$model split=$split_seed seed=$model_seed" | tee -a "$LOG_DIR/v3-lite-2split-queue.log"
                continue
            fi
            resume_args=()
            if [[ -d "$run_dir/checkpoints" ]]; then
                latest="$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)"
                if [[ -n "$latest" ]]; then
                    resume_args=(--resume "$latest")
                fi
            fi
            echo "V3_LITE_START model=$model split=$split_seed seed=$model_seed" | tee -a "$LOG_DIR/v3-lite-2split-queue.log"
            "$PYTHON" scripts/11_run_bsn_pilot.py \
                --model "$model" \
                --split-seed "$split_seed" \
                --model-seed "$model_seed" \
                --run-id "$RUN_ID" \
                "${resume_args[@]}" >"$log_path" 2>&1
            test -f "$run_dir/completed.json"
            echo "V3_LITE_COMPLETE model=$model split=$split_seed seed=$model_seed" | tee -a "$LOG_DIR/v3-lite-2split-queue.log"
        done
    done
done

echo "V3_LITE_QUEUE_COMPLETE" | tee -a "$LOG_DIR/v3-lite-2split-queue.log"
