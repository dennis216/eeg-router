#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="$ROOT/.venv/bin/python"
NAMESPACE="v3-lite"
MODEL_SEED=17
GPU_WAIT_GIB=8

wait_for_gpu() {
    while true; do
        used_gib="$($ROOT/.venv/bin/python - <<'PY'
import subprocess
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()[0]
print(float(raw) / 1024.0)
PY
)"
        if "$PYTHON" -c "raise SystemExit(0 if float('$used_gib') < $GPU_WAIT_GIB else 1)"; then
            return
        fi
        echo "V3_LITE_WAIT_GPU used_gib=$used_gib threshold_gib=$GPU_WAIT_GIB" | tee -a "$ROOT/logs/v3_lite_queue.log"
        sleep 60
    done
}

for split_seed in 42 3407 2026 777; do
    for model in tcn_gat cbramod singlem; do
        run_id="lite-unified-${model}-split${split_seed}-seed${MODEL_SEED}"
        run_dir="$ROOT/runs/$NAMESPACE/unified/$run_id"
        log_path="$ROOT/logs/v3_lite-${model}-split${split_seed}-seed${MODEL_SEED}.log"
        if [[ -f "$run_dir/completed.json" ]]; then
            echo "V3_LITE_ALREADY_COMPLETE run_id=$run_id" | tee -a "$ROOT/logs/v3_lite_queue.log"
            continue
        fi
        wait_for_gpu
        resume_args=()
        if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
            resume_args+=(--resume)
        fi
        echo "V3_LITE_START run_id=$run_id" | tee -a "$ROOT/logs/v3_lite_queue.log"
        "$PYTHON" scripts/28_run_v3_unified.py \
            --model "$model" \
            --split-seed "$split_seed" \
            --model-seed "$MODEL_SEED" \
            --run-id "$run_id" \
            --output-namespace "$NAMESPACE" \
            "${resume_args[@]}" >"$log_path" 2>&1
        echo "V3_LITE_COMPLETE run_id=$run_id" | tee -a "$ROOT/logs/v3_lite_queue.log"
    done
done

