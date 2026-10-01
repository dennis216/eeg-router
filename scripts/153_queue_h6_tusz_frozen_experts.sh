#!/usr/bin/env bash
set -euo pipefail

# Frozen CHB experts are evaluated on the official TUSZ partitions.  This is
# inference only; no checkpoint or threshold is changed.  One representative
# standard run (fold0/seed17) is frozen per detector for the router-label
# construction, so the router experiment has three experts rather than 45
# duplicate expert variants.

ROOT=/root/b_false_alarm_atlas
cd "$ROOT"
OUT_ROOT="$ROOT/outputs/tuh_external/h6_tusz_raw_pre_router_v1"
LOG_ROOT="$ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
MODELS=(singlem cbramod tcn_gat)
PARTITIONS=(train dev eval)

for partition in "${PARTITIONS[@]}"; do
  manifest="$ROOT/outputs/reports/h6-tusz-raw-pre-router-v1/manifests/${partition}_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "H6_EXPERT_QUEUE_STOP missing_manifest=$manifest" >&2
    exit 2
  fi
  for model in "${MODELS[@]}"; do
    partition_root="$OUT_ROOT/$partition"
    run_root="$partition_root/$model/fold0_seed17"
    if [[ -f "$run_root/progress.json" ]] && grep -q '"status": "complete"' "$run_root/progress.json"; then
      echo "H6_EXPERT_QUEUE_SKIP partition=$partition model=$model reason=existing_complete_output"
      continue
    fi
    # A separately launched, immutable inference for this exact output root
    # must finish before this queue can touch it; never create two writers for
    # one progress/probability directory.
    pattern="89_run_tusz_groupkfold_external_inference.py --model $model --fold 0 --seed 17 --manifest $manifest --output-root $partition_root"
    if pgrep -af "$pattern" | grep -v "$$" >/dev/null; then
      echo "H6_EXPERT_QUEUE_WAIT partition=$partition model=$model reason=existing_active_process"
      while pgrep -af "$pattern" | grep -v "$$" >/dev/null; do sleep 60; done
      if [[ -f "$run_root/progress.json" ]] && grep -q '"status": "complete"' "$run_root/progress.json"; then
        echo "H6_EXPERT_QUEUE_SKIP partition=$partition model=$model reason=parallel_output_complete"
        continue
      fi
    fi
    log="$LOG_ROOT/h6_tusz_expert_${partition}_${model}.log"
    echo "H6_EXPERT_QUEUE_START partition=$partition model=$model fold=0 seed=17"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/89_run_tusz_groupkfold_external_inference.py \
      --model "$model" --fold 0 --seed 17 --manifest "$manifest" --output-root "$partition_root" \
      2>&1 | tee "$log"
    echo "H6_EXPERT_QUEUE_DONE partition=$partition model=$model"
  done
done
echo "H6_EXPERT_QUEUE_COMPLETE"
