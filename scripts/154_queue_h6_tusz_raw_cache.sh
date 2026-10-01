#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/b_false_alarm_atlas
cd "$ROOT"
LOG_ROOT="$ROOT/logs"
mkdir -p "$LOG_ROOT"

for partition in train dev eval; do
  manifest="outputs/reports/h6-tusz-raw-pre-router-v1/manifests/${partition}_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "H6_RAW_QUEUE_STOP missing_manifest=$manifest" >&2
    exit 2
  fi
  echo "H6_RAW_QUEUE_START partition=$partition"
  .venv/bin/python scripts/148_prepare_h6_tusz_raw_router_data.py \
    --partition "$partition" --extract-raw --resume \
    2>&1 | tee "$LOG_ROOT/h6_raw_cache_${partition}.log"
  echo "H6_RAW_QUEUE_DONE partition=$partition"
done
echo "H6_RAW_QUEUE_COMPLETE"
