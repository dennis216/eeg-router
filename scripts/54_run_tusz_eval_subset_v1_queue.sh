#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
for model in singlem cbramod tcn_gat; do
  echo "QUEUE_START model=${model} utc=$(date -u +%FT%TZ)"
  PYTHONPATH=src .venv/bin/python scripts/53_run_tusz_eval_subset_v1_inference.py --model "${model}" \
    2>&1 | tee -a "logs/tusz_eval_subset_v1_${model}.log"
  echo "QUEUE_COMPLETE model=${model} utc=$(date -u +%FT%TZ)"
done
echo "TUSZ_EVAL_SUBSET_V1_QUEUE_COMPLETE utc=$(date -u +%FT%TZ)"
