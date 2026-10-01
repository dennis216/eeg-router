#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
manifest=outputs/reports/v3-compact-extension-v1/tusz_eval_full_v1_manifest.json
out=outputs/tuh_external/tusz_eval_full_v1
for model in singlem cbramod tcn_gat; do
  echo "QUEUE_START model=${model} cohort=tusz-eval-full-v1 utc=$(date -u +%FT%TZ)"
  PYTHONPATH=src .venv/bin/python scripts/53_run_tusz_eval_subset_v1_inference.py \
    --model "${model}" --manifest "${manifest}" --output-root "${out}" \
    2>&1 | tee -a "logs/tusz_eval_full_v1_${model}.log"
  echo "QUEUE_COMPLETE model=${model} cohort=tusz-eval-full-v1 utc=$(date -u +%FT%TZ)"
done
echo "TUSZ_EVAL_FULL_V1_QUEUE_COMPLETE utc=$(date -u +%FT%TZ)"
