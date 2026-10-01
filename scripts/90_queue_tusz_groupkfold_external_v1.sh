#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
manifest=outputs/reports/v3-groupkfold-confirmatory-v1/tusz_groupkfold_external_manifest.json
out=outputs/tuh_external/tusz_groupkfold_external_v1
for model in singlem cbramod tcn_gat; do
  for fold in 0 1 2 3 4; do
    for seed in 17 42 3407; do
      echo "QUEUE_START model=${model} fold=${fold} seed=${seed} utc=$(date -u +%FT%TZ)"
      PYTHONPATH=src .venv/bin/python scripts/89_run_tusz_groupkfold_external_inference.py \
        --model "${model}" --fold "${fold}" --seed "${seed}" \
        --manifest "${manifest}" --output-root "${out}" \
        2>&1 | tee -a "logs/tusz_groupkfold_external_v1_${model}_fold${fold}_seed${seed}.log"
      echo "QUEUE_COMPLETE model=${model} fold=${fold} seed=${seed} utc=$(date -u +%FT%TZ)"
    done
  done
done
echo "TUSZ_GROUPKFOLD_EXTERNAL_V1_QUEUE_COMPLETE utc=$(date -u +%FT%TZ)"
