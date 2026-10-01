#!/usr/bin/env bash
set -u

cd /root/b_false_alarm_atlas
mkdir -p logs
queue_log=logs/multisplit_queue_split42_seed42.log
printf 'QUEUE_START %s\n' "$(date -Is)" >> "$queue_log"

if [[ ! -f runs/BSN-pilot-v1-multisplit/tcn_gat/split42_seed42_main/completed.json ]]; then
  printf 'QUEUE_WAIT_TCN %s\n' "$(date -Is)" >> "$queue_log"
  for _ in $(seq 1 240); do
    [[ -f runs/BSN-pilot-v1-multisplit/tcn_gat/split42_seed42_main/completed.json ]] && break
    sleep 30
  done
fi
if [[ ! -f runs/BSN-pilot-v1-multisplit/tcn_gat/split42_seed42_main/completed.json ]]; then
  printf 'QUEUE_STOP_TCN_TIMEOUT %s\n' "$(date -Is)" >> "$queue_log"
  exit 2
fi
printf 'QUEUE_TCN_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"

.venv/bin/python scripts/11_run_bsn_pilot.py \
  --model cbramod --split-seed 42 --model-seed 42 \
  --run-id BSN-pilot-v1-multisplit \
  >> logs/multisplit_cbramod_split42_seed42.log 2>&1
if [[ ! -f runs/BSN-pilot-v1-multisplit/cbramod/split42_seed42_main/completed.json ]]; then
  printf 'QUEUE_STOP_CBRA_FAILED %s\n' "$(date -Is)" >> "$queue_log"
  exit 3
fi
printf 'QUEUE_CBRA_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"

.venv/bin/python scripts/11_run_bsn_pilot.py \
  --model singlem --split-seed 42 --model-seed 42 \
  --run-id BSN-pilot-v1-multisplit \
  >> logs/multisplit_singlem_split42_seed42.log 2>&1
if [[ ! -f runs/BSN-pilot-v1-multisplit/singlem/split42_seed42_main/completed.json ]]; then
  printf 'QUEUE_STOP_SINGLEM_FAILED %s\n' "$(date -Is)" >> "$queue_log"
  exit 4
fi
printf 'QUEUE_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"
