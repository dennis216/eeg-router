#!/usr/bin/env bash
set -u

cd /root/b_false_alarm_atlas
mkdir -p logs
queue_log=logs/multisplit_queue.log
tcn_run=runs/BSN-pilot-v1-multisplit/tcn_gat/split42_seed17_main

printf 'QUEUE_START %s\n' "$(date -Is)" >> "$queue_log"

# Wait for the current TCN-GAT process to finish.  The two-hour guard avoids
# silently launching a downstream model if the upstream run is abandoned.
for _ in $(seq 1 240); do
  if [[ -f "$tcn_run/completed.json" ]]; then
    printf 'QUEUE_TCN_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"
    break
  fi
  sleep 30
done

if [[ ! -f "$tcn_run/completed.json" ]]; then
  printf 'QUEUE_STOP_TCN_TIMEOUT %s\n' "$(date -Is)" >> "$queue_log"
  exit 2
fi

.venv/bin/python scripts/11_run_bsn_pilot.py \
  --model cbramod --split-seed 42 --model-seed 17 \
  --run-id BSN-pilot-v1-multisplit \
  >> logs/multisplit_cbramod_split42_seed17.log 2>&1
if [[ ! -f runs/BSN-pilot-v1-multisplit/cbramod/split42_seed17_main/completed.json ]]; then
  printf 'QUEUE_STOP_CBRA_FAILED %s\n' "$(date -Is)" >> "$queue_log"
  exit 3
fi
printf 'QUEUE_CBRA_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"

.venv/bin/python scripts/11_run_bsn_pilot.py \
  --model singlem --split-seed 42 --model-seed 17 \
  --run-id BSN-pilot-v1-multisplit \
  >> logs/multisplit_singlem_split42_seed17.log 2>&1
if [[ ! -f runs/BSN-pilot-v1-multisplit/singlem/split42_seed17_main/completed.json ]]; then
  printf 'QUEUE_STOP_SINGLEM_FAILED %s\n' "$(date -Is)" >> "$queue_log"
  exit 4
fi
printf 'QUEUE_COMPLETE %s\n' "$(date -Is)" >> "$queue_log"
