#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs

while pgrep -f '[1]1_run_bsn_pilot.py --model singlem' >/dev/null; do
  sleep 30
done
if [[ ! -f runs/BSN-pilot-v1/singlem/split17_seed17_main/completed.json ]]; then
  echo 'QUEUE_STOP singlem did not complete' >> logs/pilot_queue.log
  exit 1
fi

echo 'QUEUE_START cbramod' >> logs/pilot_queue.log
bash scripts/11_run_model.sh cbramod
if [[ ! -f runs/BSN-pilot-v1/cbramod/split17_seed17_main/completed.json ]]; then
  echo 'QUEUE_STOP cbramod did not complete' >> logs/pilot_queue.log
  exit 1
fi

echo 'QUEUE_START tcn_gat' >> logs/pilot_queue.log
bash scripts/11_run_model.sh tcn_gat
if [[ ! -f runs/BSN-pilot-v1/tcn_gat/split17_seed17_main/completed.json ]]; then
  echo 'QUEUE_STOP tcn_gat did not complete' >> logs/pilot_queue.log
  exit 1
fi

echo 'QUEUE_COMPLETE all models' >> logs/pilot_queue.log
