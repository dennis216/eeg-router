#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs

while pgrep -f '[1]3_cache_frozen_features.py --model singlem' >/dev/null; do
  sleep 30
done
.venv/bin/python scripts/14_verify_feature_cache.py --model singlem >> logs/pilot_recovery.log 2>&1

bash scripts/13_run_feature_cache.sh cbramod
.venv/bin/python scripts/14_verify_feature_cache.py --model cbramod >> logs/pilot_recovery.log 2>&1

singlem_checkpoint=runs/BSN-pilot-v1/singlem/split17_seed17_main/checkpoints/step_01000.pt
bash scripts/11_run_model.sh singlem --resume "${singlem_checkpoint}"
test -f runs/BSN-pilot-v1/singlem/split17_seed17_main/completed.json

bash scripts/11_run_model.sh cbramod
test -f runs/BSN-pilot-v1/cbramod/split17_seed17_main/completed.json

bash scripts/11_run_model.sh tcn_gat
test -f runs/BSN-pilot-v1/tcn_gat/split17_seed17_main/completed.json

echo 'RECOVERY_QUEUE_COMPLETE all models' >> logs/pilot_recovery.log
