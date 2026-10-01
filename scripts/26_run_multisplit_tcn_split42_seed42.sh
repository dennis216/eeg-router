#!/usr/bin/env bash
set -u
cd /root/b_false_alarm_atlas
exec .venv/bin/python scripts/11_run_bsn_pilot.py \
  --model tcn_gat --split-seed 42 --model-seed 42 \
  --run-id BSN-pilot-v1-multisplit \
  >> logs/multisplit_tcn_gat_split42_seed42.log 2>&1
