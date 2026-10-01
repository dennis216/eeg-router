#!/usr/bin/env bash
set -u
cd /root/b_false_alarm_atlas
exec .venv/bin/python scripts/11_run_bsn_pilot.py \
  --model tcn_gat --split-seed 42 --model-seed 17 \
  --run-id BSN-pilot-v1-multisplit \
  --resume runs/BSN-pilot-v1-multisplit/tcn_gat/split42_seed17_main/checkpoints/step_02000.pt \
  >> logs/multisplit_tcn_gat_resume.log 2>&1
