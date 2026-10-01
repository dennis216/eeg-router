#!/usr/bin/env bash
set -euo pipefail
model="${1:?model name required}"
shift
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/11_run_bsn_pilot.py --model "${model}" "$@" >> "logs/pilot_${model}.log" 2>&1
