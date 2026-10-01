#!/usr/bin/env bash
set -euo pipefail
model="${1:?model name required}"
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/13_cache_frozen_features.py --model "${model}" >> "logs/features_${model}.log" 2>&1
