#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/10_cache_pilot_official_views.py >> logs/pilot_cache.log 2>&1
