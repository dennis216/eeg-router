#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/06_cache_views.py --workers 6 >> logs/cache_views.log 2>&1
