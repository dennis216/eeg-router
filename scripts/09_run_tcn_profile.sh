#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/09_profile_tcn_gat.py >> logs/tcn_profile.log 2>&1
