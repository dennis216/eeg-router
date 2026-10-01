#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs
exec .venv/bin/python scripts/08_audit_singlem_units.py >> logs/singlem_audit.log 2>&1
