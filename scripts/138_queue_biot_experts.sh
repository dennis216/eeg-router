#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for round_id in 0 1 2 3 4; do
  for model in singlem cbramod tcn_gat; do
    completed="runs/chb-mit-biot-router-minimal-v1/formal/round_${round_id}/${model}/completed.json"
    if [[ -f "$completed" ]]; then
      echo "BIOT_EXPERT_SKIP round=${round_id} model=${model}"
      continue
    fi
    echo "BIOT_EXPERT_START round=${round_id} model=${model}"
    .venv/bin/python scripts/133_run_biot_expert_round.py --round "$round_id" --model "$model" --updates 5000
    echo "BIOT_EXPERT_DONE round=${round_id} model=${model}"
  done
done
echo "BIOT_EXPERT_QUEUE_COMPLETE"
