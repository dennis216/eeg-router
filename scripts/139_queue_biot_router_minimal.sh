#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/biot_router_minimal_postexpert_queue_v1.log"
EXPERT_LOG="logs/biot_router_minimal_expert_queue_v1.log"

echo "POSTEXPERT_QUEUE_WAIT"
while ! grep -q "BIOT_EXPERT_QUEUE_COMPLETE" "$EXPERT_LOG" 2>/dev/null; do
  if ! ps -eo cmd | grep -E "[1]33_run_biot_expert_round.py|[1]38_queue_biot_experts.sh" >/dev/null; then
    echo "EXPERT_QUEUE_STOPPED_BEFORE_COMPLETE"
    exit 2
  fi
  sleep 60
done

EXPERT_COUNT="$(find runs/chb-mit-biot-router-minimal-v1/formal -name completed.json -print | wc -l)"
echo "EXPERT_COMPLETED_COUNT=${EXPERT_COUNT}"
if [[ "${EXPERT_COUNT}" -ne 15 ]]; then
  echo "EXPECTED_15_EXPERTS"
  exit 3
fi

echo "EMBEDDING_START"
.venv/bin/python scripts/135_precompute_biot_embeddings.py --batch-size 64 --workers 4

for ROUND_ID in 0 1 2 3 4; do
  echo "ROUTER_DATA_START round=${ROUND_ID}"
  .venv/bin/python scripts/134_build_biot_router_training_data.py --round "${ROUND_ID}"
  echo "ROUTER_START round=${ROUND_ID}"
  .venv/bin/python scripts/136_run_biot_router_round.py --round "${ROUND_ID}" --router-updates 300
done

echo "FINALIZE_START"
.venv/bin/python scripts/137_finalize_biot_router_minimal.py --bootstrap 2000
echo "BIOT_ROUTER_MINIMAL_COMPLETE"
