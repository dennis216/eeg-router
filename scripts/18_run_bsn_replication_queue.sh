#!/usr/bin/env bash
set -euo pipefail
cd /root/b_false_alarm_atlas
mkdir -p logs

run_one() {
  local model="$1"
  local seed="$2"
  local run_dir="runs/BSN-pilot-v1/${model}/split17_seed${seed}_main"
  local log="logs/pilot_${model}_split17_seed${seed}.log"

  if [[ -f "${run_dir}/completed.json" ]]; then
    echo "REPLICATION_SKIP_COMPLETE model=${model} seed=${seed}" | tee -a logs/pilot_replication_queue.log
    return
  fi

  local resume_args=()
  if [[ -d "${run_dir}/checkpoints" ]]; then
    local latest
    latest="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)"
    if [[ -n "${latest}" ]]; then
      resume_args=(--resume "${latest}")
    fi
  fi

  echo "REPLICATION_START model=${model} seed=${seed}" | tee -a logs/pilot_replication_queue.log
  .venv/bin/python scripts/11_run_bsn_pilot.py \
    --model "${model}" --split-seed 17 --model-seed "${seed}" "${resume_args[@]}" \
    >> "${log}" 2>&1
  test -f "${run_dir}/completed.json"
  echo "REPLICATION_COMPLETE model=${model} seed=${seed}" | tee -a logs/pilot_replication_queue.log
}

for seed in 42 3407; do
  for model in singlem cbramod tcn_gat; do
    run_one "${model}" "${seed}"
  done
done

echo "REPLICATION_QUEUE_COMPLETE" | tee -a logs/pilot_replication_queue.log
