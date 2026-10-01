#!/usr/bin/env bash
set -euo pipefail

# Standard baseline addendum: the four conventional baselines use exactly the
# frozen patient-level GroupKFold manifests from v3-groupkfold-confirmatory-v1.
# This namespace is independent of the historical compact baseline runs.
ROOT="${BFA_ROOT:-/root/b_false_alarm_atlas}"
PYTHON="${ROOT}/.venv/bin/python"
SCRIPT_DIR="/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/b_false_alarm_atlas/scripts"
NAMESPACE="v3-groupkfold-baselines-v1"
LOG="${ROOT}/logs/${NAMESPACE}.log"
UPDATES="${UPDATES:-5000}"
SMOKE_UPDATES="${SMOKE_UPDATES:-0}"
WORKERS="${WORKERS:-4}"

mkdir -p "${ROOT}/logs"
exec > >(tee -a "${LOG}") 2>&1
echo "STANDARD_BASELINE_QUEUE_START namespace=${NAMESPACE} updates=${UPDATES} smoke_updates=${SMOKE_UPDATES}"
"${PYTHON}" "${SCRIPT_DIR}/110_preflight_no_duplicate_overlap.py" --namespace "${NAMESPACE}" --write-report

run_one() {
  local model="$1" fold="$2" seed="$3" script="$4"
  local split_file="${ROOT}/manifests/groupkfold_cv_v1/fold_${fold}.json"
  local args=("${PYTHON}" "${SCRIPT_DIR}/${script}" --model "${model}" --split-seed "${fold}" --split-file "${split_file}" --model-seed "${seed}" --workers "${WORKERS}" --output-namespace "${NAMESPACE}")
  if [[ "${SMOKE_UPDATES}" != "0" ]]; then
    args+=(--smoke-updates "${SMOKE_UPDATES}")
  else
    args+=(--updates "${UPDATES}")
  fi
  echo "STANDARD_BASELINE_START model=${model} fold=${fold} seed=${seed}"
  "${args[@]}"
  echo "STANDARD_BASELINE_DONE model=${model} fold=${fold} seed=${seed}"
}

for model_script in "eegnet:61_run_baseline.py" "psd_catboost:61_run_baseline.py" "deepconvnet:64_run_conv_baseline.py" "shallowconvnet:64_run_conv_baseline.py"; do
  model="${model_script%%:*}"
  script="${model_script##*:}"
  for fold in 0 1 2 3 4; do
    for seed in 17 42 3407; do
      run_one "${model}" "${fold}" "${seed}" "${script}"
    done
  done
done

echo "STANDARD_BASELINE_QUEUE_COMPLETE namespace=${NAMESPACE}"
