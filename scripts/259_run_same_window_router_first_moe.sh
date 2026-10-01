#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/b_false_alarm_atlas
PY="$ROOT/.venv/bin/python"
REPORT="$ROOT/outputs/reports/same-window-router-first-moe-v1"
LOG="$REPORT/pipeline.log"
ADJUSTMENTS="$REPORT/runtime_adjustments.jsonl"
mkdir -p "$REPORT"
cd "$ROOT"

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$LOG"; }
adjust() {
  local stage="$1" reason="$2" before="$3" after="$4"
  "$PY" - "$ADJUSTMENTS" "$stage" "$reason" "$before" "$after" <<'PY'
import json,sys
from datetime import datetime,timezone
path,stage,reason,before,after=sys.argv[1:]
with open(path,'a') as f:
    f.write(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'reason':reason,'before':before,'after':after,'test_used':False},sort_keys=True)+'\n')
PY
}

run_retry() {
  local stage="$1"; shift
  local attempt=1
  while true; do
    note "START stage=$stage attempt=$attempt command=$*"
    if "$@" >>"$LOG" 2>&1; then
      note "PASS stage=$stage attempt=$attempt"
      return 0
    else
      local code=$?
    fi
    adjust "$stage" "command_failed_exit_${code}" "attempt_${attempt}" "retry_same_mainline_after_gpu_cooldown"
    note "RETRY stage=$stage exit=$code; preserving mainline"
    attempt=$((attempt+1))
    if (( attempt > 3 )); then
      note "NEEDS_ATTENTION stage=$stage after_three_attempts"
      return "$code"
    fi
    sleep 300
  done
}

note "Waiting for existing NeuroTTT/meta-TTT GPU work to finish"
while pgrep -af 'chbmit_groupkfold_(meta_)?(train|evaluate)|chbmit_groupkfold_(meta_)?queue' >/dev/null; do
  sleep 300
done
note "Existing GPU queue is clear; starting same-window MoE"

for model in singlem cbramod tcn_gat; do
  run_retry "mit_expert_${model}" "$PY" scripts/255_train_mit_same_window_expert.py --model "$model" || exit $?
done

for partition in train val; do
  for model in singlem cbramod tcn_gat; do
    run_retry "tusz_${partition}_${model}" "$PY" scripts/256_infer_tusz_same_window_experts.py --partition "$partition" --model "$model" || exit $?
  done
done

run_retry expert_threshold_and_collapse_audit "$PY" scripts/252_select_audit_same_window_experts.py || exit $?
run_retry train_val_router_labels "$PY" scripts/253_build_same_window_router_labels.py --partitions train val || exit $?
run_retry biot_train "$PY" scripts/257_precompute_same_window_biot.py --partition train || exit $?
run_retry biot_val "$PY" scripts/257_precompute_same_window_biot.py --partition val || exit $?
run_retry router_train "$PY" scripts/254_train_same_window_router.py || exit $?

test -f "$REPORT/router_training/best.pt" || { note 'Router checkpoint missing; test remains unread'; exit 1; }
note "Router frozen; entering one-time final test"
for model in singlem cbramod tcn_gat; do
  run_retry "tusz_test_${model}" "$PY" scripts/256_infer_tusz_same_window_experts.py --partition test --model "$model" || exit $?
done
run_retry test_labels_for_scoring "$PY" scripts/253_build_same_window_router_labels.py --partitions test || exit $?
run_retry biot_test "$PY" scripts/257_precompute_same_window_biot.py --partition test || exit $?
run_retry final_test_score "$PY" scripts/258_score_same_window_moe_test.py || exit $?
note "COMPLETE same-window Router-first MoE"
