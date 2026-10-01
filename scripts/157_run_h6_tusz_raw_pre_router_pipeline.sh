#!/usr/bin/env bash
set -euo pipefail

# End-to-end resumable execution after the frozen-expert queue. Every stage
# writes to the new H6 namespace and refuses to overwrite an existing formal
# artifact. The script waits for an already-running expert queue rather than
# launching a duplicate.

ROOT=/root/b_false_alarm_atlas
cd "$ROOT"
OUT=outputs/reports/h6-tusz-raw-pre-router-v1
EXPERT=outputs/tuh_external/h6_tusz_raw_pre_router_v1
LOG=logs/h6_tusz_raw_pre_router_pipeline.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "H6_PIPELINE_START $(date -Is)"
expert_complete() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  if grep -q '"status": "complete"' "$p"; then
    return 0
  fi
  # Some completed inference jobs wrote the full record count but retained
  # the non-terminal status string. Treat an exact count with zero failures
  # as complete; do not accept a partial/stale progress file.
  local done total failed
  done=$(sed -n 's/.*"records_completed":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$p" | head -n 1)
  total=$(sed -n 's/.*"records_total":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$p" | head -n 1)
  failed=$(sed -n 's/.*"records_failed":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$p" | head -n 1)
  [[ -n "$done" && -n "$total" && "$done" -eq "$total" && "${failed:-0}" -eq 0 ]]
}
for partition in train dev eval; do
  for model in singlem cbramod tcn_gat; do
    progress="$EXPERT/$partition/$model/fold0_seed17/progress.json"
    while ! expert_complete "$progress"; do
      if [[ -f "$progress" ]] && grep -q '"records_failed": [1-9]' "$progress"; then
        echo "H6_PIPELINE_STOP expert_failures=$progress" >&2
        exit 2
      fi
      echo "H6_PIPELINE_WAIT partition=$partition model=$model"
      sleep 60
    done
    if grep -q '"records_failed": [1-9]' "$progress"; then
      echo "H6_PIPELINE_STOP expert_failures=$progress" >&2
      exit 2
    fi
  done
done

echo "H6_PIPELINE_EXPERTS_COMPLETE"
for partition in train dev eval; do
  # Always invoke the resumable extractor for a formal resume. A previous
  # progress marker can say complete even when an interrupted NPZ write left
  # a truncated file; script 148 validates every existing file and rebuilds
  # only invalid entries. Skipping on the marker alone would preserve silent
  # corruption and make the downstream training non-reproducible.
  echo "H6_PIPELINE_RAW_CACHE_VERIFY partition=$partition"
  .venv/bin/python scripts/148_prepare_h6_tusz_raw_router_data.py --partition "$partition" --extract-raw --resume
done

if [[ -f scripts/110_preflight_no_duplicate_overlap.py ]]; then
  BFA_ROOT="$ROOT" .venv/bin/python scripts/110_preflight_no_duplicate_overlap.py --namespace h6-tusz-raw-pre-router-v1 --write-report
fi

CALIB="$OUT/calibration/train_platt_formal.json"
TRAIN_LABEL="$OUT/labels/train_formal_fold0_seed17.parquet"
DEV_LABEL="$OUT/labels/dev_formal_fold0_seed17.parquet"
EVAL_LABEL="$OUT/labels/eval_formal_fold0_seed17.parquet"
if [[ ! -f "$TRAIN_LABEL" ]]; then
  .venv/bin/python scripts/151_build_h6_tusz_pre_router_labels.py --partition train --manifest "$OUT/manifests/train_manifest.json" --index "$OUT/indices/train_index.parquet" --input-root "$EXPERT/train" --output-name fold0_seed17 --label-id formal_fold0_seed17 --calibration "$CALIB" --fit-calibration
fi
if [[ ! -f "$DEV_LABEL" ]]; then
  .venv/bin/python scripts/151_build_h6_tusz_pre_router_labels.py --partition dev --manifest "$OUT/manifests/dev_manifest.json" --index "$OUT/indices/dev_index.parquet" --input-root "$EXPERT/dev" --output-name fold0_seed17 --label-id formal_fold0_seed17 --calibration "$CALIB"
fi
if [[ ! -f "$EVAL_LABEL" ]]; then
  .venv/bin/python scripts/151_build_h6_tusz_pre_router_labels.py --partition eval --manifest "$OUT/manifests/eval_manifest.json" --index "$OUT/indices/eval_index.parquet" --input-root "$EXPERT/eval" --output-name fold0_seed17 --label-id formal_fold0_seed17 --calibration "$CALIB"
fi

for partition in dev eval; do
  if [[ ! -f "$OUT/oracle_${partition}.json" ]]; then
    .venv/bin/python scripts/159_compute_h6_oracle_ceiling.py --partition "$partition" \
      --labels "$OUT/labels/${partition}_formal_fold0_seed17.parquet" \
      --manifest "$OUT/manifests/${partition}_manifest.json" \
      --expert-root "$EXPERT/$partition" --output "$OUT/oracle_${partition}.json"
  fi
done

TRAINING="$OUT/training/formal_router"
if [[ ! -f "$TRAINING/progress.json" ]] || ! grep -q '"status": "complete"' "$TRAINING/progress.json"; then
  .venv/bin/python scripts/152_train_h6_tusz_raw_pre_router.py --partition train --labels "$TRAIN_LABEL" --cache-root "$OUT/raw_cache/train" --output-dir "$TRAINING" --updates 2000 --batch-size 16 --seed 3407 --checkpoint-every 250 --resume
fi
CHECKPOINT="$TRAINING/checkpoints/router_step_002000.pt"
DEV_SCORE="$OUT/scoring/dev_formal"
if [[ ! -f "$DEV_SCORE/threshold_selection.json" ]]; then
  .venv/bin/python scripts/155_score_h6_tusz_raw_pre_router.py dev --labels "$DEV_LABEL" --checkpoint "$CHECKPOINT" --expert-root "$EXPERT/dev" --manifest "$OUT/manifests/dev_manifest.json" --grid 0.10,0.30,0.50,0.70,0.90 --output-dir "$DEV_SCORE"
fi
EVAL_SCORE="$OUT/scoring/eval_formal"
if [[ ! -f "$EVAL_SCORE/manifest.json" ]]; then
  .venv/bin/python scripts/155_score_h6_tusz_raw_pre_router.py eval --labels "$EVAL_LABEL" --checkpoint "$CHECKPOINT" --expert-root "$EXPERT/eval" --thresholds "$DEV_SCORE/threshold_selection.json" --manifest "$OUT/manifests/eval_manifest.json" --bootstrap 2000 --output-dir "$EVAL_SCORE"
fi
.venv/bin/python scripts/156_audit_h6_tusz_raw_pre_router.py --training-checkpoint "$CHECKPOINT" --eval-scoring-dir "$EVAL_SCORE" --require-complete
echo "H6_PIPELINE_COMPLETE $(date -Is)"
