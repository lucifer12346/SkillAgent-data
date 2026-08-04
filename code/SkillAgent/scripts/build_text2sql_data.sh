#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_ROOT="${RUNS_ROOT:-runs}"
mkdir -p "$RUNS_ROOT"
if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ID="run_$(date +%Y%m%d_%H%M%S)_pid$$"
  RUN_ROOT="$RUNS_ROOT/$RUN_ID"
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "[build:error] RUN_ROOT already exists: $RUN_ROOT" >&2
  exit 2
fi
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/data"

LOG_FILE="$RUN_ROOT/logs/build_data.log"
exec > >(tee -a "$LOG_FILE") 2>&1

TRAIN_N="${TRAIN_N:-1000}"
VAL_N="${VAL_N:-200}"
BIRD_TEST_N="${BIRD_TEST_N:-${TEST_N:-2000}}"
SPLIT_SEED="${SPLIT_SEED:-42}"

echo "[build] run root -> $RUN_ROOT"
echo "[build] log -> $LOG_FILE"
echo "[build:1/2] BIRD train=$TRAIN_N, SynSQL val=$VAL_N, seed=$SPLIT_SEED"
python3 -u scripts/build_splits.py \
  --out_dir "$RUN_ROOT/data/text2sql_split" \
  --train_n "$TRAIN_N" \
  --val_n "$VAL_N" \
  --seed "$SPLIT_SEED"

echo "[build:2/2] BIRD test=$BIRD_TEST_N, EHRSQL, Spider2.0"
python3 -u scripts/build_test_sets.py \
  --config configs/default.yaml \
  --bird_n "$BIRD_TEST_N" \
  --seed "$SPLIT_SEED" \
  --out_dir "$RUN_ROOT/data/test_sets"

printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "train_n=$TRAIN_N" \
  "val_n=$VAL_N" \
  "bird_test_n=$BIRD_TEST_N" \
  "seed=$SPLIT_SEED" \
  > "$RUN_ROOT/data/DATA_READY"

RUN_ROOT_ABS="$(cd "$RUN_ROOT" && pwd)"
echo "[build:complete] prepared data -> $RUN_ROOT_ABS"
echo "[build:next] bash scripts/train_text2sql.sh \"$RUN_ROOT_ABS\""
