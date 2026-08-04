#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_text2sql.sh RUN_ROOT [train.py overrides...]" >&2
  exit 2
fi
RUN_ROOT="$1"
shift

if [[ -s "$RUN_ROOT/data/DATA_READY" ]]; then
  READY_FILE="$RUN_ROOT/data/DATA_READY"
elif [[ -s "$RUN_ROOT/data/FILTERED_DATA_READY" ]]; then
  READY_FILE="$RUN_ROOT/data/FILTERED_DATA_READY"
else
  echo "[train:error] missing data readiness marker under $RUN_ROOT/data" >&2
  echo "[train:hint] run build_text2sql_data.sh or filter_annotated_data.py first" >&2
  exit 2
fi
required_files=(
  "$RUN_ROOT/data/text2sql_split/train/items.json"
  "$RUN_ROOT/data/text2sql_split/val/items.json"
  "$RUN_ROOT/data/test_sets/manifest.json"
  "$RUN_ROOT/data/test_sets/bird/items.json"
  "$RUN_ROOT/data/test_sets/ehrsql/items.json"
  "$RUN_ROOT/data/test_sets/spider2.0/items.json"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "$required_file" ]]; then
    echo "[train:error] missing or empty prepared-data file: $required_file" >&2
    echo "[train:hint] run scripts/build_text2sql_data.sh first" >&2
    exit 2
  fi
done
if [[ -e "$RUN_ROOT/output" ]]; then
  echo "[train:error] output already exists; refusing to overwrite: $RUN_ROOT/output" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"
mkdir "$RUN_ROOT/output"

LOG_FILE="$RUN_ROOT/logs/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1

export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://173.0.69.2:8888/v1,http://173.0.69.2:8889/v1}"
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-EMPTY}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-qwen3.6-35b-a3b}"
export QWEN_CHAT_TEMPERATURE="${QWEN_CHAT_TEMPERATURE:-0.0}"
export QWEN_CHAT_TIMEOUT="${QWEN_CHAT_TIMEOUT:-600}"
export QWEN_CHAT_RETRY_ROUNDS="${QWEN_CHAT_RETRY_ROUNDS:-2}"
export QWEN_CHAT_RETRY_BACKOFF="${QWEN_CHAT_RETRY_BACKOFF:-2}"

BATCH_NO_IMPROVEMENT_PATIENCE="${BATCH_NO_IMPROVEMENT_PATIENCE:-10}"

echo "[train] prepared run root -> $RUN_ROOT"
echo "[train] data readiness marker -> $READY_FILE"
echo "[train] log -> $LOG_FILE"
echo "[train] validation source=SynSQL temperature=0"
echo "[train] batch attempt policy is controlled by enable_repeated_batch_attempts in config/CLI"
echo "[train] repeated-mode patience -> $BATCH_NO_IMPROVEMENT_PATIENCE"
echo "[train] status command: python3 scripts/check_run_status.py $RUN_ROOT"

python3 -u scripts/train.py \
  --config configs/default.yaml \
  --split_dir "$RUN_ROOT/data/text2sql_split" \
  --test_sets_dir "$RUN_ROOT/data/test_sets" \
  --out_root "$RUN_ROOT/output" \
  --batch_no_improvement_patience "$BATCH_NO_IMPROVEMENT_PATIENCE" \
  "$@"
