#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Backward-compatible one-command entry point. For two-stage operation, invoke
# build_text2sql_data.sh and train_text2sql.sh separately.
RUNS_ROOT="${RUNS_ROOT:-runs}"
mkdir -p "$RUNS_ROOT"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_pid$$"
RUN_ROOT="$RUNS_ROOT/$RUN_ID"
export RUNS_ROOT RUN_ROOT

echo "[run:phase-1] building data -> $RUN_ROOT"
bash scripts/build_text2sql_data.sh
echo "[run:phase-2] starting training -> $RUN_ROOT"
bash scripts/train_text2sql.sh "$RUN_ROOT" "$@"
