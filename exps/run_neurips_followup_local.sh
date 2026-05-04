#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnlab-pu/Projects/FL_Poison}"
PLAYGROUND_ROOT="${PLAYGROUND_ROOT:-/home/cnlab-pu/Projects/Poisoning_Resilient_Federated_Learning_Playground}"
PY="${PY:-$ROOT/.venv/bin/python}"
CUDA="${CUDA:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-15}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/local_followup}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/neurips_followup_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "root=$ROOT"
echo "playground_root=$PLAYGROUND_ROOT"
echo "python=$PY"
echo "cuda=$CUDA min_free_mb=$MIN_FREE_MB max_util=$MAX_UTIL poll_seconds=$POLL_SECONDS"
echo "log_file=$LOG_FILE"

run_phase() {
  local name="$1"
  local spec="$2"
  local ids="$3"

  echo
  echo "===== phase=$name started_at=$(date '+%Y-%m-%d %H:%M:%S') ====="
  "$PY" exps/playground_results.py missing "$spec" \
    --ids "$ids" \
    --playground-root "$PLAYGROUND_ROOT" \
    --match-config exact \
    --format summary

  "$PY" exps/run_playground_incremental.py "$spec" \
    --ids "$ids" \
    --playground-root "$PLAYGROUND_ROOT" \
    --cuda "$CUDA" \
    --min-free-mb "$MIN_FREE_MB" \
    --max-util "$MAX_UTIL" \
    --poll-seconds "$POLL_SECONDS" \
    --sleep-between "$SLEEP_BETWEEN" \
    --match-config exact

  "$PY" exps/playground_results.py missing "$spec" \
    --ids "$ids" \
    --playground-root "$PLAYGROUND_ROOT" \
    --match-config exact \
    --format summary
  echo "===== phase=$name finished_at=$(date '+%Y-%m-%d %H:%M:%S') ====="
}

run_phase \
  "mechanism_ablation" \
  "exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml" \
  "6-14"

run_phase \
  "task_and_reference_sensitivity" \
  "exps/specs/CARAT/paper_neurips_sensitivity_fang_alpha05.yaml" \
  "all"

run_phase \
  "matched_public_baselines" \
  "exps/specs/CARAT/paper_neurips_matched_baselines_public.yaml" \
  "all"

echo
echo "neurips_followup_queue_complete=$(date '+%Y-%m-%d %H:%M:%S')"
