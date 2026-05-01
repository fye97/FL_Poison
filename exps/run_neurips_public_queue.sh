#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${1:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-24000}"
POLL_SEC="${POLL_SEC:-300}"

CORE_SPEC="exps/specs/CARAT/paper_neurips_core_public.yaml"
CLEAN_SPEC="exps/specs/CARAT/paper_neurips_clean_public.yaml"
ABLATION_SPEC="exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml"

# Core task ids are grouped by launcher order:
#   CARAT: ids 6-8,15-17,24-26,33-35,42-44,51-53,60-62,69-71
#   TrimmedMean/FLTrust: all remaining public-baseline ids.
CORE_CARAT_IDS="${CORE_CARAT_IDS:-6-8,15-17,24-26,33-35,42-44,51-53,60-62,69-71}"
CORE_BASELINE_IDS="${CORE_BASELINE_IDS:-0-5,9-14,18-23,27-32,36-41,45-50,54-59,63-68}"
CLEAN_IDS="${CLEAN_IDS:-0-17}"

# The T=1 rows (ids 3-5) are already present in the shared playground and are
# intentionally not rerun here; they are used only as an ablation comparator.
ABLATION_IDS="${ABLATION_IDS:-0-2,6-14}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | awk 'NR==1 {print $1}'
}

wait_for_gpu() {
  local phase="$1"
  while true; do
    local free
    free="$(free_mb || echo 0)"
    echo "[$(timestamp)] phase=${phase} gpu=${GPU} free_mb=${free} threshold=${MIN_FREE_MB}"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_FREE_MB )); then
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

run_phase() {
  local phase="$1"
  local spec="$2"
  local ids="$3"
  if [[ -z "$ids" ]]; then
    echo "[$(timestamp)] skipping ${phase}: no ids requested"
    return 0
  fi
  wait_for_gpu "$phase"
  echo "[$(timestamp)] starting ${phase}: ${spec} ids=${ids}"
  .venv/bin/python exps/launch.py local "$spec" \
    --ids "$ids" \
    --resume \
    --cuda "$GPU" \
    --jobs 1 \
    --gpu-lock-dir "${GPU_LOCK_DIR:-gpu_locks_gpu${GPU}}" \
    --stop-on-fail
  echo "[$(timestamp)] completed ${phase}"
}

run_phase "core-carat-t8" "$CORE_SPEC" "$CORE_CARAT_IDS"
run_phase "core-public-baselines" "$CORE_SPEC" "$CORE_BASELINE_IDS"
run_phase "clean-utility" "$CLEAN_SPEC" "$CLEAN_IDS"
run_phase "focused-ablation" "$ABLATION_SPEC" "$ABLATION_IDS"

echo "[$(timestamp)] all queued NeurIPS public experiments completed"
