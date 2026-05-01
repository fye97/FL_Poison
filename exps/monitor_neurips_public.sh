#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DURATION_HOURS="${DURATION_HOURS:-60}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
REPORT_ROOT="${REPORT_ROOT:-logs/reports/neurips_public_monitor}"
SUMMARY_ROOT="${SUMMARY_ROOT:-logs/reports/carat_neurips_public}"
SPEC_NAMES="paper_neurips_core_public,paper_neurips_clean_public,paper_neurips_ablation_fang_alpha05"

mkdir -p "$REPORT_ROOT" "$SUMMARY_ROOT" logs/run_logs

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

end_epoch=$(( $(date +%s) + DURATION_HOURS * 3600 ))
iteration=0

while (( $(date +%s) < end_epoch )); do
  iteration=$((iteration + 1))
  stamp="$(date +"%Y%m%d_%H%M%S")"
  report="${REPORT_ROOT}/audit_${stamp}.log"

  {
    echo "timestamp=$(timestamp)"
    echo "iteration=${iteration}"
    echo "duration_hours=${DURATION_HOURS}"
    echo
    echo "[tmux]"
    tmux ls 2>/dev/null | grep -E "carat_neurips_public_gpu|neurips_public_monitor" || true
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    echo
    echo "[processes]"
    pgrep -af -- "run_neurips_public_queue|paper_neurips_|monitor_neurips_public" || true
    echo
    echo "[summary]"
    .venv/bin/python exps/summarize_carat.py \
      --result-root logs/local_runs \
      --output-root "$SUMMARY_ROOT" \
      --spec-names "$SPEC_NAMES" || true
    echo
    echo "[audit]"
    .venv/bin/python exps/audit_neurips_public.py --show-partial 20 || true
  } 2>&1 | tee "$report"

  ln -sfn "$(basename "$report")" "${REPORT_ROOT}/latest.log"
  sleep "$INTERVAL_SEC"
done

echo "timestamp=$(timestamp)"
echo "monitor finished after ${DURATION_HOURS} hours"
