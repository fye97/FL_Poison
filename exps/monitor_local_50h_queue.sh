#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/cnlab-pu/Projects/FL_Poison}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
STOP_AT="${STOP_AT:-2026-05-04 09:00:00}"
CORE_SPEC="exps/specs/CARAT/paper_neurips_core_public.yaml"
PLAYGROUND_ROOT="${PLAYGROUND_ROOT:-/home/cnlab-pu/Projects/Poisoning_Resilient_Federated_Learning_Playground}"

cd "$ROOT" || exit 1
mkdir -p logs/local_50h

stop_epoch="$(date -d "$STOP_AT" +%s)"

log_section() {
  printf '\n===== %s | %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

latest_metrics() {
  .venv/bin/python - <<'PY'
from __future__ import annotations

import csv
from pathlib import Path

root = Path("logs/local_runs")
paths = sorted(root.rglob("metrics_exp*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]
for path in paths:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        print(f"error reading {path}: {exc}")
        continue
    if not rows:
        print(f"empty {path}")
        continue
    last = rows[-1]
    epoch = last.get("epoch", "")
    acc = last.get("eval_acc") or last.get("train_acc") or ""
    loss = last.get("eval_loss") or last.get("train_loss") or ""
    complete = (path.parent / "task.complete").exists()
    print(
        f"{'complete' if complete else 'running '} "
        f"rows={len(rows):03d} epoch={epoch:>3} acc={acc} loss={loss} path={path}"
    )
PY
}

while [ "$(date +%s)" -lt "$stop_epoch" ]; do
  log_section "tmux sessions"
  tmux ls 2>/dev/null | grep -E 'carat_t8|local50h_followup|local50h_monitor' || true

  log_section "gpu"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

  log_section "python workers"
  ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd \
    | grep -E 'run_playground_incremental|exps/launch.py local|exps/launch.py worker|python -u -m flpoison' \
    | grep -v grep \
    | head -80 || true

  log_section "latest local metrics"
  latest_metrics || true

  log_section "missing exact"
  .venv/bin/python exps/playground_results.py missing "$CORE_SPEC" \
    --playground-root "$PLAYGROUND_ROOT" \
    --match-config exact \
    --format summary || true

  log_section "missing semantic any"
  .venv/bin/python exps/playground_results.py missing "$CORE_SPEC" \
    --playground-root "$PLAYGROUND_ROOT" \
    --match-config any \
    --format summary || true

  log_section "queue log tails"
  for log in logs/local_50h/carat_t8_*.log logs/local_50h/local50h_followup_*.log; do
    [ -f "$log" ] || continue
    echo "--- $log ---"
    tail -20 "$log" || true
  done

  log_section "status checks"
  carat_session="$(cat logs/local_50h/carat_t8.tmux 2>/dev/null || true)"
  follow_session="$(cat logs/local_50h/followup.tmux 2>/dev/null || true)"
  if [ -n "$carat_session" ] && ! tmux has-session -t "$carat_session" 2>/dev/null; then
    carat_log="logs/local_50h/${carat_session}.log"
    if grep -q 'incremental run complete' "$carat_log" 2>/dev/null; then
      echo "CARAT phase complete."
    else
      echo "ERROR: CARAT phase session ended without completion marker."
    fi
  else
    echo "CARAT phase still running or session unknown."
  fi
  if [ -n "$follow_session" ] && ! tmux has-session -t "$follow_session" 2>/dev/null; then
    follow_log="logs/local_50h/${follow_session}.log"
    if grep -q 'local 50h core queue complete' "$follow_log" 2>/dev/null; then
      echo "Follow-up queue complete."
    else
      echo "WARNING: follow-up queue session ended before final completion marker."
    fi
  else
    echo "Follow-up queue still running or waiting."
  fi

  sleep "$INTERVAL_SECONDS"
done

log_section "monitor finished"
echo "Reached STOP_AT=$STOP_AT"
