#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  exps/run_cc.sh <spec> [cc args...]

Examples:
  exps/run_cc.sh smoke_mnist --chunk-size 1 --dry-run
  exps/run_cc.sh cifar10 --chunk-size 32
  exps/run_cc.sh cifar10 --start-id 0 --end-id 63 --chunk-size 16
EOF
}

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  py_bin="${PYTHON_BIN}"
elif [ -x "${repo_root}/.venv/bin/python" ]; then
  py_bin="${repo_root}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  py_bin="python"
else
  py_bin="python3"
fi

export CODE_SRC_ROOT="${CODE_SRC_ROOT:-${repo_root}}"
export PYTHON_BIN="${py_bin}"

mkdir -p "${repo_root}/logs/slurm"

exec "${py_bin}" "${script_dir}/launch.py" cc "$@"
