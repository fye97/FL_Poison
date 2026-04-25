#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/_bootstrap_env.sh"

usage() {
  cat <<'EOF'
Usage:
  exps/run_local.sh <spec> [local args...]

Examples:
  exps/run_local.sh TriguardFL/smoke_mnist --ids 0 --jobs 1
  exps/run_local.sh TriguardFL/e1_cifar10 --ids 0-7 --jobs 1 --resume
EOF
}

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

flpoison_bootstrap_python "${repo_root}"
py_bin="${PYTHON_BIN}"
flpoison_require_python_imports "${py_bin}" yaml torch torchvision

mkdir -p \
  "${repo_root}/logs/local_array" \
  "${repo_root}/logs/local_runs"

exec "${py_bin}" "${script_dir}/launch.py" local "$@"
