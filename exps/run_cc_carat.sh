#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/_bootstrap_env.sh"

usage() {
  cat <<'EOF'
Usage:
  exps/run_cc_carat.sh [cc args...]

Examples:
  exps/run_cc_carat.sh --dry-run
  exps/run_cc_carat.sh
  exps/run_cc_carat.sh --resume
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

for arg in "$@"; do
  if [ "${arg}" = "--chain-specs" ]; then
    echo "error: exps/run_cc_carat.sh only supports parallel submission; use exps/run_cc.sh for custom chaining." >&2
    exit 2
  fi
done

flpoison_bootstrap_python "${repo_root}"
py_bin="${PYTHON_BIN}"

mkdir -p "${repo_root}/logs/slurm"

exec "${py_bin}" "${script_dir}/launch.py" cc \
  CARAT/smoke_cifar100 \
  CARAT/pilot_untargeted \
  CARAT/clean_reference \
  CARAT/main_untargeted \
  CARAT/backdoor \
  CARAT/appendix_tinyimagenet \
  CARAT/appendix_chmnist \
  CARAT/ablations \
  "$@"
