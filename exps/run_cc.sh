#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/_bootstrap_env.sh"

usage() {
  cat <<'EOF'
Usage:
  exps/run_cc.sh <spec> [cc args...]
  exps/run_cc.sh <spec1> <spec2> ... --chain-specs [cc args...]

Examples:
  exps/run_cc.sh TriguardFL/smoke_mnist --chunk-size 1 --dry-run
  exps/run_cc.sh TriguardFL/omnibus --chunk-size 32
  exps/run_cc.sh TriguardFL/omnibus --start-id 0 --end-id 63 --chunk-size 16
  exps/run_cc.sh CARAT/clean_reference CARAT/pilot_untargeted --chain-specs --dry-run
EOF
}

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

flpoison_bootstrap_python "${repo_root}"
py_bin="${PYTHON_BIN}"

mkdir -p "${repo_root}/logs/slurm"

exec "${py_bin}" "${script_dir}/launch.py" cc "$@"
