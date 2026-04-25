#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/_bootstrap_env.sh"

flpoison_bootstrap_python "${repo_root}"
py_bin="${PYTHON_BIN}"
flpoison_require_python_imports "${py_bin}" yaml torch torchvision

mkdir -p "${repo_root}/logs/slurm"

exec "${py_bin}" "${script_dir}/launch.py" worker "$@"
