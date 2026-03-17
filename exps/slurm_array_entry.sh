#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [ -x "${repo_root}/.venv/bin/python" ]; then
  py_bin="${repo_root}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  py_bin="python"
else
  py_bin="python3"
fi

exec "${py_bin}" "${script_dir}/launch.py" worker "$@"
