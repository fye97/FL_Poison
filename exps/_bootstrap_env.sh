#!/bin/bash

flpoison_resolve_bootstrap_path() {
  local repo_root="$1"
  local raw_path="$2"

  if [ -z "${raw_path}" ]; then
    return 1
  fi
  if [[ "${raw_path}" = /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi
  printf '%s\n' "${repo_root}/${raw_path}"
}

flpoison_source_cc_env() {
  local repo_root="$1"
  local env_script_raw="${CC_ENV_SCRIPT:-exps/cc_env.sh}"
  local env_script=""

  export FLPOISON_REPO_ROOT="${repo_root}"
  if ! env_script="$(flpoison_resolve_bootstrap_path "${repo_root}" "${env_script_raw}")"; then
    return 0
  fi
  if [ -f "${env_script}" ]; then
    # shellcheck disable=SC1090
    . "${env_script}"
  fi
}

flpoison_resolve_python_bin() {
  local repo_root="$1"

  if [ -n "${PYTHON_BIN:-}" ]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return 0
  fi
  if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
    printf '%s\n' "${UV_PROJECT_ENVIRONMENT}/bin/python"
    return 0
  fi
  if [ -n "${VENV_PATH:-}" ] && [ -x "${VENV_PATH}/bin/python" ]; then
    printf '%s\n' "${VENV_PATH}/bin/python"
    return 0
  fi
  if [ -x "${repo_root}/.venv/bin/python" ]; then
    printf '%s\n' "${repo_root}/.venv/bin/python"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  command -v python3
}

flpoison_bootstrap_python() {
  local repo_root="$1"
  local py_bin=""

  flpoison_source_cc_env "${repo_root}"
  py_bin="$(flpoison_resolve_python_bin "${repo_root}")"

  export CODE_SRC_ROOT="${CODE_SRC_ROOT:-${repo_root}}"
  export PYTHON_BIN="${py_bin}"
}
