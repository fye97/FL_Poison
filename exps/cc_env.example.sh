#!/bin/bash

# Copy this file to exps/cc_env.sh and edit it for your Compute Canada setup.
# The launcher will source exps/cc_env.sh automatically on both the login node
# and the compute node before resolving PYTHON_BIN.

# Example module bootstrap:
# module purge
# module load python/3.12 cuda/12.2

# Example 1: activate a uv-managed environment that already exists on shared storage.
# export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/flpoison"
# source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

# Example 2: activate a regular venv on shared storage.
# export VENV_PATH="$HOME/venvs/flpoison"
# source "${VENV_PATH}/bin/activate"

# Example 3: activate the repository-local venv created by `uv sync --frozen`.
# source "${FLPOISON_REPO_ROOT}/.venv/bin/activate"

# Always pin PYTHON_BIN after activation so the worker and the launcher agree.
# export PYTHON_BIN="$(command -v python)"
