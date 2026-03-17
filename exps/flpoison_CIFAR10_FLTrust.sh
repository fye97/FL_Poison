#!/bin/bash

set -euo pipefail

submit_script="exps/flpoison_CIFAR10_FLTrust.sh"
preset_name="cifar10_fltrust"
slurm_time="0-12:00:00"
slurm_account="def-lincai"
slurm_gpus="nvidia_h100_80gb_hbm3_2g.20gb:1"
slurm_cpus_per_task="8"
slurm_mem="64G"
slurm_requeue="1"
slurm_output="/home/%u/FL_Poison/logs/slurm/%x_%A_%a.out"
slurm_error="/home/%u/FL_Poison/logs/slurm/%x_%A_%a.err"
slurm_mail_user="fengye@uvic.ca"
slurm_mail_type="ALL"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/flpoison_presets.sh"
load_flpoison_preset "${preset_name}"
# shellcheck source=/dev/null
source "${script_dir}/flpoison_run_common.sh"
