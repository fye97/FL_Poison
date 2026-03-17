#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --account=def-lincai
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/%u/FL_Poison/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/FL_Poison/logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=fengye@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-0%3

set -euo pipefail

submit_script="exps/compute_canada_flpoison.sh"
preset_name="omnibus"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/compute_canada_flpoison_presets.sh"
load_flpoison_preset "${preset_name}"
# shellcheck source=/dev/null
source "${script_dir}/compute_canada_flpoison_common.sh"
