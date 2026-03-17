#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=def-lincai
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/%u/FL_Poison/logs/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/FL_Poison/logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=fengye@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-0%3

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/flpoison_FashionMNIST.sh" "$@"
