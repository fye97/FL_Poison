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
#SBATCH --array=0-0%3    # 建议提交时用 sbatch --array=0-(TOTAL-1)%K 覆盖（脚本会打印 TOTAL）

set -euo pipefail

# 加载环境（按你的 Compute Canada 环境修改）
CODE_SRC_ROOT="${CODE_SRC_ROOT:-$HOME/FL_Poison}"
DATA_SRC_ROOT="${DATA_SRC_ROOT:-$CODE_SRC_ROOT/data}"

source "${CODE_SRC_ROOT}/.venv/bin/activate"
cd "${CODE_SRC_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# -------------------
# 基础配置（可修改）
# -------------------
gpu_idx=0 # 通常 Slurm 下 CUDA_VISIBLE_DEVICES 已重映射为 0
array_parallel="${ARRAY_PARALLEL:-3}" # 对应 sbatch --array=...%${array_parallel}

# 重复实验（写死）
# - seed 从 yaml/CLI 的 seed 开始，按 experiment_id 递增
num_experiments=5
experiment_id=0

# -------------------
# 参数网格（可修改）
# -------------------
# 选择要跑的算法和数据集（不要求每个 {algorithm}_{dataset}_config.yaml 都存在）
algorithms=("FedAvg") # 例如 "FedAvg" "FedOpt" "FedSGD"
datasets=("CIFAR10") # 例如 "MNIST" "FashionMNIST" "EMNIST" "CIFAR10" "CIFAR100" "CINIC10" "CHMNIST" "TinyImageNet"

# 场景列表：alg|dataset|config_file
# 逻辑：
# - 优先用 ./configs/{alg}_{dataset}_config.yaml
# - 不存在则选一个“同模态”fallback config，并通过 CLI 覆盖 -alg/-data
scenarios=()
for alg in "${algorithms[@]}"; do
  for ds in "${datasets[@]}"; do
    cfg="./configs/${alg}_${ds}_config.yaml"
    if [ ! -f "${cfg}" ]; then
      case "${ds}" in
        MNIST|FashionMNIST|EMNIST)
          cfg="./configs/${alg}_MNIST_config.yaml"
          ;;
        *)
          cfg="./configs/${alg}_CIFAR10_config.yaml"
          ;;
      esac
      if [ ! -f "${cfg}" ]; then
        cfg="$(ls -1 ./configs/${alg}_*_config.yaml 2>/dev/null | head -n 1 || true)"
      fi
      if [ -z "${cfg}" ] || [ ! -f "${cfg}" ]; then
        echo "ERROR: no usable config found for alg=${alg} dataset=${ds} (searched ./configs/${alg}_*_config.yaml)." >&2
        exit 1
      fi
      echo "INFO: ./configs/${alg}_${ds}_config.yaml missing; use fallback $(basename "${cfg}") + CLI overrides (-alg/-data)." >&2
    fi
    scenarios+=("${alg}|${ds}|${cfg}")
  done
done

# 分布规格（避免 iid × alpha 这种无效笛卡尔积）
# 格式：distribution|dirichlet_alpha|im_iid_gamma
# 说明：只有当 distribution=non-iid 时使用 dirichlet_alpha；只有当 distribution=class-imbalanced_iid 时使用 im_iid_gamma。
dist_specs=(
  "iid|__cfg__|__cfg__"
  "non-iid|0.1|__cfg__"
  "non-iid|0.5|__cfg__"
  "non-iid|1|__cfg__"
)

# 训练超参（可用 "__cfg__" 表示使用 config 默认值）
models=("vgg19") # 例如 "resnet18" "lenet"
epochs_list=("200") # 例如 "50" "100"
num_clients_list=("20") # 例如 "20" "50"
learning_rates=("0.05") # 例如 "0.01" "0.05"
num_advs=("0.1") # 攻击者数量：支持比例（<1）或整数（>=1），例如 "0.2" 或 "4"
seeds=("42") # base seed; repeated experiments will use 42 + experiment_id

# attack / defense 组合
attacks=("NoAttack" "MinMax" "MinSum" "ALIE" "FangAttack")
defenses=("Mean" "TriGuardFL" "FLDetector" "FLTrust" "MultiKrum" "NormClipping")

n_scenarios=${#scenarios[@]}
n_dist_specs=${#dist_specs[@]}
n_models=${#models[@]}
n_epochs=${#epochs_list[@]}
n_clients=${#num_clients_list[@]}
n_lrs=${#learning_rates[@]}
n_advs=${#num_advs[@]}
n_seeds=${#seeds[@]}
n_attacks=${#attacks[@]}
n_defenses=${#defenses[@]}

if [ "${n_scenarios}" -eq 0 ] || [ "${n_dist_specs}" -eq 0 ] || [ "${n_models}" -eq 0 ] || [ "${n_epochs}" -eq 0 ] || \
   [ "${n_clients}" -eq 0 ] || [ "${n_lrs}" -eq 0 ] || [ "${n_advs}" -eq 0 ] || [ "${n_seeds}" -eq 0 ] || \
   [ "${n_attacks}" -eq 0 ] || [ "${n_defenses}" -eq 0 ]; then
  echo "ERROR: one of the grid arrays is empty." >&2
  exit 1
fi

total=$((n_scenarios * n_dist_specs * n_models * n_epochs * n_clients * n_lrs * n_advs * n_seeds * n_attacks * n_defenses))

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "TOTAL combinations: ${total}"
  echo "Grid dims:"
  echo "  scenarios=${n_scenarios}, dist_specs=${n_dist_specs}"
  echo "  models=${n_models}, epochs_list=${n_epochs}, num_clients_list=${n_clients}, learning_rates=${n_lrs}"
  echo "  num_advs=${n_advs}, seeds=${n_seeds}, attacks=${n_attacks}, defenses=${n_defenses}"
  echo "Submit example:"
  echo "  sbatch --array=0-$((total - 1))%${array_parallel} exps/compute_canada_flpoison_CIFAR10.sh"
  exit 0
fi

task_id="${SLURM_ARRAY_TASK_ID}"
if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} out of range (0..$((total - 1)))." >&2
  exit 1
fi

idx="${task_id}"

def_idx=$((idx % n_defenses)); idx=$((idx / n_defenses))
att_idx=$((idx % n_attacks)); idx=$((idx / n_attacks))
seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
adv_idx=$((idx % n_advs)); idx=$((idx / n_advs))
lr_idx=$((idx % n_lrs)); idx=$((idx / n_lrs))
client_idx=$((idx % n_clients)); idx=$((idx / n_clients))
epoch_idx=$((idx % n_epochs)); idx=$((idx / n_epochs))
model_idx=$((idx % n_models)); idx=$((idx / n_models))
distspec_idx=$((idx % n_dist_specs)); idx=$((idx / n_dist_specs))
scenario_idx=$((idx % n_scenarios)); idx=$((idx / n_scenarios))

if [ "${idx}" -ne 0 ]; then
  echo "ERROR: internal index mapping failed (idx=${idx})." >&2
  exit 1
fi

scenario="${scenarios[scenario_idx]}"
IFS='|' read -r algorithm dataset config_file <<<"${scenario}"
dist_spec="${dist_specs[distspec_idx]}"
IFS='|' read -r distribution dirichlet_alpha im_iid_gamma <<<"${dist_spec}"

model="${models[model_idx]}"
epochs="${epochs_list[epoch_idx]}"
num_clients="${num_clients_list[client_idx]}"
learning_rate="${learning_rates[lr_idx]}"
num_adv="${num_advs[adv_idx]}"
seed="${seeds[seed_idx]}"
attack="${attacks[att_idx]}"
defense="${defenses[def_idx]}"

if [ ! -f "${config_file}" ]; then
  echo "ERROR: config not found: ${config_file}" >&2
  exit 1
fi
config_name="$(basename "${config_file}")"

# 读取 config 默认值（用于 "__cfg__" 回退 + 日志命名）
cfg_line="$(
  "${PYTHON_BIN}" - "${config_file}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(f"Invalid config: {path}")

def _get(key, default=""):
    val = cfg.get(key, default)
    return "" if val is None else str(val)

keys = [
    "dataset",
    "model",
    "epochs",
    "num_clients",
    "learning_rate",
    "algorithm",
    "seed",
    "num_adv",
    "distribution",
    "dirichlet_alpha",
    "im_iid_gamma",
]
print("\t".join(_get(k) for k in keys))
PY
)"

IFS=$'\t' read -r cfg_dataset cfg_model cfg_epochs cfg_num_clients cfg_learning_rate cfg_algorithm cfg_seed cfg_num_adv cfg_distribution cfg_dirichlet_alpha cfg_im_iid_gamma <<<"${cfg_line}"

# 用 "__cfg__" 表示使用 config 默认值；并避免为不相关分布传递 alpha/gamma
if [ "${distribution}" = "__cfg__" ] || [ -z "${distribution}" ]; then
  distribution="${cfg_distribution}"
  dist_args=()
else
  dist_args=(-dtb "${distribution}")
fi

alpha_args=()
if [ "${distribution}" = "non-iid" ]; then
  if [ "${dirichlet_alpha}" = "__cfg__" ] || [ -z "${dirichlet_alpha}" ]; then
    dirichlet_alpha="${cfg_dirichlet_alpha}"
  else
    alpha_args=(-dirichlet_alpha "${dirichlet_alpha}")
  fi
else
  dirichlet_alpha=""
fi

gamma_args=()
if [ "${distribution}" = "class-imbalanced_iid" ]; then
  if [ "${im_iid_gamma}" = "__cfg__" ] || [ -z "${im_iid_gamma}" ]; then
    im_iid_gamma="${cfg_im_iid_gamma}"
  else
    gamma_args=(-im_iid_gamma "${im_iid_gamma}")
  fi
else
  im_iid_gamma=""
fi

if [ "${model}" = "__cfg__" ] || [ -z "${model}" ]; then
  model="${cfg_model}"
  model_args=()
else
  model_args=(-model "${model}")
fi

if [ "${epochs}" = "__cfg__" ] || [ -z "${epochs}" ]; then
  epochs="${cfg_epochs}"
  epoch_args=()
else
  epoch_args=(-e "${epochs}")
fi

if [ "${num_clients}" = "__cfg__" ] || [ -z "${num_clients}" ]; then
  num_clients="${cfg_num_clients}"
  client_args=()
else
  client_args=(-num_clients "${num_clients}")
fi

if [ "${learning_rate}" = "__cfg__" ] || [ -z "${learning_rate}" ]; then
  learning_rate="${cfg_learning_rate}"
  lr_args=()
else
  lr_args=(-lr "${learning_rate}")
fi

if [ "${num_adv}" = "__cfg__" ] || [ -z "${num_adv}" ]; then
  num_adv="${cfg_num_adv}"
  adv_args=()
else
  adv_args=(-num_adv "${num_adv}")
fi

if [ "${seed}" = "__cfg__" ] || [ -z "${seed}" ]; then
  seed="${cfg_seed}"
  seed_args=()
else
  seed_args=(-seed "${seed}")
fi

echo "Running:"
echo "  config=${config_name}"
echo "  scenario_alg=${algorithm} (cfg_alg=${cfg_algorithm})"
echo "  scenario_data=${dataset} (cfg_data=${cfg_dataset}) model=${model}"
echo "  distribution=${distribution} dirichlet_alpha=${dirichlet_alpha} im_iid_gamma=${im_iid_gamma}"
echo "  epochs=${epochs} num_clients=${num_clients} lr=${learning_rate}"
echo "  num_adv=${num_adv} seed=${seed}"
echo "  num_experiments=${num_experiments} experiment_id=${experiment_id}"
echo "  attack=${attack} defense=${defense}"

# 最终输出目录（写到 $SCRATCH；训练过程会尽量用 $SLURM_TMPDIR）
scratch_root="${SCRATCH:-${HOME}/scratch}"
log_dir="${scratch_root}/FL_Poison/logs/${algorithm}/${dataset}_${model}/${distribution}"
mkdir -p "${log_dir}"

extra_tag=""
if [ "${distribution}" = "non-iid" ] && [ -n "${dirichlet_alpha}" ]; then
  extra_tag="${extra_tag}_alpha${dirichlet_alpha}"
fi
if [ "${distribution}" = "class-imbalanced_iid" ] && [ -n "${im_iid_gamma}" ]; then
  extra_tag="${extra_tag}_gamma${im_iid_gamma}"
fi

cfg_tag="_cfg${config_name%.yaml}"
dest_output_file="${log_dir}/${dataset}_${model}_${distribution}_${attack}_${defense}_${epochs}_${num_clients}_${learning_rate}_${algorithm}_adv${num_adv}_seed${seed}${extra_tag}${cfg_tag}.txt"

# 传入重复实验参数（每个 array task 内串行重复）
exp_args=(--num_experiments "${num_experiments}" --experiment_id "${experiment_id}")

# -------------------
# CUDA sanity check (avoid silent CPU fallback)
# - Retries in a fresh python process (PyTorch caches CUDA init state per-process).
# - If still unavailable, optionally requeue this array task to get a different node.
# -------------------
cuda_retry_max="${CUDA_RETRY_MAX:-3}"
cuda_retry_sleep="${CUDA_RETRY_SLEEP:-20}" # seconds
cuda_requeue_on_fail="${CUDA_REQUEUE_ON_FAIL:-1}"
cuda_max_requeue="${CUDA_MAX_REQUEUE:-2}"

cuda_ok=0
for ((i=1; i<=cuda_retry_max; i++)); do
  if "${PYTHON_BIN}" - <<'PY'
import sys

import torch

try:
    ok = torch.cuda.is_available()
    if ok:
        torch.zeros(1, device="cuda:0")
except Exception as e:
    print(f"CUDA check failed: {e}", file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    cuda_ok=1
    break
  fi
  echo "WARN: CUDA not available on $(hostname) (try ${i}/${cuda_retry_max})." >&2
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  nvidia-smi -L >&2 || true
  if [ "${i}" -lt "${cuda_retry_max}" ]; then
    sleep "${cuda_retry_sleep}"
  fi
done

if [ "${cuda_ok}" -ne 1 ]; then
  echo "ERROR: CUDA still unavailable; abort to avoid CPU fallback." >&2

  if [ "${cuda_requeue_on_fail}" = "1" ] && command -v scontrol >/dev/null 2>&1; then
    # For array jobs, requeue only this task (e.g., 12345_6)
    job_to_requeue="${SLURM_JOB_ID:-}"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      job_to_requeue="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi

    restart_count="${SLURM_RESTART_COUNT:-0}"
    if [ -n "${job_to_requeue}" ] && [ "${restart_count}" -lt "${cuda_max_requeue}" ]; then
      echo "Requeuing ${job_to_requeue} (restart_count=${restart_count})." >&2
      # Try to avoid this node on requeue (best-effort)
      if [ -n "${SLURM_NODELIST:-}" ]; then
        scontrol update JobId="${job_to_requeue}" ExcNodeList="${SLURM_NODELIST}" >/dev/null 2>&1 || true
      fi
      if scontrol requeue "${job_to_requeue}" >/dev/null 2>&1; then
        exit 0
      fi
      echo "WARN: scontrol requeue failed; exiting." >&2
    else
      echo "WARN: max requeue reached (${cuda_max_requeue}) or unknown job id; exiting." >&2
    fi
  fi

  exit 1
fi

# -------------------
# Use node-local storage to reduce parallel filesystem I/O
# - stage code + required dataset subset into $SLURM_TMPDIR
# - write logs/plots locally, then copy once to $SCRATCH at the end
# -------------------
local_root="${SLURM_TMPDIR:-}"
local_repo=""
local_results_dir=""
if [ -n "${local_root}" ] && [ -d "${local_root}" ]; then
  local_run_dir="${local_root}/flpoison_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
  local_repo="${local_run_dir}/FL_Poison"
  local_results_dir="${local_run_dir}/results"

  mkdir -p "${local_repo}" "${local_results_dir}"

  _copy_dir() {
    local src="$1"
    local dst="$2"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${src}" "${dst}"
    else
      cp -a "${src}" "${dst}"
    fi
  }

  _copy_file() {
    local src="$1"
    local dst="$2"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${src}" "${dst}"
    else
      cp -a "${src}" "${dst}"
    fi
  }

  # Stage code (exclude large/ephemeral dirs).
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude '.git' \
      --exclude '.idea' \
      --exclude '.venv' \
      --exclude 'logs' \
      --exclude 'running_caches' \
      --exclude 'data' \
      "${CODE_SRC_ROOT}/" "${local_repo}/"
  else
    cp -a "${CODE_SRC_ROOT}/." "${local_repo}/"
    rm -rf "${local_repo}/.git" "${local_repo}/.idea" "${local_repo}/.venv" "${local_repo}/logs" "${local_repo}/running_caches" "${local_repo}/data" || true
  fi

  # Stage only the required dataset to local disk (avoid downloading).
  mkdir -p "${local_repo}/data"
  if [ -d "${DATA_SRC_ROOT}" ]; then
    case "${dataset}" in
      CIFAR10)
        [ -d "${DATA_SRC_ROOT}/cifar-10-batches-py" ] && _copy_dir "${DATA_SRC_ROOT}/cifar-10-batches-py" "${local_repo}/data/" || true
        ;;
      *)
        _copy_dir "${DATA_SRC_ROOT}/" "${local_repo}/data/" || true
        ;;
    esac
  else
    echo "WARN: DATA_SRC_ROOT not found: ${DATA_SRC_ROOT}. Dataset may download and stress shared FS." >&2
  fi
else
  echo "WARN: SLURM_TMPDIR not set; running on shared filesystem (may stress I/O)." >&2
fi

# Use local repo when available; otherwise fall back to shared code dir.
run_repo="${CODE_SRC_ROOT}"
if [ -n "${local_repo}" ] && [ -d "${local_repo}" ]; then
  run_repo="${local_repo}"
fi

# Local output file (copied to $SCRATCH at the end).
output_file="${dest_output_file}"
if [ -n "${local_results_dir}" ] && [ -d "${local_results_dir}" ]; then
  output_file="${local_results_dir}/$(basename "${dest_output_file}")"
fi

cd "${run_repo}"

"${PYTHON_BIN}" -u main.py \
  -config="${config_file}" \
  -alg "${algorithm}" \
  -data "${dataset}" \
  "${dist_args[@]}" \
  "${alpha_args[@]}" \
  "${gamma_args[@]}" \
  "${model_args[@]}" \
  "${epoch_args[@]}" \
  "${client_args[@]}" \
  "${lr_args[@]}" \
  "${adv_args[@]}" \
  "${seed_args[@]}" \
  "${exp_args[@]}" \
  -attack "${attack}" \
  -defense "${defense}" \
  -gidx "${gpu_idx}" \
  -o "${output_file}"

# Copy results back once (logs + plots are adjacent to output_file).
if [ -n "${local_results_dir}" ] && [ -d "${local_results_dir}" ]; then
  mkdir -p "${log_dir}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${local_results_dir}/" "${log_dir}/"
  else
    cp -a "${local_results_dir}/." "${log_dir}/"
  fi
  echo "Saved results to: ${log_dir}" >&2
else
  echo "Saved results to: ${log_dir}" >&2
fi
