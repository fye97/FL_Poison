#!/bin/bash

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source exps/flpoison_run_common.sh from a wrapper script." >&2
  exit 2
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARN: $*" >&2
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

RUNNER_LOG_DIR=""
RUNNER_LOCAL_RESULTS_DIR=""
RUNNER_RESULTS_SYNCED=0

sync_results_from_local() {
  if [ "${RUNNER_RESULTS_SYNCED}" -eq 1 ]; then
    return 0
  fi
  RUNNER_RESULTS_SYNCED=1

  if [ -n "${RUNNER_LOCAL_RESULTS_DIR}" ] && [ -d "${RUNNER_LOCAL_RESULTS_DIR}" ]; then
    mkdir -p "${RUNNER_LOG_DIR}"
    if have_cmd rsync; then
      rsync -a "${RUNNER_LOCAL_RESULTS_DIR}/" "${RUNNER_LOG_DIR}/"
    else
      cp -a "${RUNNER_LOCAL_RESULTS_DIR}/." "${RUNNER_LOG_DIR}/"
    fi
  fi
}

runner_on_exit() {
  local rc="$?"
  trap - EXIT INT TERM
  sync_results_from_local || true
  if [ -n "${RUNNER_LOG_DIR}" ]; then
    echo "Saved results to: ${RUNNER_LOG_DIR}" >&2
  fi
  exit "${rc}"
}

runner_on_signal() {
  local signal_name="$1"
  local signal_rc="$2"
  warn "Received ${signal_name}; syncing local results before exit."
  exit "${signal_rc}"
}

copy_path_if_exists() {
  local src="$1"
  local dst="$2"
  if [ ! -e "${src}" ]; then
    return 1
  fi

  if have_cmd rsync; then
    rsync -a "${src}" "${dst}"
  else
    cp -a "${src}" "${dst}"
  fi
}

copy_dir_contents() {
  local src="$1"
  local dst="$2"
  if [ ! -d "${src}" ]; then
    return 1
  fi

  if have_cmd rsync; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}/." "${dst}/"
  fi
}

apply_csv_override() {
  local array_name="$1"
  local env_name="$2"
  local env_value="${!env_name:-}"

  if [ -z "${env_value}" ]; then
    return 0
  fi

  local raw_items=()
  IFS=',' read -r -a raw_items <<< "${env_value}"

  local cleaned_items=()
  local item=""
  for item in "${raw_items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [ -n "${item}" ] && cleaned_items+=("${item}")
  done

  if [ "${#cleaned_items[@]}" -eq 0 ]; then
    die "${env_name} is set but no valid values were parsed."
  fi

  eval "${array_name}=()"
  local idx
  for idx in "${!cleaned_items[@]}"; do
    eval "${array_name}+=(\"\${cleaned_items[${idx}]}\")"
  done
}

build_experiment_ids() {
  local repeat_count="$1"
  local start_id="$2"

  experiment_ids=()

  if [ -n "${EXPERIMENT_IDS_CSV:-}" ]; then
    local raw_items=()
    IFS=',' read -r -a raw_items <<< "${EXPERIMENT_IDS_CSV}"
    local item=""
    for item in "${raw_items[@]}"; do
      item="${item#"${item%%[![:space:]]*}"}"
      item="${item%"${item##*[![:space:]]}"}"
      [ -n "${item}" ] || continue
      is_non_negative_int "${item}" || die "EXPERIMENT_IDS_CSV must contain non-negative integers only: ${item}"
      experiment_ids+=("${item}")
    done
    [ "${#experiment_ids[@]}" -gt 0 ] || die "EXPERIMENT_IDS_CSV is set but no valid experiment ids were parsed."
    return 0
  fi

  is_positive_int "${repeat_count}" || die "NUM_EXPERIMENTS must be a positive integer, got: ${repeat_count}"
  is_non_negative_int "${start_id}" || die "EXPERIMENT_ID_START must be a non-negative integer, got: ${start_id}"

  local i
  for ((i=0; i<repeat_count; i++)); do
    experiment_ids+=("$((start_id + i))")
  done
}

resolve_runtime_platform() {
  if [ -n "${SLURM_JOB_ID:-}" ] || [ -n "${SLURM_CLUSTER_NAME:-}" ] || [ -n "${SLURM_TMPDIR:-}" ]; then
    printf '%s\n' "compute_canada"
  else
    printf '%s\n' "local"
  fi
}

emit_submit_defaults() {
  printf 'SUBMIT_DEFAULTS'
  printf ' account=%q' "${slurm_account:-}"
  printf ' time=%q' "${slurm_time:-}"
  printf ' gpus=%q' "${slurm_gpus:-}"
  printf ' cpus_per_task=%q' "${slurm_cpus_per_task:-}"
  printf ' mem=%q' "${slurm_mem:-}"
  printf ' requeue=%q' "${slurm_requeue:-1}"
  printf ' output=%q' "${slurm_output:-/home/%u/FL_Poison/logs/slurm/%x_%A_%a.out}"
  printf ' error=%q' "${slurm_error:-/home/%u/FL_Poison/logs/slurm/%x_%A_%a.err}"
  printf ' mail_user=%q' "${slurm_mail_user:-}"
  printf ' mail_type=%q' "${slurm_mail_type:-}"
  printf ' array_parallel=%q' "${ARRAY_PARALLEL:-${default_array_parallel:-3}}"
  printf '\n'
}

resolve_config_for_scenario() {
  local alg="$1"
  local ds="$2"
  local cfg="./configs/${alg}_${ds}_config.yaml"
  local fallback_candidates=()

  if [ -f "${cfg}" ]; then
    printf '%s\n' "${cfg}"
    return 0
  fi

  case "${ds}" in
    MNIST|FashionMNIST|EMNIST)
      fallback_candidates+=("./configs/${alg}_MNIST_config.yaml")
      ;;
    *)
      fallback_candidates+=("./configs/${alg}_CIFAR10_config.yaml")
      ;;
  esac

  shopt -s nullglob
  local glob_matches=(./configs/${alg}_*_config.yaml)
  shopt -u nullglob
  fallback_candidates+=("${glob_matches[@]}")

  for candidate in "${fallback_candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      warn "./configs/${alg}_${ds}_config.yaml missing; using $(basename "${candidate}") with CLI overrides."
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

read_config_defaults() {
  local config_file="$1"

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
    "attack",
    "defense",
]
print("\t".join(_get(k) for k in keys))
PY
}

stage_dataset_to_local() {
  local dataset_name="$1"
  local src_root="$2"
  local dst_root="$3"
  local copied_any=0

  mkdir -p "${dst_root}"

  case "${dataset_name}" in
    MNIST)
      copy_path_if_exists "${src_root}/MNIST" "${dst_root}/" && copied_any=1 || true
      ;;
    FashionMNIST)
      copy_path_if_exists "${src_root}/FashionMNIST" "${dst_root}/" && copied_any=1 || true
      ;;
    EMNIST)
      copy_path_if_exists "${src_root}/EMNIST" "${dst_root}/" && copied_any=1 || true
      ;;
    CIFAR10)
      copy_path_if_exists "${src_root}/cifar-10-batches-py" "${dst_root}/" && copied_any=1 || true
      ;;
    CIFAR100)
      copy_path_if_exists "${src_root}/cifar-100-python" "${dst_root}/" && copied_any=1 || true
      ;;
    TinyImageNet)
      copy_path_if_exists "${src_root}/tiny-imagenet-200" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/tiny-imagenet-200.zip" "${dst_root}/" && copied_any=1 || true
      ;;
    CINIC10)
      copy_path_if_exists "${src_root}/CINIC-10" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/CINIC-10.tar.gz" "${dst_root}/" && copied_any=1 || true
      ;;
    CHMNIST)
      copy_path_if_exists "${src_root}/Kather_texture_2016_image_tiles_5000" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/Kather_texture_2016_image_tiles_5000.zip" "${dst_root}/" && copied_any=1 || true
      ;;
    HAR)
      copy_path_if_exists "${src_root}/UCI HAR Dataset" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/UCI_HAR_Dataset" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/UCI HAR Dataset.zip" "${dst_root}/" && copied_any=1 || true
      copy_path_if_exists "${src_root}/uci_har_cache_v1.npz" "${dst_root}/" && copied_any=1 || true
      ;;
    *)
      copy_dir_contents "${src_root}" "${dst_root}" && copied_any=1 || true
      ;;
  esac

  if [ "${copied_any}" -eq 0 ]; then
    warn "No pre-staged dataset assets found for ${dataset_name} under ${src_root}; the job may trigger downloads."
  fi
}

run_flpoison_experiment() {
  local submit_script_name="${submit_script:-exps/$(basename "${BASH_SOURCE[1]}")}"
  local wrapper_path="${BASH_SOURCE[1]}"
  local wrapper_name
  wrapper_name="$(basename "${submit_script_name}")"

  local runtime_platform
  runtime_platform="$(resolve_runtime_platform)"

  local script_dir
  script_dir="$(cd "$(dirname "${wrapper_path}")" && pwd)"
  local repo_root_guess
  repo_root_guess="$(cd "${script_dir}/.." && pwd)"
  local default_code_src_root="${CODE_SRC_ROOT:-${repo_root_guess}}"

  CODE_SRC_ROOT="${default_code_src_root}"
  [ -d "${CODE_SRC_ROOT}" ] || die "CODE_SRC_ROOT not found: ${CODE_SRC_ROOT}"
  [ -f "${CODE_SRC_ROOT}/main.py" ] || die "main.py missing under CODE_SRC_ROOT: ${CODE_SRC_ROOT}"
  [ -d "${CODE_SRC_ROOT}/configs" ] || die "configs directory missing under CODE_SRC_ROOT: ${CODE_SRC_ROOT}"

  if [ -z "${DATA_SRC_ROOT:-}" ]; then
    local data_candidates=()
    [ -n "${SCRATCH:-}" ] && data_candidates+=("${SCRATCH}/FL_Poison/data")
    [ -n "${PROJECT:-}" ] && data_candidates+=("${PROJECT}/FL_Poison/data")
    data_candidates+=("${CODE_SRC_ROOT}/data")
    for candidate in "${data_candidates[@]}"; do
      if [ -d "${candidate}" ]; then
        DATA_SRC_ROOT="${candidate}"
        break
      fi
    done
    DATA_SRC_ROOT="${DATA_SRC_ROOT:-${CODE_SRC_ROOT}/data}"
  fi

  local default_venv_path="${VENV_PATH:-${CODE_SRC_ROOT}/.venv}"
  if [ -n "${VIRTUAL_ENV:-}" ] && have_cmd python; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
  elif [ -x "${default_venv_path}/bin/python" ]; then
    # shellcheck source=/dev/null
    source "${default_venv_path}/bin/activate"
    PYTHON_BIN="${PYTHON_BIN:-${default_venv_path}/bin/python}"
  elif [ -n "${PYTHON_BIN:-}" ] && { [ -x "${PYTHON_BIN}" ] || have_cmd "${PYTHON_BIN}"; }; then
    :
  elif have_cmd python; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
    warn "Using python from PATH because no project virtualenv was found at ${default_venv_path}."
  else
    die "No usable Python runtime found. Set PYTHON_BIN or create ${default_venv_path}."
  fi
  [ -n "${PYTHON_BIN:-}" ] || die "PYTHON_BIN is empty after environment initialization."

  cd "${CODE_SRC_ROOT}"

  if ! declare -p algorithms >/dev/null 2>&1; then
    algorithms=("FedAvg")
  fi
  if ! declare -p datasets >/dev/null 2>&1; then
    datasets=("CIFAR10" "CIFAR100" "TinyImageNet")
  fi
  if ! declare -p dist_specs >/dev/null 2>&1; then
    dist_specs=(
      "iid|__cfg__|__cfg__"
      "non-iid|0.1|__cfg__"
      "non-iid|0.5|__cfg__"
      "non-iid|1|__cfg__"
    )
  fi
  if ! declare -p models >/dev/null 2>&1; then
    models=("vgg11" "resnet18")
  fi
  if ! declare -p epochs_list >/dev/null 2>&1; then
    epochs_list=("200")
  fi
  if ! declare -p num_clients_list >/dev/null 2>&1; then
    num_clients_list=("20")
  fi
  if ! declare -p learning_rates >/dev/null 2>&1; then
    learning_rates=("0.05")
  fi
  if ! declare -p num_advs >/dev/null 2>&1; then
    num_advs=("0.1" "0.2" "0.3")
  fi
  if ! declare -p seeds >/dev/null 2>&1; then
    seeds=("42")
  fi
  if ! declare -p attacks >/dev/null 2>&1; then
    attacks=("NoAttack" "MinMax" "MinSum" "ALIE" "FangAttack")
  fi
  if ! declare -p defenses >/dev/null 2>&1; then
    defenses=("Mean" "TriGuardFL" "FLDetector" "FLTrust" "MultiKrum" "NormClipping")
  fi

  apply_csv_override algorithms ALGORITHMS_CSV
  apply_csv_override datasets DATASETS_CSV
  apply_csv_override dist_specs DIST_SPECS_CSV
  apply_csv_override models MODELS_CSV
  apply_csv_override epochs_list EPOCHS_CSV
  apply_csv_override num_clients_list NUM_CLIENTS_CSV
  apply_csv_override learning_rates LEARNING_RATES_CSV
  apply_csv_override num_advs NUM_ADVS_CSV
  apply_csv_override seeds SEEDS_CSV
  apply_csv_override attacks ATTACKS_CSV
  apply_csv_override defenses DEFENSES_CSV

  local default_gpu_idx="${default_gpu_idx:-0}"
  local default_array_parallel="${default_array_parallel:-3}"
  local default_num_experiments="${default_num_experiments:-5}"
  local default_experiment_id="${default_experiment_id:-0}"
  local default_num_workers_override="${default_num_workers_override:-}"

  local gpu_idx="${GPU_IDX:-${default_gpu_idx}}"
  local array_parallel="${ARRAY_PARALLEL:-${default_array_parallel}}"
  local num_experiments="${NUM_EXPERIMENTS:-${default_num_experiments}}"
  local experiment_id_start="${EXPERIMENT_ID_START:-${default_experiment_id}}"
  local num_workers_override="${NUM_WORKERS:-${default_num_workers_override}}"

  is_positive_int "${array_parallel}" || die "ARRAY_PARALLEL must be a positive integer, got: ${array_parallel}"
  build_experiment_ids "${num_experiments}" "${experiment_id_start}"

  local scenarios=()
  local alg
  local ds
  local cfg
  for alg in "${algorithms[@]}"; do
    for ds in "${datasets[@]}"; do
      cfg="$(resolve_config_for_scenario "${alg}" "${ds}")" || die "No usable config found for alg=${alg} dataset=${ds}."
      scenarios+=("${alg}|${ds}|${cfg}")
    done
  done

  local n_scenarios="${#scenarios[@]}"
  local n_dist_specs="${#dist_specs[@]}"
  local n_models="${#models[@]}"
  local n_epochs="${#epochs_list[@]}"
  local n_clients="${#num_clients_list[@]}"
  local n_lrs="${#learning_rates[@]}"
  local n_advs="${#num_advs[@]}"
  local n_seeds="${#seeds[@]}"
  local n_attacks="${#attacks[@]}"
  local n_defenses="${#defenses[@]}"
  local n_experiment_ids="${#experiment_ids[@]}"

  if [ "${n_scenarios}" -eq 0 ] || [ "${n_dist_specs}" -eq 0 ] || [ "${n_models}" -eq 0 ] || [ "${n_epochs}" -eq 0 ] || \
     [ "${n_clients}" -eq 0 ] || [ "${n_lrs}" -eq 0 ] || [ "${n_advs}" -eq 0 ] || [ "${n_seeds}" -eq 0 ] || \
     [ "${n_attacks}" -eq 0 ] || [ "${n_defenses}" -eq 0 ] || [ "${n_experiment_ids}" -eq 0 ]; then
    die "One of the experiment grid arrays is empty."
  fi

  local total=$((n_scenarios * n_dist_specs * n_models * n_epochs * n_clients * n_lrs * n_advs * n_seeds * n_attacks * n_defenses * n_experiment_ids))

  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "TOTAL combinations: ${total}"
    echo "Grid dims:"
    echo "  scenarios=${n_scenarios}, dist_specs=${n_dist_specs}"
    echo "  models=${n_models}, epochs_list=${n_epochs}, num_clients_list=${n_clients}, learning_rates=${n_lrs}"
    echo "  num_advs=${n_advs}, seeds=${n_seeds}, attacks=${n_attacks}, defenses=${n_defenses}, experiment_ids=${n_experiment_ids}"
    echo "Experiment scheduling:"
    echo "  repeat_mode=split_per_task experiment_ids=${experiment_ids[*]}"
    echo "Local run examples:"
    echo "  python exps/local_run_array.py ${submit_script_name}"
    echo "  python exps/local_run_array.py ${submit_script_name} --ids 0-$((total - 1)) --jobs 1"
    echo "Compute Canada submit examples:"
    echo "  python exps/submit_compute_canada_chunks.py ${submit_script_name}"
    echo "  sbatch --array=0-$((total - 1))%${array_parallel} ${submit_script_name}"
    emit_submit_defaults
    exit 0
  fi

  local task_id="${SLURM_ARRAY_TASK_ID}"
  is_non_negative_int "${task_id}" || die "SLURM_ARRAY_TASK_ID must be a non-negative integer, got: ${task_id}"
  if [ "${task_id}" -lt 0 ] || [ "${task_id}" -ge "${total}" ]; then
    die "SLURM_ARRAY_TASK_ID=${task_id} out of range (0..$((total - 1)))."
  fi

  local idx="${task_id}"
  local expid_idx=$((idx % n_experiment_ids)); idx=$((idx / n_experiment_ids))
  local def_idx=$((idx % n_defenses)); idx=$((idx / n_defenses))
  local att_idx=$((idx % n_attacks)); idx=$((idx / n_attacks))
  local seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
  local adv_idx=$((idx % n_advs)); idx=$((idx / n_advs))
  local lr_idx=$((idx % n_lrs)); idx=$((idx / n_lrs))
  local client_idx=$((idx % n_clients)); idx=$((idx / n_clients))
  local epoch_idx=$((idx % n_epochs)); idx=$((idx / n_epochs))
  local model_idx=$((idx % n_models)); idx=$((idx / n_models))
  local distspec_idx=$((idx % n_dist_specs)); idx=$((idx / n_dist_specs))
  local scenario_idx=$((idx % n_scenarios)); idx=$((idx / n_scenarios))

  [ "${idx}" -eq 0 ] || die "Internal index mapping failed (idx=${idx})."

  local scenario="${scenarios[scenario_idx]}"
  local dist_spec="${dist_specs[distspec_idx]}"
  local algorithm dataset config_file distribution dirichlet_alpha im_iid_gamma
  IFS='|' read -r algorithm dataset config_file <<<"${scenario}"
  IFS='|' read -r distribution dirichlet_alpha im_iid_gamma <<<"${dist_spec}"

  local model="${models[model_idx]}"
  local epochs="${epochs_list[epoch_idx]}"
  local num_clients="${num_clients_list[client_idx]}"
  local learning_rate="${learning_rates[lr_idx]}"
  local num_adv="${num_advs[adv_idx]}"
  local seed="${seeds[seed_idx]}"
  local attack="${attacks[att_idx]}"
  local defense="${defenses[def_idx]}"
  local experiment_id="${experiment_ids[expid_idx]}"
  local effective_seed=$((seed + experiment_id))

  [ -f "${config_file}" ] || die "Config not found: ${config_file}"
  local config_name
  config_name="$(basename "${config_file}")"

  local cfg_line
  cfg_line="$(read_config_defaults "${config_file}")"

  local cfg_dataset cfg_model cfg_epochs cfg_num_clients cfg_learning_rate cfg_algorithm cfg_seed cfg_num_adv cfg_distribution cfg_dirichlet_alpha cfg_im_iid_gamma cfg_attack cfg_defense
  IFS=$'\t' read -r cfg_dataset cfg_model cfg_epochs cfg_num_clients cfg_learning_rate cfg_algorithm cfg_seed cfg_num_adv cfg_distribution cfg_dirichlet_alpha cfg_im_iid_gamma cfg_attack cfg_defense <<<"${cfg_line}"

  local dist_args=()
  if [ "${distribution}" = "__cfg__" ] || [ -z "${distribution}" ]; then
    distribution="${cfg_distribution}"
  else
    dist_args=(-dtb "${distribution}")
  fi

  local alpha_args=()
  if [ "${distribution}" = "non-iid" ]; then
    if [ "${dirichlet_alpha}" = "__cfg__" ] || [ -z "${dirichlet_alpha}" ]; then
      dirichlet_alpha="${cfg_dirichlet_alpha}"
    else
      alpha_args=(-dirichlet_alpha "${dirichlet_alpha}")
    fi
  else
    dirichlet_alpha=""
  fi

  local gamma_args=()
  if [ "${distribution}" = "class-imbalanced_iid" ]; then
    if [ "${im_iid_gamma}" = "__cfg__" ] || [ -z "${im_iid_gamma}" ]; then
      im_iid_gamma="${cfg_im_iid_gamma}"
    else
      gamma_args=(-im_iid_gamma "${im_iid_gamma}")
    fi
  else
    im_iid_gamma=""
  fi

  local model_args=()
  if [ "${model}" = "__cfg__" ] || [ -z "${model}" ]; then
    model="${cfg_model}"
  else
    model_args=(-model "${model}")
  fi

  local epoch_args=()
  if [ "${epochs}" = "__cfg__" ] || [ -z "${epochs}" ]; then
    epochs="${cfg_epochs}"
  else
    epoch_args=(-e "${epochs}")
  fi

  local client_args=()
  if [ "${num_clients}" = "__cfg__" ] || [ -z "${num_clients}" ]; then
    num_clients="${cfg_num_clients}"
  else
    client_args=(-num_clients "${num_clients}")
  fi

  local lr_args=()
  if [ "${learning_rate}" = "__cfg__" ] || [ -z "${learning_rate}" ]; then
    learning_rate="${cfg_learning_rate}"
  else
    lr_args=(-lr "${learning_rate}")
  fi

  local adv_args=()
  if [ "${num_adv}" = "__cfg__" ] || [ -z "${num_adv}" ]; then
    num_adv="${cfg_num_adv}"
  else
    adv_args=(-num_adv "${num_adv}")
  fi

  local seed_args=()
  if [ "${seed}" = "__cfg__" ] || [ -z "${seed}" ]; then
    seed="${cfg_seed}"
    effective_seed=$((seed + experiment_id))
  else
    seed_args=(-seed "${seed}")
  fi

  if [ "${attack}" = "__cfg__" ] || [ -z "${attack}" ]; then
    attack="${cfg_attack}"
  fi
  [ -n "${attack}" ] || die "Attack resolved to empty for config ${config_name}."

  if [ "${defense}" = "__cfg__" ] || [ -z "${defense}" ]; then
    defense="${cfg_defense}"
  fi
  [ -n "${defense}" ] || die "Defense resolved to empty for config ${config_name}."

  local worker_args=()
  if [ -n "${num_workers_override}" ]; then
    worker_args=(-num_workers "${num_workers_override}")
  fi

  local scratch_root="${SCRATCH:-${HOME}/scratch}"
  local default_result_root=""
  case "${runtime_platform}" in
    local)
      default_result_root="${CODE_SRC_ROOT}/logs/local_runs"
      ;;
    *)
      default_result_root="${scratch_root}/FL_Poison/logs"
      ;;
  esac
  local result_root="${RESULT_ROOT:-${default_result_root}}"
  local log_dir=""
  local extra_tag=""
  local cfg_tag=""
  local dest_output_file=""
  local local_results_dir=""
  local local_repo=""
  local local_run_dir=""
  local run_repo="${CODE_SRC_ROOT}"
  local output_file=""

  log_dir="${result_root}/${algorithm}/${dataset}_${model}/${distribution}"
  mkdir -p "${log_dir}"
  RUNNER_LOG_DIR="${log_dir}"
  RUNNER_LOCAL_RESULTS_DIR=""
  RUNNER_RESULTS_SYNCED=0

  if [ "${distribution}" = "non-iid" ] && [ -n "${dirichlet_alpha}" ]; then
    extra_tag="${extra_tag}_alpha${dirichlet_alpha}"
  fi
  if [ "${distribution}" = "class-imbalanced_iid" ] && [ -n "${im_iid_gamma}" ]; then
    extra_tag="${extra_tag}_gamma${im_iid_gamma}"
  fi

  cfg_tag="_cfg${config_name%.yaml}"
  dest_output_file="${log_dir}/${dataset}_${model}_${distribution}_${attack}_${defense}_${epochs}_${num_clients}_${learning_rate}_${algorithm}_adv${num_adv}_seed${seed}_exp${experiment_id}${extra_tag}${cfg_tag}.txt"

  trap runner_on_exit EXIT
  trap 'runner_on_signal TERM 143' TERM
  trap 'runner_on_signal INT 130' INT

  local exp_args=(--num_experiments "1" --experiment_id "${experiment_id}")

  echo "Job context:"
  echo "  script=${submit_script_name}"
  echo "  runtime_platform=${runtime_platform}"
  echo "  host=$(hostname)"
  echo "  job_id=${SLURM_JOB_ID:-n/a} array_task_id=${SLURM_ARRAY_TASK_ID:-n/a}"
  echo "  node=${SLURMD_NODENAME:-${SLURM_NODELIST:-$(hostname)}}"
  echo "  code_root=${CODE_SRC_ROOT}"
  echo "  data_root=${DATA_SRC_ROOT}"
  echo "  python_bin=${PYTHON_BIN}"
  echo "  python_version=$("${PYTHON_BIN}" --version 2>&1)"
  echo "  scratch_root=${scratch_root}"
  echo "  result_root=${result_root}"
  echo "  slurm_tmpdir=${SLURM_TMPDIR:-}"
  echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "Resolved experiment:"
  echo "  config=${config_name}"
  echo "  scenario_alg=${algorithm} (cfg_alg=${cfg_algorithm})"
  echo "  scenario_data=${dataset} (cfg_data=${cfg_dataset}) model=${model}"
  echo "  distribution=${distribution} dirichlet_alpha=${dirichlet_alpha} im_iid_gamma=${im_iid_gamma}"
  echo "  epochs=${epochs} num_clients=${num_clients} lr=${learning_rate}"
  echo "  num_adv=${num_adv} seed=${seed} effective_seed=${effective_seed}"
  echo "  experiment_id=${experiment_id} experiment_ids_count=${n_experiment_ids}"
  echo "  num_workers_override=${num_workers_override:-__cfg__}"
  echo "  attack=${attack} defense=${defense}"

  local cuda_retry_max="${CUDA_RETRY_MAX:-3}"
  local cuda_retry_sleep="${CUDA_RETRY_SLEEP:-20}"
  local cuda_requeue_on_fail="${CUDA_REQUEUE_ON_FAIL:-1}"
  local cuda_max_requeue="${CUDA_MAX_REQUEUE:-2}"
  local cuda_ok=0
  local i
  for ((i=1; i<=cuda_retry_max; i++)); do
    if "${PYTHON_BIN}" - <<'PY'
import sys

import torch

try:
    ok = torch.cuda.is_available()
    if ok:
        torch.zeros(1, device="cuda:0")
except Exception as exc:
    print(f"CUDA check failed: {exc}", file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
PY
    then
      cuda_ok=1
      break
    fi

    warn "CUDA not available on $(hostname) (try ${i}/${cuda_retry_max})."
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
    nvidia-smi -L >&2 || true
    if [ "${i}" -lt "${cuda_retry_max}" ]; then
      sleep "${cuda_retry_sleep}"
    fi
  done

  if [ "${cuda_ok}" -ne 1 ]; then
    if [ "${cuda_requeue_on_fail}" = "1" ] && have_cmd scontrol; then
      local job_to_requeue="${SLURM_JOB_ID:-}"
      if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        job_to_requeue="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
      fi

      local restart_count="${SLURM_RESTART_COUNT:-0}"
      if [ -n "${job_to_requeue}" ] && [ "${restart_count}" -lt "${cuda_max_requeue}" ]; then
        warn "CUDA unavailable; requeueing ${job_to_requeue} (restart_count=${restart_count})."
        if [ -n "${SLURM_NODELIST:-}" ]; then
          scontrol update JobId="${job_to_requeue}" ExcNodeList="${SLURM_NODELIST}" >/dev/null 2>&1 || true
        fi
        if scontrol requeue "${job_to_requeue}" >/dev/null 2>&1; then
          exit 0
        fi
        warn "scontrol requeue failed; falling back to a hard failure."
      else
        warn "Max CUDA requeue attempts reached (${cuda_max_requeue}) or job id unavailable."
      fi
    fi

    die "CUDA still unavailable; aborting to avoid silent CPU fallback."
  fi

  if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
    local_run_dir="${SLURM_TMPDIR}/flpoison_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
    local_repo="${local_run_dir}/FL_Poison"
    local_results_dir="${local_run_dir}/results"
    mkdir -p "${local_repo}" "${local_results_dir}"

    if have_cmd rsync; then
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

    if [ -d "${DATA_SRC_ROOT}" ]; then
      stage_dataset_to_local "${dataset}" "${DATA_SRC_ROOT}" "${local_repo}/data"
    else
      warn "DATA_SRC_ROOT not found: ${DATA_SRC_ROOT}. Dataset may download and stress shared filesystems."
    fi

    run_repo="${local_repo}"
    output_file="${local_results_dir}/$(basename "${dest_output_file}")"
    RUNNER_LOCAL_RESULTS_DIR="${local_results_dir}"
  else
    warn "SLURM_TMPDIR not set; running directly from the shared filesystem."
    output_file="${dest_output_file}"
  fi

  local metadata_dir="${log_dir}"
  if [ -n "${local_results_dir}" ]; then
    metadata_dir="${local_results_dir}"
  fi
  local metadata_file="${metadata_dir}/$(basename "${dest_output_file%.txt}")_jobmeta.txt"

  local cmd=(
    "${PYTHON_BIN}" -u main.py
    -config="${config_file}"
    -alg "${algorithm}"
    -data "${dataset}"
    "${dist_args[@]}"
    "${alpha_args[@]}"
    "${gamma_args[@]}"
    "${model_args[@]}"
    "${epoch_args[@]}"
    "${client_args[@]}"
    "${lr_args[@]}"
    "${adv_args[@]}"
    "${seed_args[@]}"
    "${worker_args[@]}"
    "${exp_args[@]}"
    -attack "${attack}"
    -defense "${defense}"
    -gidx "${gpu_idx}"
    -o "${output_file}"
  )

  {
    echo "script=${submit_script_name}"
    echo "wrapper_name=${wrapper_name}"
    echo "runtime_platform=${runtime_platform}"
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID:-}"
    echo "restart_count=${SLURM_RESTART_COUNT:-0}"
    echo "host=$(hostname)"
    echo "node=${SLURMD_NODENAME:-${SLURM_NODELIST:-$(hostname)}}"
    echo "code_root=${CODE_SRC_ROOT}"
    echo "data_root=${DATA_SRC_ROOT}"
    echo "run_repo=${run_repo}"
    echo "result_root=${result_root}"
    echo "output_file=${dest_output_file}"
    echo "runtime_output_file=${output_file}"
    echo "config_file=${config_file}"
    echo "experiment_id=${experiment_id}"
    echo "base_seed=${seed}"
    echo "effective_seed=${effective_seed}"
    echo "python_bin=${PYTHON_BIN}"
    echo "python_version=$("${PYTHON_BIN}" --version 2>&1)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    if have_cmd git; then
      echo "git_commit=$(git -C "${CODE_SRC_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
      echo "git_branch=$(git -C "${CODE_SRC_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    fi
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${metadata_file}"

  echo "Launch command:"
  printf '  %q ' "${cmd[@]}"
  printf '\n'

  (
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
    cd "${run_repo}"
    "${cmd[@]}"
  )
}

run_flpoison_experiment "$@"
