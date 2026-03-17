#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source exps/compute_canada_flpoison_presets.sh from a wrapper script." >&2
  exit 2
fi

# 实验者主要在这里维护 attack/defense grid。
# 临时覆盖也可以直接传环境变量：
#   ATTACKS_CSV='NoAttack,MinMax'
#   DEFENSES_CSV='Mean,FLTrust'

use_config_defaults_training_grid() {
  dist_specs=("__cfg__|__cfg__|__cfg__")
  models=("__cfg__")
  epochs_list=("__cfg__")
  num_clients_list=("__cfg__")
  learning_rates=("__cfg__")
  num_advs=("__cfg__")
  seeds=("__cfg__")
}

use_default_attack_grid() {
  attacks=("NoAttack" "MinMax" "MinSum" "ALIE" "FangAttack")
}

use_default_defense_grid() {
  defenses=("Mean" "TriGuardFL" "FLDetector" "FLTrust" "MultiKrum" "NormClipping")
}

load_flpoison_preset() {
  local preset_name="$1"

  default_array_parallel=3
  default_num_experiments=5
  default_experiment_id=0
  default_gpu_idx=0

  case "${preset_name}" in
    omnibus)
      algorithms=("FedAvg")
      datasets=("CIFAR10" "CIFAR100" "TinyImageNet")
      use_config_defaults_training_grid
      use_default_attack_grid
      use_default_defense_grid
      ;;
    cifar10)
      algorithms=("FedAvg")
      datasets=("CIFAR10")
      use_config_defaults_training_grid
      use_default_attack_grid
      use_default_defense_grid
      ;;
    cifar10_fltrust)
      algorithms=("FedAvg")
      datasets=("CIFAR10")
      use_config_defaults_training_grid
      use_default_attack_grid
      defenses=("FLTrust")
      ;;
    cifar100)
      algorithms=("FedAvg")
      datasets=("CIFAR100")
      use_config_defaults_training_grid
      use_default_attack_grid
      use_default_defense_grid
      ;;
    fashion_mnist)
      algorithms=("FedAvg")
      datasets=("FashionMNIST")
      use_config_defaults_training_grid
      use_default_attack_grid
      use_default_defense_grid
      ;;
    tiny_imagenet)
      algorithms=("FedAvg")
      datasets=("TinyImageNet")
      use_config_defaults_training_grid
      use_default_attack_grid
      use_default_defense_grid
      ;;
    *)
      echo "ERROR: unknown FL Poison preset: ${preset_name}" >&2
      return 1
      ;;
  esac
}
