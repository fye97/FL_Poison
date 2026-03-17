from aggregators import all_aggregators
from attackers import data_poisoning_attacks, model_poisoning_attacks

from config_utils import (
    list_preset_files,
    load_dataset_catalog,
    load_experiment_config,
    preset_path,
    resolve_config_path,
    validate_experiment_config,
)


KNOWN_ATTACKS = {"NoAttack", *model_poisoning_attacks, *data_poisoning_attacks}


def test_canonical_preset_path_resolves():
    resolved = resolve_config_path("configs/presets/FedSGD/MNIST.yaml")
    assert resolved == preset_path("FedSGD", "MNIST").resolve()


def test_legacy_aliases_are_rejected():
    for legacy_path in ("configs/FedSGD_MNIST_config.yaml", "configs/dataset_config.yaml"):
        try:
            resolve_config_path(legacy_path)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(f"{legacy_path} should not resolve")


def test_dataset_catalog_uses_native_yaml_lists():
    datasets = load_dataset_catalog()
    assert isinstance(datasets["MNIST"]["mean"], list)
    assert isinstance(datasets["MNIST"]["std"], list)


def test_all_presets_load_and_validate():
    for preset in list_preset_files():
        config = load_experiment_config(preset)
        validate_experiment_config(
            config,
            source=preset,
            known_attacks=KNOWN_ATTACKS,
            known_defenses=all_aggregators,
        )


def test_har_preset_excludes_image_backdoor_attacks():
    config = load_experiment_config(preset_path("FedAvg", "HAR"))
    attack_names = {item["attack"] for item in config["attacks"]}
    assert "BadNets" not in attack_names
    assert "BadNets_image" not in attack_names
    assert "DBA" not in attack_names
