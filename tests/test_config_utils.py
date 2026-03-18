from flpoison.aggregators import all_aggregators
from flpoison.attackers import data_poisoning_attacks, model_poisoning_attacks

from flpoison.utils.config_utils import (
    attacks_catalog_path,
    defenses_catalog_path,
    list_preset_files,
    load_dataset_catalog,
    load_experiment_config,
    load_yaml_mapping,
    preset_path,
    resolve_config_path,
    validate_experiment_config,
)


KNOWN_ATTACKS = {"NoAttack", *model_poisoning_attacks, *data_poisoning_attacks}
EXPECTED_SHARED_ATTACKS = {
    "NoAttack",
    "IPM",
    "ALIE",
    "Gaussian",
    "Mimic",
    "MinMax",
    "MinSum",
    "FangAttack",
    "BadNets",
    "BadNets_image",
    "LabelFlipping",
    "ModelReplacement",
    "DBA",
    "EdgeCase",
    "Neurotoxin",
    "AlterMin",
}
EXPECTED_SHARED_DEFENSES = {
    "Mean",
    "SimpleClustering",
    "Krum",
    "MultiKrum",
    "TrimmedMean",
    "Median",
    "Bulyan",
    "RFA",
    "FLTrust",
    "CenteredClipping",
    "DnC",
    "Bucketing",
    "SignGuard",
    "TriGuardFL",
    "LASA",
    "Auror",
    "FoolsGold",
    "NormClipping",
    "CRFL",
    "DeepSight",
    "FLAME",
}


def test_canonical_preset_path_resolves():
    resolved = resolve_config_path("configs/FedSGD_MNIST_Lenet.yaml")
    assert resolved == preset_path("FedSGD", "MNIST", "lenet").resolve()


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


def test_shared_attack_catalog_matches_curated_defaults():
    payload = load_yaml_mapping(attacks_catalog_path())
    attack_names = {item["attack"] for item in payload["attacks"]}
    assert attack_names == EXPECTED_SHARED_ATTACKS
    assert attack_names <= KNOWN_ATTACKS


def test_shared_defense_catalog_matches_curated_defaults():
    payload = load_yaml_mapping(defenses_catalog_path())
    defense_names = {item["defense"] for item in payload["defenses"]}
    assert defense_names == EXPECTED_SHARED_DEFENSES
    assert defense_names <= set(all_aggregators)
