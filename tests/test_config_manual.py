from pathlib import Path

import yaml

from flpoison.aggregators import all_aggregators
from flpoison.fl.configuration import KNOWN_ATTACKS
from flpoison.fl.models import all_models
from flpoison.utils.config_utils import load_dataset_catalog, list_preset_files


DOC_PATH = Path("docs/config-manual.md")
DOC_RUNTIME_ONLY_FIELDS = {
    "output",
    "attacks",
    "defenses",
    "attack_params",
    "defense_params",
    "log_color",
    "eval_batch_size",
    "eval_interval",
    "gpu_sample_interval_ms",
    "torch_profile",
    "torch_profile_wait",
    "torch_profile_warmup",
    "torch_profile_active",
    "torch_profile_repeat",
    "torch_profile_record_shapes",
    "torch_profile_memory",
    "torch_profile_with_stack",
    "aug",
    "partition_visualization",
}
DOC_DATASET_METADATA_FIELDS = {
    "num_training_sample",
    "num_channels",
    "num_classes",
    "mean",
    "std",
    "num_dims",
    "num_features",
}
DOC_ENUM_VALUES = {
    "FedSGD",
    "FedAvg",
    "FedOpt",
    "SGD",
    "Adam",
    "MultiStepLR",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "iid",
    "class-imbalanced_iid",
    "non-iid",
    "pat",
    "imbalanced_pat",
}


def _load_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _assert_all_mentioned(doc: str, names: set[str], *, label: str) -> None:
    missing = sorted(name for name in names if name not in doc)
    assert not missing, f"{label} missing from docs/config-manual.md: {missing}"


def test_config_manual_mentions_top_level_fields():
    preset_fields = set()
    for preset in list_preset_files():
        payload = yaml.safe_load(Path(preset).read_text(encoding="utf-8"))
        preset_fields.update(payload.keys())

    expected_fields = preset_fields | DOC_RUNTIME_ONLY_FIELDS | DOC_DATASET_METADATA_FIELDS
    _assert_all_mentioned(_load_doc(), expected_fields, label="config fields")


def test_config_manual_mentions_registered_components_and_datasets():
    doc = _load_doc()
    _assert_all_mentioned(doc, set(KNOWN_ATTACKS), label="attacks")
    _assert_all_mentioned(doc, set(all_aggregators), label="defenses")
    _assert_all_mentioned(doc, set(all_models), label="models")
    _assert_all_mentioned(doc, set(load_dataset_catalog().keys()), label="datasets")
    _assert_all_mentioned(doc, DOC_ENUM_VALUES, label="enum values")
