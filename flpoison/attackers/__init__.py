import importlib
from pathlib import Path

from flpoison.utils.global_utils import Register

attacker_registry = Register()
_OPTIONAL_ATTACKERS = {
    "edgecase": {
        "class_name": "EdgeCase",
        "dependencies": {"rarfile"},
        "category": "attacker",
        "attributes": ("data_poisoning", "model_poisoning", "non_omniscient"),
    },
}


def _missing_optional_attacker(class_name, module_name, exc, *, category, attributes):
    class MissingOptionalAttacker:
        _category = category
        _attributes = attributes

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"{class_name} requires optional dependency `{exc.name}`; "
                f"install it to use `flpoison/attackers/{module_name}.py`."
            ) from exc

    MissingOptionalAttacker.__name__ = class_name
    return MissingOptionalAttacker


def _import_attacker_modules():
    current_dir = Path(__file__).resolve().parent
    package_name = __name__

    for path in sorted(current_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module_name = path.stem
        try:
            importlib.import_module(f".{module_name}", package=package_name)
        except ModuleNotFoundError as exc:
            optional = _OPTIONAL_ATTACKERS.get(module_name)
            if optional and exc.name in optional["dependencies"]:
                attacker_registry[optional["class_name"]] = _missing_optional_attacker(
                    optional["class_name"],
                    module_name,
                    exc,
                    category=optional["category"],
                    attributes=optional["attributes"],
                )
                continue
            raise


_import_attacker_modules()

# pure data poisoning attacks
data_poisoning_attacks = [name for name in attacker_registry.keys(
) if "data_poisoning" in attacker_registry[name]._attributes]

# hybrid attackers with data poisoning and model poisoning capabilities simultaneously
hybrid_attacks = [name for name in attacker_registry.keys() if all(
    attr in attacker_registry[name]._attributes for attr in ["model_poisoning", "data_poisoning"])]

# get pure model poisoning attacks
model_poisoning_attacks = [name for name in attacker_registry.keys(
) if "data_poisoning" not in attacker_registry[name]._attributes]


def get_attacker_handler(name):
    assert name != "NoAttack", f"NoAttack should not specify num_adv argument"
    return attacker_registry[name]
