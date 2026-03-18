import importlib
from pathlib import Path

from flpoison.utils.global_utils import Register

aggregator_registry = Register()
_OPTIONAL_AGGREGATORS = {
    "deepsight": {"class_name": "DeepSight", "dependencies": {"hdbscan"}},
    "flame": {"class_name": "FLAME", "dependencies": {"hdbscan"}},
}


def _missing_optional_aggregator(class_name, module_name, exc):
    class MissingOptionalAggregator:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"{class_name} requires optional dependency `{exc.name}`; "
                f"install it to use `flpoison/aggregators/{module_name}.py`."
            ) from exc

    MissingOptionalAggregator.__name__ = class_name
    return MissingOptionalAggregator


def _import_aggregator_modules():
    current_dir = Path(__file__).resolve().parent
    package_name = __name__

    for path in sorted(current_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module_name = path.stem
        try:
            importlib.import_module(f".{module_name}", package=package_name)
        except ModuleNotFoundError as exc:
            optional = _OPTIONAL_AGGREGATORS.get(module_name)
            if optional and exc.name in optional["dependencies"]:
                aggregator_registry[optional["class_name"]] = _missing_optional_aggregator(
                    optional["class_name"], module_name, exc
                )
                continue
            raise


_import_aggregator_modules()
all_aggregators = list(aggregator_registry.keys())


def get_aggregator(name):
    return aggregator_registry[name]
