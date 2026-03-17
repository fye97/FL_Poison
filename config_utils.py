from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def configs_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs"


def presets_dir(root: Path | None = None) -> Path:
    return configs_dir(root) / "presets"


def catalog_dir(root: Path | None = None) -> Path:
    return configs_dir(root) / "catalog"


def dataset_catalog_path(root: Path | None = None) -> Path:
    return catalog_dir(root) / "datasets.yaml"


def preset_path(algorithm: str, dataset: str, root: Path | None = None) -> Path:
    return presets_dir(root) / algorithm / f"{dataset}.yaml"


def preset_relpath(algorithm: str, dataset: str) -> Path:
    return Path("configs") / "presets" / algorithm / f"{dataset}.yaml"


def catalog_path(kind: str, name: str, root: Path | None = None) -> Path:
    return catalog_dir(root) / kind / f"{name}.yaml"


def list_preset_files(root: Path | None = None) -> list[Path]:
    return sorted(p.resolve() for p in presets_dir(root).glob("*/*.yaml"))


def resolve_config_path(path_like: str | Path, *, root: Path | None = None) -> Path:
    root = (root or repo_root()).resolve()
    raw = Path(path_like)

    direct_candidates: list[Path] = [raw] if raw.is_absolute() else [root / raw]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    fallback = direct_candidates[0].resolve()
    raise FileNotFoundError(f"config not found: {fallback}")


def resolve_preset_for_scenario(
    algorithm: str,
    dataset: str,
    *,
    root: Path | None = None,
    allow_fallback: bool = True,
) -> Path:
    root = (root or repo_root()).resolve()
    candidate = preset_path(algorithm, dataset, root=root)
    if candidate.exists() or not allow_fallback:
        return candidate.resolve()

    fallback_datasets: list[str] = []
    if dataset in {"MNIST", "FashionMNIST", "EMNIST"}:
        fallback_datasets.append("MNIST")
    else:
        fallback_datasets.append("CIFAR10")

    for fallback_dataset in fallback_datasets:
        fallback = preset_path(algorithm, fallback_dataset, root=root)
        if fallback.exists():
            return fallback.resolve()

    algorithm_dir = presets_dir(root) / algorithm
    for item in sorted(algorithm_dir.glob("*.yaml")):
        if item.exists():
            return item.resolve()

    raise FileNotFoundError(
        f"no usable config found for algorithm={algorithm} dataset={dataset}"
    )


def load_yaml_mapping(path_like: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(path_like, root=root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config file: {path}")
    return raw


def load_experiment_config(path_like: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    path = resolve_config_path(path_like, root=root)
    raw = load_yaml_mapping(path, root=root)

    catalogs = raw.pop("catalogs", {}) or {}
    if not isinstance(catalogs, dict):
        raise ValueError(f"`catalogs` must be a mapping in {path}")

    attack_catalog = raw.pop("attack_catalog", None) or catalogs.get("attacks")
    defense_catalog = raw.pop("defense_catalog", None) or catalogs.get("defenses")

    if "attacks" not in raw:
        if not attack_catalog:
            raise ValueError(f"missing `attacks` or `catalogs.attacks` in {path}")
        raw["attacks"] = load_catalog_entries("attacks", attack_catalog, root=root)
    if "defenses" not in raw:
        if not defense_catalog:
            raise ValueError(f"missing `defenses` or `catalogs.defenses` in {path}")
        raw["defenses"] = load_catalog_entries("defenses", defense_catalog, root=root)

    return raw


def load_catalog_entries(kind: str, name: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    path = catalog_path(kind, name, root=root)
    raw = load_yaml_mapping(path, root=root)
    entries = raw.get(kind)
    if not isinstance(entries, list):
        raise ValueError(f"`{kind}` must be a list in {path}")
    return entries


def load_dataset_catalog(*, root: Path | None = None) -> dict[str, dict[str, Any]]:
    path = dataset_catalog_path(root)
    raw = load_yaml_mapping(path, root=root)
    validate_dataset_catalog(raw, source=path)
    return raw


def validate_dataset_catalog(catalog: dict[str, Any], *, source: str | Path) -> None:
    for dataset_name, values in catalog.items():
        if not isinstance(values, dict):
            raise ValueError(f"dataset entry `{dataset_name}` must be a mapping in {source}")
        for stat_key in ("mean", "std"):
            if stat_key not in values:
                continue
            stat_value = values[stat_key]
            if not isinstance(stat_value, list):
                raise ValueError(
                    f"{dataset_name}.{stat_key} must be a YAML list in {source}; tuple-like strings are no longer supported"
                )
            if not all(isinstance(item, (int, float)) for item in stat_value):
                raise ValueError(f"{dataset_name}.{stat_key} must contain only numbers in {source}")


def validate_experiment_config(
    config: dict[str, Any],
    *,
    source: str | Path,
    known_attacks: Iterable[str] | None = None,
    known_defenses: Iterable[str] | None = None,
) -> None:
    attacks = config.get("attacks")
    defenses = config.get("defenses")

    validate_named_entries(attacks, field="attack", source=source, allowed_names=known_attacks)
    validate_named_entries(defenses, field="defense", source=source, allowed_names=known_defenses)

    active_attack = config.get("attack")
    if active_attack and active_attack not in {item["attack"] for item in attacks or []}:
        raise ValueError(f"default attack `{active_attack}` is not declared in {source}")

    active_defense = config.get("defense")
    if active_defense and active_defense not in {item["defense"] for item in defenses or []}:
        raise ValueError(f"default defense `{active_defense}` is not declared in {source}")


def validate_named_entries(
    entries: Any,
    *,
    field: str,
    source: str | Path,
    allowed_names: Iterable[str] | None = None,
) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"`{field}s` must be a list in {source}")

    allowed = set(allowed_names) if allowed_names is not None else None
    seen: set[str] = set()

    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"every `{field}` entry must be a mapping in {source}")
        name = item.get(field)
        if not isinstance(name, str) or not name:
            raise ValueError(f"every `{field}` entry must have a non-empty `{field}` field in {source}")
        if name in seen:
            raise ValueError(f"duplicate `{field}` entry `{name}` in {source}")
        seen.add(name)

        if allowed is not None and name not in allowed:
            raise ValueError(f"unknown `{field}` entry `{name}` in {source}")

        params_key = f"{field}_params"
        if params_key in item and item[params_key] is not None and not isinstance(item[params_key], dict):
            raise ValueError(f"`{params_key}` for `{name}` must be a mapping in {source}")
