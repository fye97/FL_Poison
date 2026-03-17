# Configs

`configs/` now separates three responsibilities:

- `presets/`: runnable experiment defaults for a specific algorithm + dataset pair.
- `catalog/attacks.yaml` and `catalog/defenses.yaml`: shared global attack/defense registries reused by presets.
- `catalog/datasets.yaml`: dataset metadata consumed during preprocessing.
