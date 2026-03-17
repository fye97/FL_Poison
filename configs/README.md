# Configs

`configs/` now separates three responsibilities:

- `presets/`: runnable experiment defaults for a specific algorithm + dataset pair.
- `catalog/attacks` and `catalog/defenses`: shared attack/defense registries reused by presets.
- `catalog/datasets.yaml`: dataset metadata consumed during preprocessing.
