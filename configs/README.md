# Configs

`configs/` keeps runnable presets at the top level and shared metadata under `configs/presets/`:

- `presets/attacks.yaml` and `presets/defenses.yaml`: shared global attack/defense registries.
- `presets/datasets.yaml`: dataset metadata consumed during preprocessing.
- `FedAvg_CIFAR10_vgg19.yaml`-style files: runnable experiment presets whose names encode algorithm, dataset, and default model.
