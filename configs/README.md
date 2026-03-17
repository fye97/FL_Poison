# Configs

`configs/` is now flat:

- `attacks.yaml` and `defenses.yaml`: shared global attack/defense registries.
- `datasets.yaml`: dataset metadata consumed during preprocessing.
- `FedAvg_CIFAR10_Resnet18.yaml`-style files: runnable experiment presets whose names encode algorithm, dataset, and default model.
