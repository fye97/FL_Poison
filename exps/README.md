# Experiments

`exps/` now has one job:

- `specs/*.yaml`: define experiment matrices
- `launch.py`: `list`, `plan`, `local`, `cc`, `worker`
- `slurm_array_entry.sh`: generic Compute Canada array entrypoint

Examples:

```bash
python exps/launch.py list
python exps/launch.py plan cifar10
python exps/launch.py local cifar10 --ids 0-7 --jobs 1
python exps/launch.py cc cifar10 --chunk-size 32
```

The design intent is strict separation:

- Experiment content lives in `specs/*.yaml`
- Execution framework lives in `launch.py`
- Backend choice is only `local` vs `cc`

Typical spec shape:

```yaml
name: cifar10
description: Compare attacks and defenses on CIFAR10.

scenarios:
  - algorithm: FedAvg
    dataset: CIFAR10

matrix:
  distributions:
    - type: iid
    - type: non-iid
      dirichlet_alpha: 0.1
    - type: non-iid
      dirichlet_alpha: 0.5
    - type: non-iid
      dirichlet_alpha: 1
  num_advs:
    - 0.2
    - 0.3
    - 0.4
  models:
    - vgg19
  epochs:
    - 200
  num_clients:
    - 20
  learning_rates:
    - 0.05
  seeds:
    - 42
  attacks:
    - NoAttack
    - FangAttack
  defenses:
    - Mean
    - FLTrust

repeats:
  count: 5
  start: 0
```

Notes:

- Put your experiment design in `matrix`
- `matrix.distributions` is where `dirichlet_alpha` lives
- `matrix.num_advs` is the attacker-ratio sweep, for example `0.2/0.3/0.4`
- If you omit a field like `models` or `epochs`, the launcher falls back to the scenario config file
