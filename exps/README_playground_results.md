# Playground Result Workflow

The shared result library is:

```bash
/home/cnlab-pu/Projects/Poisoning_Resilient_Federated_Learning_Playground
```

Existing files use the legacy text-log format:

```text
FedAvg/CIFAR100_resnet18/non-iid/CIFAR100_resnet18_non-iid_ALIE_MultiKrum_200_20_0.05_FedAvg_adv0.2_seed42_alpha0.1_cfgFedAvg_CIFAR100_config_exp0.txt
```

Use `exps/playground_results.py` to avoid rerunning completed experiments.

Check which CARAT paper-main tasks are still missing:

```bash
.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_main_cifar100_carat_200.yaml \
  --match-config any \
  --format summary
```

Get only missing task ids:

```bash
IDS=$(.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_main_cifar100_carat_200.yaml \
  --match-config any \
  --format ids)
```

Run only the missing tasks:

```bash
.venv/bin/python exps/launch.py local \
  exps/specs/CARAT/paper_main_cifar100_carat_200.yaml \
  --ids "$IDS" \
  --resume \
  --cuda 0 \
  --jobs 1
```

Export completed CSV artifacts back to the playground text-log format:

```bash
.venv/bin/python exps/playground_results.py export \
  --spec exps/specs/CARAT/paper_main_cifar100_carat_200.yaml \
  --ids "$IDS"
```

The exporter divides CSV losses by `batch_size` by default. This matches the legacy playground scale, where CIFAR-100 initial cross-entropy appears around `0.072` instead of raw CE around `4.6`.
