## CARAT Experiment Suite

This directory defines the experiment matrix for the CARAT paper.

Recommended execution order:

1. `CARAT/smoke_cifar100`
2. `CARAT/clean_reference`
3. `CARAT/pilot_untargeted`
4. `CARAT/main_untargeted`
5. `CARAT/backdoor`
6. `CARAT/ablations`

What each spec is for:

- `smoke_cifar100`: quick end-to-end sanity check with evaluation enabled.
- `clean_reference`: clean-utility baseline under iid and non-iid splits with `num_adv=0`.
- `pilot_untargeted`: one-seed pilot on representative untargeted attacks before launching the full sweep.
- `main_untargeted`: main poisoning benchmark for the paper.
- `backdoor`: clean accuracy plus ASR benchmark under classic and durable backdoor attacks.
- `ablations`: one-factor-at-a-time CARAT ablations on the hardest untargeted setting.

Design choices:

- The paper should keep CIFAR100 + ResNet-18 as the primary benchmark because CARAT is class-aware and benefits most from a many-class setting.
- Clean runs are separated from attack runs so `NoAttack` is not redundantly crossed with nonzero `num_adv`.
- The untargeted benchmark compares CARAT against a compact but diverse baseline set:
  `Mean`, `FLTrust`, `MultiKrum`, `CARAT`.
- The backdoor benchmark swaps in backdoor-oriented baselines:
  `Mean`, `FLTrust`, `FLAME`, `DeepSight`, `CARAT`.
- All paper-facing specs set `evaluate: true`.

Expected task counts:

- `smoke_cifar100`: 2 tasks
- `clean_reference`: 36 tasks
- `pilot_untargeted`: 12 tasks
- `main_untargeted`: 216 tasks
- `backdoor`: 60 tasks
- `ablations`: 33 tasks

Suggested commands:

```bash
python exps/launch.py plan CARAT/smoke_cifar100
python exps/launch.py plan CARAT/main_untargeted
python exps/launch.py plan CARAT/backdoor
python exps/launch.py plan CARAT/ablations
```

```bash
./exps/run_local.sh CARAT/smoke_cifar100 --ids 0-1 --jobs 1
./exps/run_cc.sh CARAT/main_untargeted --chunk-size 24
./exps/run_cc.sh CARAT/backdoor --chunk-size 16
./exps/run_cc.sh CARAT/ablations --chunk-size 16
```

Result-handling note:

- `metrics.csv` directly stores `train_*` and `eval_*`.
- ASR is not written into `metrics.csv`; it must be parsed from `logs/run_logs/.../*.log` or from `logs/perf_logs/.../*.json` when runtime profiling is enabled.
