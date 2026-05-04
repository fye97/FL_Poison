# CARAT NeurIPS Follow-up Experiment Plan

This document records the compact follow-up experiment plan for the CARAT NeurIPS submission. The goal is to address the most important reviewer risks under limited local compute, without expanding the empirical scope beyond what can be finished reliably.

## Objective

The follow-up experiments target two reviewer concerns:

1. CARAT looks like a combination of several engineering components, so the paper needs focused ablations showing which components matter.
2. Some public baselines in the main table were previously taken from compatible legacy logs, so the paper needs a matched-seed rerun protocol for the displayed public baselines.

The plan intentionally does not add new datasets, backdoor attacks, or a true CARAT-aware adaptive attack. Those additions would require new data setup, new baselines, and substantial debugging, and are not the highest-return use of a single local GPU.

## Scope

Run only CIFAR-100 / ResNet18 / FedAvg experiments under the current paper protocol:

- Dataset: `CIFAR100`
- Model: `resnet18`
- Clients: `20`
- Malicious fraction: `0.2`
- Epochs: `200`
- Learning rate: `0.05`
- Seeds: `42, 43, 44`, implemented as `seed=42` and `experiment_id=0,1,2`
- Attacks for matched main-table baselines: `ALIE`, `FangAttack`, `MinMax`, `MinSum`
- Main-table defenses: `Mean`, `NormClipping`, `MultiKrum`, `FLTrust`, `FLDetector`, `CARAT`
- Excluded defenses: `TrimmedMean`, unpublished `TriGuardFL`

## Files Added For This Follow-up

These files should be copied to the new machine together with the repository:

- `exps/specs/CARAT/paper_neurips_matched_baselines_public.yaml`
- `exps/specs/CARAT/paper_neurips_sensitivity_fang_alpha05.yaml`
- `exps/run_neurips_followup_local.sh`
- `docs/carat_neurips_followup_experiment_plan.md`

Existing spec reused:

- `exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml`

## Experiment Phases

### Phase 1: Mechanism Ablation

Spec:

```bash
exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml
```

Run only task ids `6-14`.

These ids cover the real missing CARAT variants:

- `rank0`: CARAT without coordinate-rank anomaly penalty
- `prior0`: CARAT without temporal prior
- `pool400`: CARAT with smaller reference probe pool

Do not rerun ids `0-2`; they correspond to full CARAT with a config-name difference and are semantically already covered by the existing protocol CARAT runs. Do not rerun ids `3-5`; those are the existing `T=1` variant already exported in the result library.

Expected new tasks: `9`.

### Phase 2: Task Count And Reference Sensitivity

Spec:

```bash
exps/specs/CARAT/paper_neurips_sensitivity_fang_alpha05.yaml
```

Run all ids.

This adds a compact sensitivity table around the representative hard setting `alpha=0.5 + FangAttack`:

- `T=4`
- `T=16`
- `probe_pool=1200`

Together with existing `T=1`, full CARAT `T=8`, and `probe_pool=400`, this supports a clean appendix sensitivity table without exploding compute.

Expected new tasks: `9`.

### Phase 3: Matched Public Baselines

Spec:

```bash
exps/specs/CARAT/paper_neurips_matched_baselines_public.yaml
```

Run all ids with exact matching.

The spec contains six displayed public defenses, but the existing result library already has the matched FLTrust and CARAT runs. The incremental runner should skip those and run only:

- `Mean`: `24` tasks
- `NormClipping`: `24` tasks
- `MultiKrum`: `24` tasks
- `FLDetector`: `24` tasks

Expected new tasks: `96`.

Total expected new tasks: `114`.

## Estimated Runtime

Approximate runtime on one RTX 4090-class GPU:

- CARAT ablation and sensitivity variants: about `25-35 minutes` per task.
- Public baselines: about `12-15 minutes` per task.
- Total expected runtime: about `30-40 GPU hours`, depending on GPU availability and dataloader overhead.

Do not run multiple jobs concurrently on a single 24GB GPU unless memory has been checked carefully. The recommended queue uses one job at a time.

## Machine Setup Checklist

Before running on the new machine:

1. Sync the `FL_Poison` repository including the four follow-up files listed above.
2. Ensure the Python environment works:

```bash
cd /path/to/FL_Poison
.venv/bin/python -m flpoison --help >/dev/null
```

3. Ensure CIFAR-100 data exists:

```bash
ls data/cifar-100-python
```

4. Ensure the result library exists or create it:

```bash
mkdir -p /path/to/Poisoning_Resilient_Federated_Learning_Playground
```

5. If continuing from an existing result library, copy the existing playground results to the new machine first. This is important because the incremental runner uses the library to skip completed experiments.

## Recommended Run Command

Use the queue script. It waits for a sufficiently free GPU before each task, runs one task, exports it to the playground result library, and then moves to the next task.

```bash
cd /path/to/FL_Poison

PLAYGROUND_ROOT=/path/to/Poisoning_Resilient_Federated_Learning_Playground \
CUDA=0 \
MIN_FREE_MB=22000 \
MAX_UTIL=20 \
POLL_SECONDS=300 \
bash exps/run_neurips_followup_local.sh
```

If the new machine has a larger GPU and no competing jobs, the same command is still safe. If it has less than 24GB memory, lower `MIN_FREE_MB` cautiously only after checking that CARAT runs fit.

## Running In tmux

Recommended:

```bash
cd /path/to/FL_Poison

tmux new-session -s carat_followup

PLAYGROUND_ROOT=/path/to/Poisoning_Resilient_Federated_Learning_Playground \
CUDA=0 \
MIN_FREE_MB=22000 \
MAX_UTIL=20 \
POLL_SECONDS=300 \
bash exps/run_neurips_followup_local.sh
```

Detach with `Ctrl-b d`.

Monitor:

```bash
tail -f logs/local_followup/neurips_followup_*.log
nvidia-smi
```

## Pre-run Missing Checks

Run these before starting the queue:

```bash
cd /path/to/FL_Poison

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml \
  --ids 6-14 \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_sensitivity_fang_alpha05.yaml \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_matched_baselines_public.yaml \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary
```

Expected before any new follow-up runs:

- Ablation ids `6-14`: `9` missing
- Sensitivity: `9` missing
- Matched baselines: `96` missing out of `144`, because FLTrust and CARAT should already be present

## Post-run Completion Checks

After the queue finishes, all three checks should report `missing=0` for the requested ids:

```bash
cd /path/to/FL_Poison

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_ablation_fang_alpha05.yaml \
  --ids 6-14 \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_sensitivity_fang_alpha05.yaml \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary

.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_matched_baselines_public.yaml \
  --playground-root /path/to/Poisoning_Resilient_Federated_Learning_Playground \
  --match-config exact \
  --format summary
```

Also confirm the queue log ends with:

```text
neurips_followup_queue_complete=...
```

## Result Handling

The incremental runner exports completed local CSV logs to the playground result library after each task. The exported results are intended to be appended to the existing result folder, not to replace previous results.

Main local outputs:

```bash
logs/local_runs/
logs/local_followup/
```

Main exported result library:

```bash
/path/to/Poisoning_Resilient_Federated_Learning_Playground
```

If the new machine uses a different playground root, set `PLAYGROUND_ROOT` explicitly in every command.

## Paper Update After Results Finish

Once all follow-up runs are complete, update the paper in this order:

1. Regenerate the main Table 2 using the matched-seed public baselines, not the legacy baseline logs.
2. Add an appendix ablation table for full CARAT, `T=1`, w/o rank penalty, w/o temporal prior, and probe-pool variants.
3. Add a compact reference/task sensitivity table with `T=1/4/8/16` and probe pool `400/800/1200`.
4. Revise the experiment text to say the displayed public baselines are run under a matched local protocol with three seeds.
5. Do not mention TrimmedMean in the paper, tables, captions, artifact summaries, or related experiment text.

## Current Machine Note

The current machine has a waiting tmux queue:

```bash
tmux attach -t carat_followup_0504_123406
```

Because another user's GPU job is running at high utilization, this queue has not started any experiment. If the follow-up will run on a new machine, stop the current waiting queue to avoid duplicate runs:

```bash
tmux kill-session -t carat_followup_0504_123406
```

