# CARAT 实验运行说明

这个目录定义了与 CARAT 论文当前实验设计对应的 spec 矩阵。目标不是复用 TriGuardFL 的 omnibus 逻辑，而是围绕 CARAT 的核心主张来组织实验：

- 主 benchmark 使用 `CIFAR100 + ResNet18`
- main text 分开写 `clean utility`、`untargeted poisoning`、`backdoor`
- appendix 单独承接 supplementary benchmark 和 ablation
- `FLTrust` 与 `CARAT` 在协议 config 中使用匹配的 trusted-data budget

命令里使用的 spec id 是：

```bash
CARAT/smoke_cifar100
CARAT/pilot_untargeted
CARAT/clean_reference
CARAT/main_untargeted
CARAT/backdoor
CARAT/appendix_tinyimagenet
CARAT/appendix_chmnist
CARAT/appendix_benchmarks
CARAT/ablations
```

## 推荐执行顺序

1. `CARAT/smoke_cifar100`
2. `CARAT/pilot_untargeted`
3. `CARAT/clean_reference`
4. `CARAT/main_untargeted`
5. `CARAT/backdoor`
6. `CARAT/appendix_tinyimagenet`
7. `CARAT/appendix_chmnist`
8. `CARAT/ablations`

各 spec 的用途：

- `smoke_cifar100`: 5 epoch 快速连通性检查，只跑 `ALIE` 对 `Mean/CARAT`
- `clean_reference`: 主 benchmark 的 clean utility，对比 `Mean/TrimmedMean/MultiKrum/FLTrust/CARAT`
- `pilot_untargeted`: 一轮小规模 preflight，用来先检查主 untargeted 矩阵是否正常展开和落盘
- `main_untargeted`: 主文 untargeted poisoning benchmark，攻击是 `ALIE/MinMax/MinSum`
- `backdoor`: 主文 backdoor benchmark，攻击是 `ModelReplacement/DBA/Neurotoxin`
- `appendix_tinyimagenet`: appendix supplementary benchmark 的 `TinyImageNet` 部分
- `appendix_chmnist`: appendix supplementary benchmark 的 `CHMNIST` 部分
- `appendix_benchmarks`: 兼容旧工作流的合并版 appendix spec，不建议再作为 Compute Canada 主入口
- `ablations`: CARAT 机制消融，覆盖 hidden-task 数、reference budget、rank/prior 权重和 clipping 强度

## 先看矩阵

先确认 launcher 能识别这些 spec：

```bash
python exps/launch.py list | rg CARAT
```

查看某个 spec 会展开成多少个 task：

```bash
python exps/launch.py plan CARAT/smoke_cifar100
python exps/launch.py plan CARAT/main_untargeted
python exps/launch.py plan CARAT/backdoor
python exps/launch.py plan CARAT/appendix_tinyimagenet
python exps/launch.py plan CARAT/appendix_chmnist
python exps/launch.py plan CARAT/ablations
```

当前矩阵规模：

- `smoke_cifar100`: 2 tasks
- `pilot_untargeted`: 20 tasks
- `clean_reference`: 45 tasks
- `main_untargeted`: 270 tasks
- `backdoor`: 135 tasks
- `appendix_tinyimagenet`: 36 tasks
- `appendix_chmnist`: 36 tasks
- `appendix_benchmarks`: 72 tasks
- `ablations`: 33 tasks

## 协议 config

主实验和 supplementary benchmark 都显式绑定了协议 config：

- `configs/CARAT/FedAvg_CIFAR100_Resnet18_protocol.yaml`
- `configs/CARAT/FedAvg_TinyImageNet_Resnet18_protocol.yaml`
- `configs/CARAT/FedAvg_CHMNIST_Resnet18_protocol.yaml`

这些协议 config 做了两件事：

- 固定 `FedAvg + ResNet18 + 20 clients + 300 rounds + lr 0.05`
- 显式把 `FLTrust.num_sample` 和 `CARAT.num_probe_pool` 设成匹配预算，而不是继续沿用共享 preset 里的不公平默认值

当前预算设计：

- `CIFAR100`: `FLTrust=800`, `CARAT probe pool=800`
- `TinyImageNet`: `FLTrust=1000`, `CARAT probe pool=1000`
- `CHMNIST`: `FLTrust=256`, `CARAT probe pool=256`

其中 `TinyImageNet` 的 `probe_samples_per_class=2`，`CHMNIST` 的 `probe_samples_per_class=8`，是为了避免所有数据集机械复用同一个 hidden-task 大小。

## 本地运行

先跑 smoke test：

```bash
./exps/run_local.sh CARAT/smoke_cifar100 --jobs 1 --gpu-tokens 1
```

如果只想跑完整矩阵中的一小段 task：

```bash
./exps/run_local.sh CARAT/main_untargeted --ids 0-3 --jobs 1 --gpu-tokens 1
./exps/run_local.sh CARAT/backdoor --ids 0-3 --jobs 1 --gpu-tokens 1
./exps/run_local.sh CARAT/appendix_tinyimagenet --ids 0-3 --jobs 1 --gpu-tokens 1
./exps/run_local.sh CARAT/appendix_chmnist --ids 0-3 --jobs 1 --gpu-tokens 1
```

失败后补跑未完成 task：

```bash
./exps/run_local.sh CARAT/main_untargeted --ids 0-47 --jobs 1 --gpu-tokens 1 --resume
```

说明：

- `--jobs` 控制本地并发 worker 数
- `--gpu-tokens` 控制同时真正占 GPU 的 task 数
- 当前这些 spec 都显式设置了 `evaluate: true`

## Compute Canada 运行

在 Compute Canada 上，直接按下面的顺序执行即可：

```bash
git clone <your-repo-url> FL_Poison
cd FL_Poison
module purge
module load python/3.12 cuda/12.2
python -m pip install --user uv
uv sync --frozen
source .venv/bin/activate
./exps/run_cc_carat.sh --dry-run
./exps/run_cc_carat.sh
```

`run_cc_carat.sh` 只做并行提交。
它会一次性把整套 CARAT spec 都提交到 Slurm，各个 array job 独立排队、独立运行。

推荐先提交一条 smoke / preflight：

```bash
./exps/run_cc.sh CARAT/smoke_cifar100
./exps/run_cc.sh CARAT/pilot_untargeted
```

确认 smoke 和 pilot 正常后，再提交整套：

```bash
./exps/run_cc_carat.sh
```

如果你想按 spec 分批提交，也可以单独提：

```bash
./exps/run_cc.sh CARAT/clean_reference
./exps/run_cc.sh CARAT/main_untargeted
./exps/run_cc.sh CARAT/backdoor
./exps/run_cc.sh CARAT/appendix_tinyimagenet
./exps/run_cc.sh CARAT/appendix_chmnist
./exps/run_cc.sh CARAT/ablations
```

断点补跑：

```bash
./exps/run_cc_carat.sh --resume
```

这个脚本会按下面顺序依次执行 `sbatch` 提交：

1. `CARAT/smoke_cifar100`
2. `CARAT/pilot_untargeted`
3. `CARAT/clean_reference`
4. `CARAT/main_untargeted`
5. `CARAT/backdoor`
6. `CARAT/appendix_tinyimagenet`
7. `CARAT/appendix_chmnist`
8. `CARAT/ablations`

按当前 spec 默认的 `array_parallel` 来看，如果你把整套都提交出去，理论上的最大并发大约是 `24` 个 GPU task：

- `smoke_cifar100`: 1
- `pilot_untargeted`: 2
- `clean_reference`: 4
- `main_untargeted`: 4
- `backdoor`: 4
- `appendix_tinyimagenet`: 2
- `appendix_chmnist`: 4
- `ablations`: 3

如果你不知道自己能拿到多少并发，就先用这套默认值。它已经偏激进，通常足够把结果尽快铺开。
实际能同时跑到多少，还取决于账号 fair-share、当时排队情况和集群资源。

如果只提交矩阵中的一段：

```bash
./exps/run_cc.sh CARAT/main_untargeted --start-id 0 --end-id 47 --chunk-size 16
```

## 结果保存

本地默认结果根目录：

```text
logs/local_runs/
```

本地 task 级 runner 日志：

```text
logs/local_array/
```

详细训练日志：

```text
logs/run_logs/
```

每个 task 的结果目录大致是：

```text
logs/local_runs/
  FedAvg/
    CIFAR100_resnet18/
      <distribution>/
        <attack>__<defense>/
          ep<epochs>_clients<num_clients>_lr<learning_rate>_adv<num_adv>_seed<effective_seed>_exp<experiment_id>[_alpha<dirichlet_alpha>]_<...>/
            metrics_exp<experiment_id>.csv
            jobmeta.txt
            task.complete
```

结果说明：

- `metrics_exp*.csv` 保存每轮 `train_*` 和 `eval_*`
- 详细训练过程写在 `logs/run_logs/.../*.log`
- backdoor 的 ASR 仍然需要从 `run_logs` 或后处理脚本解析

## 设计说明

- 主文的 clean utility、untargeted、backdoor 三套实验现在完全分开
- `main_untargeted` 的对照组是 `Mean/TrimmedMean/MultiKrum/FLTrust/CARAT`
- `backdoor` 的对照组是 `Mean/FLTrust/FLAME/DeepSight/CARAT`
- supplementary benchmark 在提交层面拆成 `appendix_tinyimagenet` 和 `appendix_chmnist`
- `ablations` 不再围绕 `beta` 做一组孤立超参扫描，而是改成更贴近论文机制的 ablation 轴：
  - hidden-task count
  - reference budget
  - rank weight
  - temporal prior
  - clipping strength

## 建议流程

1. 先 `python exps/launch.py plan <spec>` 看矩阵大小
2. 先本地跑 `smoke_cifar100` 或小范围 `--ids`
3. 确认输出目录、日志和 `metrics_exp*.csv` 正常
4. 再提交 `clean_reference -> main_untargeted -> backdoor`
5. supplementary benchmark 分成 `appendix_tinyimagenet` 和 `appendix_chmnist` 两套单独提
6. 最后补 `ablations`

## 结果汇总

新增的汇总脚本是：

```bash
python exps/summarize_carat.py --result-root logs/local_runs
```

默认会扫描 `logs/local_runs` 下发现的全部 CARAT spec，并在下面生成汇总 CSV：

```text
logs/reports/carat/
  all_runs.csv
  clean_summary.csv
  untargeted_summary.csv
  backdoor_summary.csv
  runtime_summary.csv
```

脚本会优先读取 `metrics_exp*.csv` 中的新字段：

- `eval_acc`
- `eval_loss`
- `macro_acc`
- `worst_class_acc`
- `asr`
- `asr_loss`
- `round_time_sec`

如果旧结果没有这些字段，它会尽量从 `perf_logs/*.json` 或 `run_logs/*.log` 回填 `ASR` 和平均 `round time`。
