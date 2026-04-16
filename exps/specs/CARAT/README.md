# CARAT 实验运行说明

这个目录定义了 CARAT 论文对应的实验矩阵。

命令里使用的 spec id 是：

```bash
CARAT/smoke_cifar100
CARAT/clean_reference
CARAT/pilot_untargeted
CARAT/main_untargeted
CARAT/backdoor
CARAT/ablations
```

## 推荐执行顺序

1. `CARAT/smoke_cifar100`
2. `CARAT/clean_reference`
3. `CARAT/pilot_untargeted`
4. `CARAT/main_untargeted`
5. `CARAT/backdoor`
6. `CARAT/ablations`

各 spec 的用途：

- `smoke_cifar100`: 5 epoch 的快速连通性检查，只跑 `FangAttack` 对 `Mean/CARAT`。
- `clean_reference`: clean utility 基线，`num_adv=0`，比较 `Mean/FLTrust/MultiKrum/CARAT`。
- `pilot_untargeted`: 正式大矩阵前的小规模 untargeted 试跑。
- `main_untargeted`: 论文主体的 untargeted poisoning benchmark。
- `backdoor`: backdoor benchmark，关注 clean accuracy 和 ASR。
- `ablations`: CARAT 关键组件和超参数的消融实验。

## 先看矩阵

先确认 launcher 能识别这些 spec：

```bash
python exps/launch.py list | rg CARAT
```

看某个 spec 会展开成多少个 task：

```bash
python exps/launch.py plan CARAT/smoke_cifar100
python exps/launch.py plan CARAT/main_untargeted
python exps/launch.py plan CARAT/backdoor
python exps/launch.py plan CARAT/ablations
```

当前矩阵规模：

- `smoke_cifar100`: 2 tasks
- `clean_reference`: 36 tasks
- `pilot_untargeted`: 12 tasks
- `main_untargeted`: 216 tasks
- `backdoor`: 60 tasks
- `ablations`: 33 tasks

## 本地运行

先跑 smoke test：

```bash
./exps/run_local.sh CARAT/smoke_cifar100 --jobs 1 --gpu-tokens 1
```

如果只想跑完整矩阵中的一小段 task：

```bash
./exps/run_local.sh CARAT/main_untargeted --ids 0-3 --jobs 1 --gpu-tokens 1
./exps/run_local.sh CARAT/backdoor --ids 0-3 --jobs 1 --gpu-tokens 1
```

失败后补跑未完成 task：

```bash
./exps/run_local.sh CARAT/main_untargeted --ids 0-31 --jobs 1 --gpu-tokens 1 --resume
```

说明：

- `--jobs` 控制本地并发 worker 数。
- `--gpu-tokens` 控制同时真正占 GPU 的 task 数。
- 现在 batch worker 默认关闭 `log_stream`，所以 `logs/local_array/*.out` 不再镜像整份训练日志，只保留 task 级别的起止和退出状态。

## Compute Canada 运行

先 dry-run 看 sbatch 命令是否正确：

```bash
./exps/run_cc.sh CARAT/smoke_cifar100 --chunk-size 1 --dry-run
./exps/run_cc.sh CARAT/main_untargeted --chunk-size 24 --dry-run
./exps/run_cc_carat.sh --dry-run
```

正式提交常用命令：

```bash
./exps/run_cc.sh CARAT/clean_reference --chunk-size 12
./exps/run_cc.sh CARAT/pilot_untargeted --chunk-size 12
./exps/run_cc.sh CARAT/main_untargeted --chunk-size 24
./exps/run_cc.sh CARAT/backdoor --chunk-size 16
./exps/run_cc.sh CARAT/ablations --chunk-size 16
```

如果要按推荐顺序一条命令串行提交完整 CARAT 主实验：

```bash
./exps/run_cc_carat.sh
```

这个脚本会按下面顺序提交，并用 `afterok` 串起来：

1. `CARAT/clean_reference`
2. `CARAT/pilot_untargeted`
3. `CARAT/main_untargeted`
4. `CARAT/backdoor`
5. `CARAT/ablations`

补跑缺失结果时：

```bash
./exps/run_cc_carat.sh --resume
```

如果只提交矩阵中的一段：

```bash
./exps/run_cc.sh CARAT/main_untargeted --start-id 0 --end-id 47 --chunk-size 16
```

如果只补提交还没完成的 task：

```bash
./exps/run_cc.sh CARAT/main_untargeted --chunk-size 24 --resume
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

- 现在训练默认只保留原始指标数据，不再自动生成 `png` 图。
- `metrics_exp*.csv` 直接保存每轮 `train_*` 和 `eval_*`。
- 详细训练过程写在 `logs/run_logs/.../*.log`。
- backdoor 的 ASR 不在 `metrics_exp*.csv` 里，需要从 `run_logs` 解析。

## 设计说明

- 主 benchmark 固定用 `CIFAR100 + ResNet-18`，因为 CARAT 是 class-aware，更多类别时更能体现差异。
- clean runs 和 attack runs 分开写，避免 `NoAttack` 和非零 `num_adv` 被无意义地交叉。
- untargeted benchmark 的对照组是 `Mean`、`FLTrust`、`MultiKrum`、`CARAT`。
- backdoor benchmark 换成更适合 backdoor 的对照组：`Mean`、`FLTrust`、`FLAME`、`DeepSight`、`CARAT`。
- 所有论文主结果 spec 都显式设置了 `evaluate: true`。

## 建议流程

1. 先 `python exps/launch.py plan <spec>` 看矩阵大小。
2. 先本地跑一个 smoke 或小范围 `--ids`。
3. 确认输出目录、日志和 `metrics_exp*.csv` 正常。
4. 再提交完整矩阵。
