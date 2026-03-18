# Experiments

`exps/` 现在只保留两个用户入口脚本：

- `run_local.sh`: 本地运行实验
- `run_cc.sh`: 在 Compute Canada 上提交实验

其余文件职责：

- `specs/*.yaml`: 实验矩阵定义
- `launch.py`: 后端实现，负责 `list`、`plan`、矩阵展开、worker 执行、`sbatch` 生成
- `slurm_array_entry.sh`: Slurm array 内部 worker 入口

## 常用命令

查看可用 spec：

```bash
python exps/launch.py list
```

查看某个 spec 会展开成多少个 task：

```bash
python exps/launch.py plan smoke_mnist
python exps/launch.py plan cifar10
```

本地运行：

```bash
./exps/run_local.sh smoke_mnist --ids 0 --jobs 1
./exps/run_local.sh cifar10 --ids 0-3 --jobs 1 --resume
```

提交到 Compute Canada：

```bash
./exps/run_cc.sh smoke_mnist --chunk-size 1 --dry-run
./exps/run_cc.sh cifar10 --chunk-size 32
```

## 推荐流程

1. 先 `plan` 看任务总数。
2. 先用 `smoke_mnist` 或者你自己的小范围 `--ids` 在本地 smoke test。
3. 确认参数和输出路径正常后，再用 `run_cc.sh` 提交大矩阵。

## Local Runs

`run_local.sh` 实际等价于：

```bash
python exps/launch.py local <spec> ...
```

常用参数：

- `--ids 0-7` 或 `--ids 0,3,8-10`: 只跑部分 task
- `--jobs N`: 本地并发 worker 数
- `--gpu-tokens N`: GPU token 锁，限制同时占用 GPU 的 worker 数
- `--resume`: 跳过已经成功结束的 task
- `--dry-run`: 只生成本地 runner log，不真正启动 worker
- `--log-dir <path>`: 改本地 runner log 目录

例子：

```bash
./exps/run_local.sh smoke_mnist --ids 0 --jobs 1
./exps/run_local.sh cifar10 --ids 0-7 --jobs 2 --gpu-tokens 1
./exps/run_local.sh cifar10 --ids 0-31 --jobs 1 --resume
```

输出位置：

- 实验结果：`logs/local_runs/...`
- runner 日志：`logs/local_array/...`

默认行为：

- 本地如果 CUDA 不可用，允许回退到 CPU/MPS
- 如果 spec 里显式设置 `runtime.require_cuda: true`，本地也会强制要求 CUDA

## Compute Canada Runs

`run_cc.sh` 实际等价于：

```bash
python exps/launch.py cc <spec> ...
```

常用参数：

- `--chunk-size N`: 每个 `sbatch` 提交里包含多少个 task id
- `--start-id N --end-id M`: 只提交矩阵的一部分
- `--array-parallel N`: 覆盖 `slurm.array_parallel`
- `--sbatch-arg=...`: 追加额外 `sbatch` 参数
- `--dry-run`: 只打印 `sbatch` 命令，不实际提交

例子：

```bash
./exps/run_cc.sh smoke_mnist --chunk-size 1 --dry-run
./exps/run_cc.sh cifar10 --chunk-size 32
./exps/run_cc.sh cifar10 --start-id 0 --end-id 63 --chunk-size 16
```

默认行为：

- Compute Canada worker 默认强制要求 CUDA
- spec 里的 `slurm.output` 和 `slurm.error` 现在写成相对路径，最终会解析到仓库下的 `logs/slurm/...`

提交前建议检查这些字段：

- `slurm.account`
- `slurm.time`
- `slurm.gpus`
- `slurm.cpus_per_task`
- `slurm.mem`
- `slurm.mail_user`
- `slurm.mail_type`

## Spec 结构

典型结构：

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
  num_advs:
    - 0.2
    - 0.3
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

runtime:
  gpu_idx: 0
  num_workers: 4
  require_cuda: true

slurm:
  account: def-lincai
  time: 0-12:00:00
  gpus: nvidia_h100_80gb_hbm3_2g.20gb:1
  cpus_per_task: 8
  mem: 64G
  array_parallel: 3
```

说明：

- 如果某个 matrix 字段省略，会回退到 scenario 对应 config 里的默认值
- `matrix.distributions` 里放 `dirichlet_alpha` 或 `im_iid_gamma`
- `matrix.num_advs` 是攻击者比例或数量 sweep
- `repeats.count/start` 会展开成不同的 `experiment_id` 和有效 seed
- `runtime.require_cuda` 可选；不写时，本地默认 `false`，Compute Canada 默认 `true`

## Smoke Test

`smoke_mnist.yaml` 是一个极小 spec，用来快速验证流程：

```bash
python exps/launch.py plan smoke_mnist
./exps/run_local.sh smoke_mnist --ids 0 --jobs 1
./exps/run_cc.sh smoke_mnist --chunk-size 1 --dry-run
```
