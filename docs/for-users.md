---
title: 用户使用手册
description: 安装环境、运行单个 FL 训练，以及在本地或 Compute Canada 上批量运行实验。
outline: [2, 3]
---

# 用户使用手册

这份手册面向想直接运行 FLPoison 的用户，覆盖三类常见流程：

1. 运行单个 FL 训练任务
2. 在本地机器上批量运行实验矩阵
3. 在 Compute Canada 的 Slurm 机器上批量提交实验

当前仓库里仍保留了根目录的 `batchrun.py` 作为兼容入口，但新的批量实验流程推荐统一使用 `exps/` 目录下的 spec 驱动脚本。

## 环境准备

项目要求 Python 3.10 及以上。推荐先创建虚拟环境，再以 editable 模式安装：

```bash
uv venv
uv pip install -e .
```

如果你不用 `uv`，也可以使用标准 `venv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果你需要运行 `EdgeCase` 一类依赖压缩包解压的数据流程，可能还需要系统里的 `unrar`：

Ubuntu:

```bash
sudo apt install unrar
```

macOS:

```bash
brew install unrar
```

## 运行单个 FL 训练

### 选择配置文件

单次训练直接从 `configs/` 下的实验配置启动。

- `configs/*.yaml` 是可直接运行的实验 preset
- `configs/presets/attacks.yaml`、`configs/presets/defenses.yaml`、`configs/presets/datasets.yaml` 存放共享元数据

例如：

- `configs/FedSGD_MNIST_Lenet.yaml`
- `configs/FedAvg_CIFAR10_Resnet18.yaml`
- `configs/FedAvg_HAR_Fcn.yaml`

### 最短运行命令

```bash
python main.py --config configs/FedSGD_MNIST_Lenet.yaml
```

等价写法：

```bash
python -m flpoison --config configs/FedSGD_MNIST_Lenet.yaml
```

这会读取 YAML 配置，完成数据集加载、客户端划分、训练、评估和日志输出。

### 命令行覆盖配置

你可以在 YAML 的基础上，通过命令行覆盖常用参数，例如训练轮数、攻击、防御、学习率、客户端数量等：

```bash
python main.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --epochs 10 \
  --attack MinSum \
  --defense Krum \
  --num_adv 0.2 \
  --learning_rate 0.005
```

也可以直接切换数据划分、模型或随机种子：

```bash
python main.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --distribution non-iid \
  --dirichlet_alpha 0.1 \
  --model lenet \
  --seed 123
```

如果要覆盖攻击或防御的参数对象，需要整体传入该参数对象，而不是只改其中一个字段：

```bash
python main.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --attack IPM \
  --attack_params "{'scaling_factor': 0.5}"
```

如果你只覆盖 `--attack` 或 `--defense`，但没有额外提供 `--attack_params` 或 `--defense_params`，运行时会优先从共享 preset 里查找对应默认参数；找不到时回退到实现内部默认值。

### 输出文件在哪里

如果不显式指定 `--output`，单次训练默认把结果写到：

```text
./logs/{algorithm}/{dataset}_{model}/{distribution}/{dataset}_{model}_{distribution}_{attack}_{defense}_{epochs}_{num_clients}_{learning_rate}_{algorithm}.txt
```

例如：

```text
./logs/FedSGD/MNIST_lenet/iid/MNIST_lenet_iid_NoAttack_Mean_300_50_0.01_FedSGD.txt
```

你也可以自己指定输出文件：

```bash
python main.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --output logs/manual_runs/mnist_smoke.txt
```

### 重复运行多个 seed

单次入口本身也支持重复实验：

```bash
python main.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --num_experiments 3 \
  --experiment_id 0
```

运行时会自动调整 `seed` 和输出文件名中的 `exp{}` / `seed{}` 标记，避免不同重复实验互相覆盖。

### 什么时候用单次入口

下面这些情况适合直接用 `main.py`：

- 你只想跑一个配置文件
- 你正在调试某个 attack / defense / dataset 组合
- 你想先确认配置本身能否正常训练，再扩展成批量实验

完整参数说明见 [配置手册](/config-manual)。

## 批量运行总览

推荐使用 `exps/` 目录下的 spec 驱动流程：

- `exps/specs/*.yaml`: 实验矩阵定义
- `exps/run_local.sh`: 本地批量运行入口
- `exps/run_cc.sh`: Compute Canada / Slurm 提交入口
- `exps/launch.py`: 后端实现，支持 `list`、`plan`、`local`、`cc`

建议按下面顺序使用：

1. 列出可用 spec
2. 用 `plan` 查看任务矩阵大小
3. 先跑一个很小的本地 smoke test
4. 再决定是在本地批量跑，还是提交到 Compute Canada

### 查看可用 spec

```bash
python exps/launch.py list
```

### 查看某个 spec 会展开成多少 task

```bash
python exps/launch.py plan smoke_mnist
python exps/launch.py plan cifar10
```

这里的 `spec` 既可以写名字，也可以写路径，例如：

```bash
python exps/launch.py plan exps/specs/smoke_mnist.yaml
```

### spec 文件长什么样

批量实验由 `exps/specs/*.yaml` 描述，一个典型 spec 包含以下部分：

```yaml
name: cifar10

scenarios:
  - algorithm: FedAvg
    dataset: CIFAR10

matrix:
  distributions:
    - type: iid
    - type: non-iid
      dirichlet_alpha: 0.1
  attacks:
    - NoAttack
    - MinSum
  defenses:
    - Mean
    - FLTrust
  num_advs:
    - 0.2
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

repeats:
  count: 5
  start: 0

runtime:
  gpu_idx: 0
  num_workers: 4
  require_cuda: true

slurm:
  account: def-yourgroup
  time: 0-12:00:00
  gpus: nvidia_h100_80gb_hbm3_2g.20gb:1
  cpus_per_task: 8
  mem: 64G
  array_parallel: 3
```

字段含义可以先按下面理解：

- `scenarios`: 基础场景，定义算法、数据集和可选的基础 config
- `matrix`: 要 sweep 的维度，最终会展开成 task 网格
- `repeats`: 每个组合重复多少次；会自动映射到不同 `experiment_id` 和有效 seed
- `runtime`: 本地或 worker 运行时设置，例如 `gpu_idx`、`num_workers`、是否强制 CUDA
- `slurm`: 只有走 Slurm 提交时才会用到的资源申请参数

如果某个 `matrix` 字段省略，launcher 会回退到对应 config 文件里的默认值。

## 本地批量运行

### 推荐流程

先做一个 smoke test：

```bash
python exps/launch.py plan smoke_mnist
./exps/run_local.sh smoke_mnist --ids 0 --jobs 1
```

确认 smoke test 没问题后，再扩大任务范围：

```bash
./exps/run_local.sh cifar10 --ids 0-7 --jobs 2 --gpu-tokens 1
./exps/run_local.sh cifar10 --ids 0-31 --jobs 1 --resume
```

`run_local.sh` 实际调用的是：

```bash
python exps/launch.py local <spec> ...
```

### 常用参数

- `--ids 0-7` 或 `--ids 0,3,8-10`: 只运行部分 task
- `--jobs N`: 同时启动的本地 worker 数
- `--cuda 0`: 设置 `CUDA_VISIBLE_DEVICES`
- `--gpu-tokens N`: 用文件锁限制同时占用 GPU 的 worker 数
- `--resume`: 跳过已经成功结束的 task
- `--dry-run`: 只生成 runner 日志，不真正启动 worker
- `--stop-on-fail`: 遇到第一个失败任务就停止
- `--log-dir <path>`: 修改本地 runner 日志目录

### 输出位置

本地批量运行会产生两类输出：

- 实验结果：`logs/local_runs/...`
- runner 日志：`logs/local_array/...`

`logs/local_array/` 里每个 task 会有一个单独的 `*.out` 文件，记录该 task 的启动命令、开始时间、退出码和 worker 输出。

实验结果文件名会包含数据集、模型、攻击、防御、seed 和 experiment id，便于后续筛选和汇总。

### 本地运行时的默认行为

- 如果本地检测不到 CUDA，launcher 默认允许回退到 CPU 或 MPS
- 如果 spec 里写了 `runtime.require_cuda: true`，本地也会强制要求 CUDA
- `run_local.sh` 会优先使用 `PYTHON_BIN`，否则尝试仓库下的 `.venv/bin/python`

## 在 Compute Canada / Slurm 上批量运行

### 提交前先检查 spec

在 Compute Canada 上提交前，至少检查 `slurm` 段里的这些字段：

- `account`
- `time`
- `gpus`
- `cpus_per_task`
- `mem`
- `array_parallel`
- `mail_user`
- `mail_type`

一个最小可用流程通常是：

1. 把仓库放到集群环境中
2. 创建虚拟环境并安装依赖
3. 根据账号和资源队列修改 spec 里的 `slurm` 参数
4. 先做一次 `--dry-run`
5. 再正式 `sbatch` 提交

### 先 dry run 看 sbatch 命令

```bash
./exps/run_cc.sh smoke_mnist --chunk-size 1 --dry-run
```

### 正式提交

```bash
./exps/run_cc.sh cifar10 --chunk-size 32
./exps/run_cc.sh cifar10 --start-id 0 --end-id 63 --chunk-size 16
```

`run_cc.sh` 实际调用的是：

```bash
python exps/launch.py cc <spec> ...
```

### 常用参数

- `--chunk-size N`: 每次 `sbatch` 提交包含多少个 task id
- `--array-parallel N`: 覆盖 `spec.slurm.array_parallel`
- `--start-id N --end-id M`: 只提交矩阵中的一段任务
- `--sbatch-arg=...`: 追加额外 `sbatch` 参数，例如 `--sbatch-arg=--qos=high`
- `--dry-run`: 只打印命令，不真正提交

### Compute Canada 运行时的默认行为

- Compute Canada worker 默认要求 CUDA 可用
- 如果 spec 没显式设置 `runtime.require_cuda`，在 Slurm 环境里默认会按 `true` 处理
- 如果节点上暂时拿不到可用 CUDA，worker 会按 runtime 里的重试和 requeue 策略处理

### Compute Canada 上的数据与结果目录

批量 worker 会按下面顺序寻找数据目录：

1. `DATA_SRC_ROOT`
2. `$SCRATCH/FL_Poison/data`
3. `$PROJECT/FL_Poison/data`
4. `<repo>/data`

训练结果默认写到：

- `$SCRATCH/FL_Poison/logs/...`

如果没有 `SCRATCH`，则回退到：

- `~/scratch/FL_Poison/logs/...`

Slurm 的 stdout / stderr 默认写到仓库里的：

- `logs/slurm/%x_%A_%a.out`
- `logs/slurm/%x_%A_%a.err`

如果节点提供了 `SLURM_TMPDIR`，worker 会把代码和对应数据集先拷到本地临时盘运行，任务结束后再把结果同步回最终日志目录。这通常比直接在共享文件系统上跑更稳。

## 常用环境变量

如果你需要定制运行环境，可以使用这些环境变量：

- `PYTHON_BIN`: 显式指定 Python 解释器
- `CODE_SRC_ROOT`: 指定代码根目录
- `DATA_SRC_ROOT`: 指定数据源目录
- `RESULT_ROOT`: 覆盖结果输出根目录

这几个变量对本地批量和 Compute Canada 批量流程都生效。

## 兼容旧批量入口

根目录的 `batchrun.py` 仍可用，但它属于旧式批量脚本，更适合兼容已有命令而不是组织新实验。新的实验矩阵、任务拆分、`plan` 预览和 Slurm 提交流程，建议统一使用 `exps/` 下的 spec 驱动入口。

如果你只是想快速看一个旧命令长什么样，示例如下：

```bash
python batchrun.py \
  --algorithms FedSGD \
  --dataset MNIST \
  --model lenet \
  --distributions non-iid \
  --attacks MinMax MinSum \
  --defenses Krum Median \
  --gpu_idx 1 \
  --max_processes 3
```

## 进一步阅读

- [配置手册](/config-manual)
- [性能 Profiling](/performance-profiling)
