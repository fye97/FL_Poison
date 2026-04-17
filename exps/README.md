# Experiments

`exps/` 只保留两个用户入口脚本：

- `run_local.sh`: 本地批量运行实验矩阵
- `run_cc.sh`: 在 Compute Canada 上批量提交实验矩阵

其余文件职责：

- `specs/<family>/*.yaml`: 实验矩阵定义，`launch.py list` 会递归发现子目录里的 spec
- `launch.py`: 后端实现，负责 `list`、`plan`、矩阵展开、worker 执行、`sbatch` 生成
- `slurm_array_entry.sh`: Slurm array 内部 worker 入口

推荐在命令里使用相对 `exps/specs/` 的 grouped spec id，例如 `TriguardFL/smoke_mnist` 或 `CARAT/omnibus`。
如果 bare spec 名称在所有子目录里全局唯一，也仍然可以直接传 bare name；一旦重名，launcher 会要求你写全目录前缀。

## 常用命令

查看可用 spec：

```bash
python exps/launch.py list
```

查看某个 spec 会展开成多少个 task：

```bash
python exps/launch.py plan TriguardFL/smoke_mnist
python exps/launch.py plan TriguardFL/e1_cifar10
```

本地批量运行：

```bash
./exps/run_local.sh TriguardFL/smoke_mnist --ids 0 --jobs 1
./exps/run_local.sh TriguardFL/e1_cifar10 --ids 0-3 --jobs 1 --resume
```

提交到 Compute Canada：

```bash
./exps/run_cc.sh TriguardFL/smoke_mnist --chunk-size 1 --dry-run
./exps/run_cc.sh TriguardFL/e1_cifar10 --chunk-size 32
./exps/run_cc.sh CARAT/clean_reference CARAT/pilot_untargeted --chain-specs --dry-run
```

## 批量实验矩阵设计

这个 launcher 的核心设计是：一个 spec 不再描述“单次实验”，而是描述“一整批要系统性扫过的实验组合”。

一个 spec 由三层信息组成：

- `scenarios`: 基础场景。通常定义算法、数据集，以及可选的 base config。
- `matrix`: 要 sweep 的维度，例如 `distribution`、`attack`、`defense`、`num_advs`、`seed`。
- `repeats`: 重复次数，会展开成不同的 `experiment_id`，并和 base seed 组合出最终的 `effective_seed`。

launcher 会对这些维度做笛卡尔积展开。展开后的每一个组合就是一个独立 `task`，有稳定的 `task_id`，从 `0` 开始编号。

总任务数可以理解为：

```text
total tasks =
  scenarios
  x distributions
  x models
  x epochs
  x num_clients
  x learning_rates
  x num_advs
  x seeds
  x attacks
  x defenses
  x experiment_ids
```

几个设计约定：

- 如果某个 `matrix` 字段省略，会回退到对应 config 里的默认值。
- `matrix.distributions` 里可以带 `dirichlet_alpha` 或 `im_iid_gamma` 这样的分布参数。
- `repeats.count/start` 会展开为不同的 `experiment_id`，并叠加到 seed 上。
- `plan` 只做展开和检查，不运行任务；适合先确认矩阵大小、场景和示例命令。
- `--ids`、`--start-id/--end-id` 允许只跑矩阵中的一部分 task。

推荐流程：

1. 先 `plan` 看矩阵是否符合预期。
2. 先用 `TriguardFL/smoke_mnist` 或你自己的一小段 `--ids` 做 smoke test。
3. 确认输出路径、参数和结果结构都正常后，再跑完整矩阵。

## Local 批量运行

`run_local.sh` 实际等价于：

```bash
python exps/launch.py local <spec> ...
```

它的定位是：

- 在一台机器上批量执行一段 task id
- 适合 smoke test、小范围 sweep、失败任务补跑
- 适合先验证矩阵配置、输出路径和训练逻辑

### 并发模型

本地批量运行有两层控制：

- `--jobs N`: 同时启动多少个本地 worker 线程
- `--gpu-tokens N`: 同时允许多少个 worker 真正占用 GPU

也就是说：

- 如果 `--jobs 4 --gpu-tokens 1`，最多会有 4 个调度线程在排队，但任意时刻只有 1 个任务拿到 GPU token 并开始执行。
- 如果 `--jobs 4 --gpu-tokens 2`，最多同时有 2 个任务真正跑在 GPU 上。
- 如果只是 CPU / MPS smoke test，可以把两者都设成较小值，避免资源争抢。

常用参数：

- `--ids 0-7` 或 `--ids 0,3,8-10`: 只跑部分 task
- `--jobs N`: 本地并发 worker 数
- `--gpu-tokens N`: GPU token 锁，限制同时占用 GPU 的 worker 数
- `--resume`: 只补跑未完成的 task
- `--dry-run`: 只生成本地 runner log，不真正启动 worker
- `--log-dir <path>`: 改本地 runner log 目录
- `--stop-on-fail`: 遇到第一个失败 task 就停

例子：

```bash
./exps/run_local.sh TriguardFL/smoke_mnist --ids 0 --jobs 1
./exps/run_local.sh TriguardFL/e1_cifar10 --ids 0-7 --jobs 2 --gpu-tokens 1
./exps/run_local.sh TriguardFL/e1_cifar10 --ids 0-31 --jobs 1 --resume
```

### 结果保存

本地结果根目录默认是：

```text
logs/local_runs/
```

本地 runner 日志默认在：

```text
logs/local_array/
```

其中每个 task 的 runner 日志文件名是：

```text
logs/local_array/<spec-slug>_task<task_id>.out
```

例如：

```text
logs/local_array/TriguardFL__smoke_mnist_task0.out
```

真正的实验结果按 task 参数分层保存。目录结构是：

```text
logs/local_runs/
  <algorithm>/
    <dataset>_<model>/
      <distribution>/
        <attack>__<defense>/
          ep<epochs>_clients<num_clients>_lr<learning_rate>_adv<num_adv>_seed<effective_seed>_exp<experiment_id>[_alpha<dirichlet_alpha> | _gamma<im_iid_gamma>]_cfg<config-stem>/
            metrics.csv
            run.log
            jobmeta.txt
            task.complete
```

例子：

```text
logs/local_runs/FedAvg/CIFAR10_vgg19/non-iid/FangAttack__TriGuardFL/ep200_clients20_lr0.05_adv0.1_seed42_exp0_alpha0.1_cfgFedAvg_CIFAR10_vgg19/
```

每个 task 目录里的关键文件：

- `metrics.csv`: 训练/评估指标
- `run.log`: 训练过程自身输出的日志
- `jobmeta.txt`: launcher 记录的元数据，包括 spec、task_id、host、python、git commit、输出路径等
- `task.complete`: 完成标记。`--resume` 会用它和结果文件一起判断是否需要跳过该 task

默认行为：

- 本地如果 CUDA 不可用，允许回退到 CPU/MPS
- 如果 spec 里显式设置 `runtime.require_cuda: true`，本地也会强制要求 CUDA
- `evaluate` 默认开启；如果某个 spec 需要关闭，可以在 spec 顶层写 `evaluate: false`

## Compute Canada 批量提交

`run_cc.sh` 实际等价于：

```bash
python exps/launch.py cc <spec> ...
```

在 Compute Canada 上，最简单的做法就是先手动初始化环境，再直接运行提交脚本：

```bash
git clone <your-repo-url> FL_Poison
cd FL_Poison
module purge
module load python/3.12 cuda/12.2
python -m pip install --user uv
uv sync --frozen
source .venv/bin/activate
./exps/run_cc.sh <spec> --dry-run
./exps/run_cc.sh <spec>
```

如果你想把整套 CARAT spec 一次性并行提交，直接：

```bash
./exps/run_cc_carat.sh --dry-run
./exps/run_cc_carat.sh
```

这个脚本只做并行提交；它会连续执行多次 `sbatch`，让各个 spec 在 Slurm 里并行排队。

它的定位是：

- 把一个大矩阵拆成多批 `sbatch` 提交
- 利用 Slurm array 并发跑大量 task
- 适合完整 benchmark、大规模参数 sweep、长时间运行

### 并发模型

Compute Canada 的并发不是简单地“一次提交一个实验”，而是两层并发：

- `--chunk-size N`: 把选中的 task id 切成多少大小的块；每个块对应一次 `sbatch` 提交
  - 不传时，优先使用 `spec.slurm.chunk_size`
  - spec 里也没写时，默认回退到 `32`
- `array_parallel`: 每个 array job 内，最多允许多少个 array element 同时运行

例如，如果你选中了 `0-255` 共 256 个 task，并设置：

```bash
./exps/run_cc.sh TriguardFL/e1_cifar10 --start-id 0 --end-id 255 --chunk-size 32
```

那么 launcher 会：

1. 把 256 个 task 切成 8 个 chunk
2. 提交 8 次 `sbatch`
3. 每次 `sbatch` 都是一个包含 32 个 array element 的 Slurm array
4. 每个 array 里同时运行多少个 element，由 `spec.slurm.array_parallel` 或 `--array-parallel` 决定

这意味着：

- 一个 spec 可以同时在队列里挂着多个 array job
- 多个 array job 会被调度器并行调度
- 同一个 array job 内也可以同时跑多个 task

如果你希望整体并发更可控，最稳妥的做法是把 `chunk_size` 设成接近 spec 总 task 数，
这样一个 spec 只会生成一个 array job，而总 GPU 并发几乎完全由 `array_parallel` 决定。

所以从使用效果上说，`cc` 模式确实是在“同时跑多个实验”，而不是串行一个一个地提。

常用参数：

- `--chunk-size N`: 每个 `sbatch` 提交里包含多少个 task id
- `--start-id N --end-id M`: 只提交矩阵的一部分
- `--resume`: 只提交结果目录尚未完成的 task
- `--array-parallel N`: 覆盖 `spec.slurm.array_parallel`
- `--chain-specs`: 当一次给多个 spec 时，用 `afterok` 把它们串成顺序提交
- `--sbatch-arg=...`: 追加额外 `sbatch` 参数
- `--dry-run`: 只打印 `sbatch` 命令，不实际提交

例子：

```bash
./exps/run_cc.sh TriguardFL/smoke_mnist --chunk-size 1 --dry-run
./exps/run_cc.sh TriguardFL/e1_cifar10 --chunk-size 32
./exps/run_cc.sh TriguardFL/e1_cifar10 --start-id 0 --end-id 63 --chunk-size 16
./exps/run_cc.sh TriguardFL/e1_cifar10 --start-id 0 --end-id 255 --chunk-size 32 --resume
./exps/run_cc.sh CARAT/clean_reference CARAT/pilot_untargeted CARAT/main_untargeted --chain-specs
```

### 结果保存

Compute Canada 结果根目录默认是：

- `$SCRATCH/FL_Poison/logs`
- 如果没有 `$SCRATCH`，回退到 `~/scratch/FL_Poison/logs`

在结果根目录下面，task 的内部目录结构和本地完全一致：

```text
<result-root>/
  <algorithm>/
    <dataset>_<model>/
      <distribution>/
        <attack>__<defense>/
          <run-dir>/
            metrics.csv
            run.log
            jobmeta.txt
            task.complete
```

也就是说，本地和 CC 的区别主要在“结果根目录不同”，而不是“task 内部目录结构不同”。

此外还有两类额外日志：

- Slurm stdout/stderr：默认在仓库里的 `logs/slurm/%x_%A_%a.out` 和 `logs/slurm/%x_%A_%a.err`
- worker 内部生成的训练日志：同步保存在每个 task 目录里的 `run.log`

默认行为：

- Compute Canada worker 默认强制要求 CUDA
- 如果节点提供 `SLURM_TMPDIR`，worker 会优先把代码和数据 stage 到节点本地目录，再把结果同步回最终结果目录
- `slurm.output` / `slurm.error` 如果写相对路径，会解析到仓库下的 `logs/slurm/...`

提交前建议检查这些字段：

- `slurm.account`
- `slurm.time`
- `slurm.gpus`
- `slurm.cpus_per_task`
- `slurm.mem`
- `slurm.mail_user`
- `slurm.mail_type`
- `slurm.array_parallel`
- `slurm.chunk_size`

## 结果完整性与 `--resume`

`local` 和 `cc` 都支持 `--resume`，但它们检查的是各自平台对应的结果根目录：

- `local` 看 `logs/local_runs`
- `cc` 看 `$SCRATCH/FL_Poison/logs` 或 `~/scratch/FL_Poison/logs`

一个 task 被判定为“完成”时，launcher 会检查：

- `metrics.csv` 存在
- `jobmeta.txt` 存在
- `task.complete` 存在

如果标记文件不在，但 `metrics.csv` 的 epoch 行数已经达到预期 epoch 数，launcher 也会把它当作已完成。

这让你可以：

- 中断后继续补跑
- 只重提失败或缺结果的 task
- 在大矩阵里多次分批提交而不重复计算已经完成的组合

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
  chunk_size: 16
  array_parallel: 3
```

字段说明：

- 如果某个 `matrix` 字段省略，会回退到对应 config 里的默认值
- `matrix.distributions` 里可以放 `dirichlet_alpha` 或 `im_iid_gamma`
- `matrix.num_advs` 是攻击者比例或数量 sweep
- `repeats.count/start` 会展开成不同的 `experiment_id` 和有效 seed
- `runtime.require_cuda` 可选；不写时，本地默认 `false`，Compute Canada 默认 `true`
- `runtime.num_workers` 会传给训练程序本身，不等于 launcher 的 `--jobs`
- 顶层 `evaluate` 控制 launcher 是否显式传 `--evaluate` / `--no-evaluate`
- `slurm.array_parallel` 控制单个 array job 内允许同时运行的 task 数
- `slurm.chunk_size` 控制默认每次 `sbatch` 包含多少个 task；CLI 传 `--chunk-size` 时会覆盖它

## Smoke Test

`TriguardFL/smoke_mnist` 是一个极小 spec，用来快速验证流程：

```bash
python exps/launch.py plan TriguardFL/smoke_mnist
./exps/run_local.sh TriguardFL/smoke_mnist --ids 0 --jobs 1
./exps/run_cc.sh TriguardFL/smoke_mnist --chunk-size 1 --dry-run
```
