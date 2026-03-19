---
title: 性能剖析
description: 单次 profiling 工作流、输出文件位置与指标解释。
outline: deep
---

# 性能剖析 / Performance Profiling

本文档记录本项目当前已经落地的性能测试与 profiling 工作，包括：
- 基线实验如何固定
- 如何运行单次 profiling
- 输出文件在哪里
- 指标如何解读
- 2026-03-16 的两次基线实验记录

适用范围：
- `flpoison/fl/training.py` 中的运行时性能采集（由 `python -m flpoison` 入口调用）
- `tests/perf/profile_single_run.py` 单次基线脚本
- `torch.profiler` trace 输出

---

## 1. 这套测试现在能做什么

当前已经支持：
- 固定基线实验配置：模型、batch size、client 数、local epoch、数据切分、聚合算法、seed
- 记录系统信息：GPU 型号、PyTorch 版本、CUDA 版本
- 记录每轮 round 的总时间
- 拆分每轮阶段耗时：
  - `sync`
  - `data`
  - `fwd_bwd`
  - `gpu_compute`
  - `opt_step`
  - `pack_update`
  - `collect_updates`
  - `aggregate`
  - `defense`
  - `evaluation`
  - `logging`
- 记录 GPU utilization、显存平均占用、显存峰值
- 记录每轮轻量日志里的 `train accuracy` / `train loss` / `train samples` / `round time`
- 记录最终 `train accuracy` / `train samples` / `Test Acc`
- 运行 `torch.profiler`，输出 CPU/CUDA trace 与 profiler 摘要

---

## 2. 推荐使用方式

推荐优先用单次 profiling 脚本：

```bash
python tests/perf/profile_single_run.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --defense Mean \
  --epochs 20 \
  --num-clients 10 \
  --batch-size 64 \
  --eval-batch-size 1024 \
  --local-epochs 1 \
  --seed 7
```

这条默认命令现在会自动把 profiling 配置里的 `eval_interval` 设成当前 `epochs`，也就是默认只在最后一轮做完整评估。这样保留最终指标，同时避免评估把训练主路径淹没。

如果你要做“端到端”基线，而不是默认的“训练吞吐”基线，显式传 `--eval-interval`：

```bash
python tests/perf/profile_single_run.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --defense Mean \
  --epochs 20 \
  --num-clients 10 \
  --batch-size 64 \
  --eval-batch-size 1024 \
  --local-epochs 1 \
  --seed 7 \
  --eval-interval 10
```

如果还要同时跑 `torch.profiler`：

```bash
python tests/perf/profile_single_run.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --defense Mean \
  --epochs 20 \
  --num-clients 10 \
  --batch-size 64 \
  --eval-batch-size 1024 \
  --local-epochs 1 \
  --seed 7 \
  --torch-profile
```

说明：
- `tests/perf/profile_single_run.py` 会生成一份临时配置文件，并强制开启 `record_time=True`
- 若未显式传 `--eval-interval`，脚本会把 `eval_interval` 设成 `epochs`，即默认仅最后一轮评估
- 默认会固定 `num_experiments=1`、`experiment_id=0`
- 输出文件名会带 `_exp0`
- 对当前 `FedSGD + lenet + MNIST + 10 clients` 基线，建议显式设置 `--eval-batch-size 1024`
- 2026-03-16 的 sweep 中位数结果：`64 -> sec/round 0.2828, evaluation 0.1591`；`256 -> 0.2074, 0.0823`；`512 -> 0.1900, 0.0518`；`1024 -> 0.1858, 0.0485`

---

## 3. 关键输出文件

以一次运行目录 `logs/perf_baseline/<timestamp>_FedSGD_MNIST_config/` 为例：

- 运行命令与 stdout：
  - `logs/perf_baseline/.../runner.stdout.log`
- 训练日志：
  - `logs/perf_baseline/.../single_run_exp0.txt`
- 性能 JSON：
  - `logs/perf_logs/perf_baseline/.../single_run_exp0.json`
- 老的 time recorder 日志：
  - `logs/time_logs/perf_baseline/.../single_run_exp0.log`
- `torch.profiler` trace：
  - `logs/torch_traces/perf_baseline/.../single_run_exp0/*.pt.trace.json`

其中最重要的是：
- `runner.stdout.log`
- `single_run_exp0.txt`
- `single_run_exp0.json`

---

## 4. 如何看指标

### 4.1 Baseline summary

`tests/perf/profile_single_run.py` 结束后会打印一段摘要，例如：

```text
Baseline summary
config: model=lenet batch_size=64 eval_batch_size=1024 clients=10 local_epochs=1 distribution=iid defense=Mean seed=7
sec/round=0.3324 rounds/sec=3.0084 sec/client=0.0227
gpu_util=87.15 gpu_compute_ratio=0.2372 gpu_mem_peak_mb=31.05
train_acc=0.23125 val_acc=0.2562
```

这些字段的含义：
- `sec/round`: 平均每轮总耗时
- `rounds/sec`: 吞吐
- `sec/client`: 平均单 client 的训练路径耗时
- `gpu_util`: 运行期间采样得到的平均 GPU utilization
- `gpu_compute_ratio`: `纯 GPU compute 时间 / round 总时间`
- `gpu_mem_peak_mb`: 该次 run 中观测到的显存峰值
- `train_acc`: 最后一轮训练精度
- `val_acc`: 最后一次评估的 `Test Acc`

补充说明：
- `train_metrics` 来自训练过程中的在线统计，不会为了记录它再完整跑一遍 train set。
- `test_metrics` 只会在命中评估调度的 round 填充；非评估轮为空对象。

### 4.2 Round summary

训练日志里每轮都会输出：

```text
Round 19 summary
total: 1.97s
sync: 0.01s
data: 0.17s
fwd_bwd: 0.04s
gpu_compute: 0.05s
opt_step: 0.00s
pack_update: 0.01s
aggregate: 0.00s
defense: 0.00s
collect_updates: 0.00s
evaluation: 1.73s
logging: 0.00s
sec/client: 0.028s
gpu util avg: 44.6%
gpu compute/round: 2.4%
gpu mem avg: 24.9MB
gpu mem peak: 31.0MB
train accuracy: 0.2313
train samples: 640
val accuracy: 0.2562
```

阶段定义：
- `sync`: client 加载 global model
- `data`: DataLoader 取 batch + host/device 数据搬运等待
- `fwd_bwd`: forward + backward 的 wall time
- `gpu_compute`: 由 CUDA event 记录的纯 GPU 时间
- `opt_step`: `optimizer.step()`
- `pack_update`: 本地 update 提取与打包
- `collect_updates`: server 侧收集并堆叠 client update
- `aggregate`: server 最终聚合
- `defense`: defense/filtering 的额外时间
- `evaluation`: 测试集评估
- `logging`: round 结果拼接与日志输出

### 4.3 JSON 里最值得先看的字段

优先看：
- `overall.sec_per_round`
- `overall.rounds_per_sec`
- `overall.sec_per_client`
- `overall.gpu_utilization_pct_avg`
- `overall.gpu_compute_ratio_avg`
- `overall.gpu_memory_peak_allocated_mb`
- `overall.stage_time_sec_avg`
- `overall.final_train_accuracy`
- `overall.final_train_samples`
- `overall.final_test_metrics`

如果要看逐轮细节，读：
- `rounds[i].stage_times`
- `rounds[i].gpu`
- `rounds[i].train_metrics`
- `rounds[i].test_metrics`

---

## 5. 如何解读这些结果

### 5.1 先区分两种 baseline

建议至少保留两种 baseline：

1. 端到端 baseline
- 例如 `--eval-interval 10`
- 适合看完整训练轮的真实业务耗时

2. 训练吞吐 baseline
- `profile_single_run.py` 默认就是这个模式：`eval_interval=epochs`
- 适合看训练主路径，不让评估淹没训练优化效果

如果你需要旧的“每轮都评估”行为，显式传 `--eval-interval 1`。

### 5.2 如何看 `gpu_compute_ratio`

经验口径：
- `<30%`: 系统瓶颈很重
- `30%~50%`: 中等
- `>50%`: 比较健康
- `>60%`: 已经不错

注意：
- 当前 `gpu_compute_ratio` 的分母是整个 round 总时间
- 如果某轮包含 `evaluation`，而分子只统计训练路径里的 GPU compute，这个比例会被明显拉低
- 所以做系统吞吐分析时，更建议看降低评估频率后的 baseline

### 5.3 如何看 `evaluation`

如果 `evaluation` 显著大于其他阶段：
- 不说明训练慢
- 说明“当前 round 的主要成本在评估，而不是训练”

这对后续系统优化非常重要，因为它决定你应该把优化精力放在哪条路径上。

---

## 6. torch.profiler 怎么看

启用 `--torch-profile` 后，会输出 trace 文件：

```text
logs/torch_traces/perf_baseline/.../single_run_exp0/*.pt.trace.json
```

可以用以下方式查看：
- TensorBoard Profile 插件
- Chrome trace viewer

训练日志尾部还会打印：
- top ops 表
- `cudaMemcpy` 提示
- `DataLoader self CPU`
- Python operator 开销
- 小 kernel 热点数量

当前实现主要用它来回答这些问题：
- 是否存在大量 `cudaMemcpy`
- DataLoader 是否占大头
- 是否有大量碎小 kernel
- Python 自身是否成为主要热点

---

## 7. 2026-03-16 基线实验记录

说明：
- 以下数值是当日实现下的历史记录。
- 当前 profiling 脚本默认会把评估调度设成“最后一轮强制评估”，因此 round 级触发点和平均耗时会与下面记录不同。

### 7.1 基线 A：默认按最后输出的配置跑

命令：

```bash
python tests/perf/profile_single_run.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --defense Mean \
  --epochs 20 \
  --num-clients 10 \
  --batch-size 64 \
  --local-epochs 1 \
  --eval-interval 10 \
  --seed 7
```

结果摘要：
- `sec/round = 1.1678`
- `rounds/sec = 0.8563`
- `sec/client = 0.0195`
- `gpu_util = 78.40%`
- `gpu_compute_ratio = 23.72%`
- `gpu_mem_peak = 31.05 MB`
- `train_acc = 0.23125`
- `val_acc = 0.2562`

主要观察：
- 这一版里 `evaluation` 仍然是 round 大头
- 不能把这个 baseline 直接当成训练吞吐基线

### 7.2 基线 B：降低评估频率并启用 torch.profiler

命令：

```bash
python tests/perf/profile_single_run.py \
  --config configs/FedSGD_MNIST_Lenet.yaml \
  --defense Mean \
  --epochs 20 \
  --num-clients 10 \
  --batch-size 64 \
  --local-epochs 1 \
  --seed 7 \
  --eval-interval 20 \
  --torch-profile
```

结果摘要：
- `sec/round = 0.3324`
- `rounds/sec = 3.0084`
- `sec/client = 0.0227`
- `gpu_util = 87.15%`
- `gpu_compute_ratio = 23.72%`
- `gpu_mem_peak = 31.05 MB`
- `train_acc = 0.23125`
- `val_acc = 0.2562`

进一步观察：
- 全部 20 轮中，只有 round `0` 和 round `19` 触发了 evaluation
- 非评估轮平均 `sec/round` 约为 `0.1648`
- 非评估轮平均 `gpu_compute_ratio` 约为 `24.88%`

profiler 摘要结论：
- 存在不少 `Memcpy`
- `DataLoader self CPU` 已经可见
- 当前未观察到明显的“小 kernel 爆炸”
- Python 自身不是主要瓶颈

---

## 8. 当前结论

从“测试基础设施是否到位”这个目标来看，当前已经具备：
- 可重复的单次 baseline 脚本
- 结构化 JSON 指标输出
- round 级阶段拆解
- 纯 GPU compute 计时
- `torch.profiler` trace
- 可以直接用于后续系统优化前的基线建立

当前文档和脚本主要用于“先测清楚”，不是直接做优化。

如果后续继续扩展测试体系，优先级建议是：
- 补更多 defense 的细粒度 `defense/aggregate` 拆分
- 增加 train-only 派生指标导出
- 增加多次重复运行后的聚合报告
