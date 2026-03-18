---
title: 配置手册
description: "`configs/*.yaml`、`configs/presets/*.yaml` 与运行时默认值的完整说明。"
outline: deep
---

# 配置手册

`FL_Poison` 的配置分成三层：

- 可运行实验预设：`configs/{Algorithm}_{Dataset}_{Model}.yaml`
- 共享攻击/防御目录：`configs/presets/attacks.yaml`、`configs/presets/defenses.yaml`
- 数据集元数据目录：`configs/presets/datasets.yaml`

这份手册按代码中的真实生效顺序说明所有配置项、默认值来源、以及攻击/防御参数的覆盖规则。

## 配置解析与覆盖顺序

运行 `python -m flpoison --config ...` 时，配置按下面顺序生效：

1. 读取主实验 YAML。
2. 如果主 YAML 没有 `attacks` 或 `defenses`，自动注入 `configs/presets/attacks.yaml` 与 `configs/presets/defenses.yaml`。
3. 校验 `attack` / `defense` / `attacks` / `defenses` 名称是否合法。
4. 应用 CLI 覆盖。除 `attack` / `defense` / `attack_params` / `defense_params` 外，其余非空 CLI 参数会直接覆盖同名 YAML 字段。
5. `single_preprocess()` 从 `configs/presets/datasets.yaml` 读取当前 `dataset` 的元数据，并直接写回 `args`。
6. `single_preprocess()` 补齐运行时默认值，例如 `eval_batch_size`、`eval_interval`、`record_time`、`torch_profile`、`log_color`、`output`。
7. 攻击器和聚合器构造时，会把 `attack_params` / `defense_params` 合并到各自实现类的 `default_attack_params` / `default_defense_params` 上。

几个关键规则：

- `attack_params` / `defense_params` 是“按键合并”，不是整包替换。你只写要改的键即可，未写的键保留实现默认值。
- 如果主 YAML 没写 `attack_params` / `defense_params`，运行时会先尝试从 `attacks` / `defenses` 列表中找到当前 `attack` / `defense` 的默认参数；找不到时再回退到实现类默认值。
- `num_adv` 和部分攻击里的 `poison_frequency` 都支持“比例或绝对值”。`< 1` 按比例乘总数后取 `int()`，`>= 1` 直接取 `int()`。
- `eval_interval` 不会阻止最后一轮评估。最后一轮始终评估。
- 数据集目录写入是强覆盖：`num_classes`、`mean`、`std`、`num_channels`、`num_features`、`num_dims` 等字段应维护在 `configs/presets/datasets.yaml`，不要期待在主 YAML 里覆写它们。
- 旧字段 `catalogs`、`attack_catalog`、`defense_catalog` 已被废弃；当前代码会直接报错。

## 推荐 YAML 骨架

下面是一个推荐的主配置骨架；`attacks` / `defenses` 一般可以省略，因为共享目录会自动注入。

```yaml
seed: 42
num_experiments: 1
experiment_id: 0

epochs: 300
algorithm: FedAvg
optimizer: SGD
momentum: 0.9
weight_decay: 5.0e-4
learning_rate: 0.05
local_epochs: 4

model: vgg19
dataset: CIFAR10
distribution: iid
dirichlet_alpha: 0.5
im_iid_gamma: 0.01
tail_cls_from: 4

num_clients: 20
num_adv: 0
batch_size: 64
eval_batch_size: 1024
cache_partition: false

gpu_idx: [0]
num_workers: 0
log_stream: true
log_color: auto

record_time: false
gpu_sample_interval_ms: 100
torch_profile: false

attack: NoAttack
defense: Mean
# attack_params:
# defense_params:
# output:
```

## 主实验配置字段

### 运行控制

| 字段 | 类型 / 可选值 | 默认值 / 生效规则 | 说明 |
| --- | --- | --- | --- |
| `seed` | int | 必须最终存在 | 实验随机种子。重复实验时，实际使用的 seed 为 `seed + experiment_id`。 |
| `num_experiments` | int | `1` | 非 benchmark 路径下重复运行次数。每次运行会递增 `experiment_id` 和 seed。 |
| `experiment_id` | int | `0` | 重复实验起始编号。也会写入输出文件名。 |
| `output` | string | 缺省时自动生成 | 日志文件路径。若重复实验启用，会自动追加或改写 `_exp{experiment_id}`；若文件名已有 `seedN` / `expN` 片段，也会同步改写。 |

### 训练与优化

| 字段 | 类型 / 可选值 | 默认值 / 生效规则 | 说明 |
| --- | --- | --- | --- |
| `epochs` | int | 无内置默认 | 全局轮数。 |
| `algorithm` | `FedSGD`, `FedAvg`, `FedOpt` | 无内置默认 | 联邦算法。 |
| `optimizer` | `SGD`, `Adam` | 无内置默认 | 客户端优化器。 |
| `momentum` | float | 仅 `SGD` 使用 | 传给 `torch.optim.SGD(..., momentum=...)`。 |
| `weight_decay` | float | 无内置默认 | 传给 `SGD` 或 `Adam`。 |
| `learning_rate` | float | 无内置默认 | 客户端本地学习率。 |
| `local_epochs` | int | 仅 `FedAvg` / `FedOpt` 使用 | `FedSGD` 会强制把本地轮数设为 `1`。 |
| `lr_scheduler` | `MultiStepLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR` | 不写则恒定学习率 | 当前固定实现为：`MultiStepLR(gamma=0.1)`、`StepLR(step_size=80, gamma=0.5)`、`ExponentialLR(gamma=0.9)`、`CosineAnnealingLR(T_max=epochs 或 epochs*local_epochs)`。 |
| `milestones` | YAML list[int/float] | 仅 `MultiStepLR` 使用 | 元素 `< 1` 时按 `int(value * epochs)` 解释，所以比例写法只适合 YAML；CLI 当前只接受整数列表。 |
| `eval_interval` | int | `10` | 每隔多少轮做一次完整评估；最后一轮始终评估。运行时会用 `max(1, int(eval_interval))`。 |

### 数据、模型与划分

| 字段 | 类型 / 可选值 | 默认值 / 生效规则 | 说明 |
| --- | --- | --- | --- |
| `dataset` | 见下方“数据集与模型可选值” | 无内置默认 | 数据集名称。 |
| `model` | 见下方“数据集与模型可选值” | 无内置默认 | 模型名称。运行时会转为小写。 |
| `distribution` | `iid`, `class-imbalanced_iid`, `non-iid`, `pat`, `imbalanced_pat` | 无内置默认 | 当前真正实现的划分逻辑只有 `iid`、`class-imbalanced_iid`、`non-iid`。`pat` 与 `imbalanced_pat` 仅被 parser 接受，`split_dataset()` 没有对应实现。 |
| `dirichlet_alpha` | float | 仅 `non-iid` 使用 | Dirichlet alpha；越小越异质。 |
| `im_iid_gamma` | float | 仅 `class-imbalanced_iid` 使用 | 类不均衡强度；越小越偏。 |
| `tail_cls_from` | int | 仅不均衡评估时使用 | 当 `distribution` 字符串包含 `imbalanced` 时，评估额外输出 `Tail Acc`，统计标签 `>= tail_cls_from` 的类别。 |
| `num_clients` | int | 无内置默认 | 客户端总数。 |
| `num_adv` | float 或 int | 无内置默认 | 对抗客户端数。`0.2` 表示 `int(0.2 * num_clients)`，`4` 表示 4 个。若 `attack != NoAttack` 且结果为 0，会在初始化客户端时报错。 |
| `batch_size` | int | 无内置默认 | 训练 batch size。 |
| `eval_batch_size` | int | 缺省回退到 `batch_size` | 测试、推理、ASR 评估的 batch size。 |
| `cache_partition` | bool 或 string | 无内置默认 | `true` 时总是缓存分区；若写成某个分布名，例如 `non-iid`，则只有当前 `distribution` 等于它时才启用缓存。缓存文件写到 `running_caches/`。 |
| `download` | bool | 仅 `HAR` 真正读取该值 | 当前 `HAR` 会尊重 `download`；`MNIST`、`FashionMNIST`、`EMNIST`、`CIFAR10`、`CIFAR100`、`CHMNIST`、`CINIC10`、`TinyImageNet` 的加载路径里仍然硬编码了 `download=True`。 |

### 攻击与防御选择

| 字段 | 类型 / 可选值 | 默认值 / 生效规则 | 说明 |
| --- | --- | --- | --- |
| `attack` | 见下方“攻击配置” | 无内置默认 | 当前实验使用的攻击。 |
| `defense` | 见下方“防御配置” | 无内置默认 | 当前实验使用的聚合 / 防御方法。 |
| `attack_params` | mapping | 缺省时按“共享目录 -> 实现默认值”回退 | 当前 `attack` 的专属参数。可以只写部分键。 |
| `defense_params` | mapping | 缺省时按“共享目录 -> 实现默认值”回退 | 当前 `defense` 的专属参数。可以只写部分键。 |
| `attacks` | list[dict] | 主 YAML 缺省时自动注入共享目录 | 攻击候选集。每项格式是 `{attack: 名称, attack_params: {...}}`。 |
| `defenses` | list[dict] | 主 YAML 缺省时自动注入共享目录 | 防御候选集。每项格式是 `{defense: 名称, defense_params: {...}}`。 |

### 运行时、日志与 Profiling

| 字段 | 类型 / 可选值 | 默认值 / 生效规则 | 说明 |
| --- | --- | --- | --- |
| `gpu_idx` | list[int] | 无内置默认 | CUDA 可用时只使用 `gpu_idx[0]` 作为主卡；MPS/CPU 下忽略。 |
| `num_workers` | int | 无内置默认 | DataLoader 的 worker 数。 |
| `log_stream` | bool | `true` | 是否把日志同步打印到 stdout。 |
| `log_color` | `auto`, bool | `auto` | `auto` 仅在 TTY 上输出彩色 ANSI 日志；文件日志始终纯文本。 |
| `record_time` | bool | `false` | 记录每轮训练时间拆解、GPU 利用率与汇总 JSON。 |
| `gpu_sample_interval_ms` | int | `100` | `record_time=true` 时的 GPU 采样间隔。 |
| `torch_profile` | bool | `false` | 启用 `torch.profiler`。 |
| `torch_profile_wait` | int | `0` | `torch.profiler.schedule(wait=...)`。 |
| `torch_profile_warmup` | int | `1` | `torch.profiler.schedule(warmup=...)`。 |
| `torch_profile_active` | int | `3` | `torch.profiler.schedule(active=...)`。 |
| `torch_profile_repeat` | int | `1` | `torch.profiler.schedule(repeat=...)`。 |
| `torch_profile_record_shapes` | bool | `true` | profiler 是否记录 shape。 |
| `torch_profile_memory` | bool | `true` | profiler 是否记录显存 / 内存。 |
| `torch_profile_with_stack` | bool | `false` | profiler 是否记录 Python stack。 |

### 兼容保留字段

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `root` | 已保留，但当前数据加载代码未使用 | `load_data()` 仍然固定读写 `./data`。 |
| `aug` | 已保留，但当前未接入 | 代码里没有对应的数据增强开关分支。 |
| `partition_visualization` | 已保留，但当前未接入 | 标签分布可视化代码仍被注释掉。 |

### Profiling 输出位置

- `record_time: true` 时，会在训练日志里输出每轮的 `data`、`fwd_bwd`、`gpu_compute`、`opt_step`、`aggregate`、`logging` 等阶段耗时，并写出 `logs/perf_logs/...json`。
- `torch_profile: true` 时，会在 `logs/torch_traces/...` 输出 trace，并在训练末尾打印 profiler 摘要。

## 数据集与模型可选值

### 数据集

当前 CLI / `load_data()` 真正支持的 `dataset`：

- `MNIST`
- `FashionMNIST`
- `EMNIST`
- `CIFAR10`
- `CIFAR100`
- `CINIC10`
- `CHMNIST`
- `TinyImageNet`
- `HAR`

`configs/presets/datasets.yaml` 里另外还有 `FEMNIST` 条目，但当前 CLI 选项和 `load_data()` 都没有把 `FEMNIST` 接上。

### 模型

当前注册的 `model`：

- `simplecnn`
- `fcn`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `lr`
- `mlp`
- `lenet`
- `lenet_bn`
- `vgg11`
- `vgg11_bn`
- `vgg13`
- `vgg13_bn`
- `vgg16`
- `vgg16_bn`
- `vgg19`
- `vgg19_bn`

模型族与约束：

| 模型族 | 成员 | 约束 |
| --- | --- | --- |
| 灰度向量模型 | `lr` | 仅支持单通道，输入依赖 `num_dims`。 |
| 向量模型 | `mlp`, `fcn` | 依赖 `num_features`，当前主要用于 `HAR`。 |
| 自适应卷积模型 | `lenet`, `lenet_bn` | 根据 `num_channels` 自适应。 |
| RGB 模型 | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn` | 仅支持三通道输入。 |
| 通用小模型 | `simplecnn` | 需要 `num_channels` 和 `num_dims`。 |

### 其他离散可选值

- `algorithm`: `FedSGD`, `FedAvg`, `FedOpt`
- `optimizer`: `SGD`, `Adam`
- `lr_scheduler`: `MultiStepLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR`
- `distribution`: `iid`, `class-imbalanced_iid`, `non-iid`, `pat`, `imbalanced_pat`

### 数据预处理补充说明

- `MNIST`、`FashionMNIST`、`EMNIST` 搭配 `lenet`、`lenet_bn`、`lr` 时，会被 resize 到 `32x32`，并在运行时设置 `num_dims = 32`。
- `CIFAR10`、`CIFAR100`、`TinyImageNet` 的训练集默认带随机裁剪和水平翻转；测试集只做标准化。
- `CINIC10` 只做 `ToTensor + Normalize`。
- `CHMNIST` 会使用 `num_dims`；如果目录里没有写，代码会回退到 `150`。
- `HAR` 不走 torchvision transform。

## 数据集目录：`configs/presets/datasets.yaml`

这个文件负责提供数据集元数据。`single_preprocess()` 会把当前数据集条目里的所有键直接写入运行时 `args`。

通用字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `num_training_sample` | int | 训练样本数。 |
| `num_channels` | int | 输入通道数。 |
| `num_classes` | int | 类别数。 |
| `mean` | YAML list[float] | 数据集均值；运行时会转为 tuple。 |
| `std` | YAML list[float] | 数据集标准差；运行时会转为 tuple。 |
| `num_dims` | int | 图像边长或占位尺寸；只有部分数据集需要。 |
| `num_features` | int | 向量模型输入维度；目前 `HAR` 需要。 |

当前目录里的数据集条目：

| 数据集 | 额外字段 / 特殊点 | 说明 |
| --- | --- | --- |
| `MNIST` | 无额外字段 | 灰度图。 |
| `FashionMNIST` | 无额外字段 | 灰度图。 |
| `EMNIST` | 无额外字段 | 当前固定使用 digits split。 |
| `FEMNIST` | 目录中存在，但当前未接入 loader | 仅元数据占位。 |
| `CIFAR10` | 无额外字段 | RGB 图像。 |
| `CIFAR100` | 无额外字段 | RGB 图像。 |
| `CINIC10` | 无额外字段 | RGB 图像。 |
| `CHMNIST` | `num_dims` | 当前目录默认 `150`。 |
| `TinyImageNet` | `num_dims` | 当前目录默认 `64`。 |
| `HAR` | `num_dims`, `num_features` | 非图像数据；`fcn` / `mlp` 依赖 `num_features`。 |

## 攻击配置

### 攻击名称总览

- 禁用攻击：`NoAttack`
- 纯模型投毒：`ALIE`、`FangAttack`、`Gaussian`、`IPM`、`Mimic`、`MinMax`、`MinSum`、`SignFlipping`、`ThreeDFed`
- 纯数据投毒：`LabelFlipping`、`BadNets`、`BadNets_image`、`DBA`
- 混合攻击（同时含数据投毒与模型投毒能力）：`AlterMin`、`EdgeCase`、`ModelReplacement`、`Neurotoxin`

### `attack_params` 常用字段

#### 数据投毒常用字段

| 字段 | 常见取值 | 说明 |
| --- | --- | --- |
| `attack_model` | `all2one`, `all2all`, `targeted`, `random` | 具体支持哪些取值由攻击实现决定。 |
| `poisoning_ratio` | 0 到 1 的 float | 训练样本中被投毒的比例。 |
| `target_label` | int | 目标标签。 |
| `source_label` | int | 源标签；`targeted` 模式常用。 |
| `trigger_size` | int | 像素触发器边长。 |
| `trigger_path` | path string | 图片触发器路径；可写绝对路径，也可写相对仓库根目录或 `flpoison/` 的相对路径。 |
| `attack_strategy` | `continuous`, `single-shot`, `fixed-frequency` | 投毒轮次策略。 |
| `single_epoch` | int | `single-shot` 下投毒的唯一轮次。 |
| `poison_frequency` | float 或 int | `fixed-frequency` 下的投毒频率；支持比例或绝对轮数。 |
| `attack_start_epoch` | int 或 null | 从指定全局轮开始投毒；仅实现中显式使用它的攻击会读取该值。 |

#### 模型投毒常用字段

| 字段 | 常见取值 | 说明 |
| --- | --- | --- |
| `scaling_factor` | float | 更新缩放系数。 |
| `alpha` | float | `ModelReplacement` 中分类损失与异常损失的权重。 |
| `gamma_init` | float | `MinMax` / `MinSum` 搜索起点。 |
| `stop_threshold` | float | 二分搜索或停止阈值。 |
| `noise_mean` | float | 高斯噪声均值。 |
| `noise_std` | float | 高斯噪声标准差。 |
| `z_max` | float 或 null | `ALIE` 使用的上界参数。 |
| `choice` | int | `Mimic` 模仿的 benign 客户端索引。 |

#### 其他攻击专用字段

| 字段 | 归属攻击 | 说明 |
| --- | --- | --- |
| `trigger_factor` | `DBA` | 三元组 `[trigger_size, gap, shift]`。 |
| `epsilon` | `EdgeCase` | PGD 投影半径。 |
| `PGD_attack` | `EdgeCase` | 是否执行 PGD 投影。 |
| `projection_type` | `EdgeCase` | `l_2` 或 `l_inf`。 |
| `l2_proj_frequency` | `EdgeCase` | `l_2` 投影频率。 |
| `scaling_attack` | `EdgeCase` | 是否叠加缩放攻击。 |
| `num_sample` | `Neurotoxin` | 用于构造掩码的采样数量。 |
| `topk_ratio` | `Neurotoxin` | Top-k 掩码比例。 |
| `norm_threshold` | `Neurotoxin` | 范数阈值。 |
| `poisoned_sample_cnt` | `AlterMin` | 每轮投毒样本数。 |
| `boosting_factor` | `AlterMin` | 更新增强系数。 |
| `rho` | `AlterMin` | 正则项系数。 |
| `benign_epochs` | `AlterMin` | 预热 benign 训练轮数。 |
| `malicous_epochs` | `AlterMin` | 恶意训练轮数。注意代码中的拼写就是 `malicous_epochs`。 |

### 各攻击默认参数

- `NoAttack`：无参数。建议同时设置 `num_adv: 0`。
- `SignFlipping`：无参数。
- `ThreeDFed`：代码顶部明确标注“unfinished yet”，当前没有稳定配置接口，不建议写入正式实验手册或批量实验。
- `ALIE`：默认 `z_max: null`、`attack_start_epoch: null`。适合延迟启动的模型投毒场景。
- `Gaussian`：默认 `noise_mean: 0`、`noise_std: 1`。向更新直接加噪。
- `IPM`：默认 `scaling_factor: 0.5`、`attack_start_epoch: null`。
- `Mimic`：默认 `choice: 0`，表示模仿第 0 个 benign 更新。
- `FangAttack`：默认 `stop_threshold: 1.0e-5`。
- `MinMax`：默认 `gamma_init: 10`、`stop_threshold: 1.0e-5`。
- `MinSum`：默认 `gamma_init: 10`、`stop_threshold: 1.0e-5`。
- `BadNets`：默认 `trigger_size: 10`、`attack_model: all2one`、`poisoning_ratio: 0.32`、`target_label: 7`、`source_label: 1`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`、`attack_start_epoch: null`。
- `BadNets_image`：默认 `trigger_path: ./attackers/triggers/trigger_white.png`、`trigger_size: 5`、`attack_model: all2one`、`poisoning_ratio: 0.32`、`target_label: 7`、`source_label: 1`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`。
- `LabelFlipping`：默认 `attack_model: targeted`、`source_label: 3`、`target_label: 7`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`、`poisoning_ratio: 0.32`。`targeted` 模式主要翻转 `source_label`，`poisoning_ratio` 不是主控制量。
- `ModelReplacement`：默认 `scaling_factor: 20`、`alpha: 0.5`、`attack_model: all2one`、`poisoning_ratio: 0.32`、`target_label: 7`、`source_label: 2`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`。
- `DBA`：默认 `attack_model: all2one`、`scaling_factor: 100`、`trigger_factor: [14, 2, 0]`、`poisoning_ratio: 0.32`、`source_label: 2`、`target_label: 7`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`、`attack_start_epoch: null`。
- `EdgeCase`：默认 `poisoning_ratio: 0.5`、`epsilon: 0.25`、`PGD_attack: true`、`projection_type: l_2`、`l2_proj_frequency: 1`、`scaling_attack: true`、`scaling_factor: 50`、`target_label: 1`。
- `Neurotoxin`：默认 `num_sample: 64`、`topk_ratio: 0.1`、`norm_threshold: 0.2`、`attack_model: all2one`、`poisoning_ratio: 0.32`、`target_label: 6`、`source_label: 1`、`attack_strategy: continuous`、`single_epoch: 0`、`poison_frequency: 5`。
- `AlterMin`：默认 `attack_model: targeted`、`source_label: 3`、`target_label: 7`、`poisoned_sample_cnt: 1`、`boosting_factor: 2`、`rho: 1.0e-4`、`benign_epochs: 10`、`malicous_epochs: 5`。

## 防御配置

### 防御名称总览

当前注册的 `defense`：

- `Auror`
- `Bucketing`
- `Krum`
- `Bulyan`
- `CenteredClipping`
- `CRFL`
- `DeepSight`
- `DnC`
- `FLAME`
- `FLDetector`
- `FLTrust`
- `FoolsGold`
- `LASA`
- `Mean`
- `Median`
- `MultiKrum`
- `NormClipping`
- `RFA`
- `SignGuard`
- `SimpleClustering`
- `TriGuardFL`
- `TrimmedMean`

### `defense_params` 常用字段

| 字段 | 常见取值 | 说明 |
| --- | --- | --- |
| `enable_check` | bool | `Krum` / `Bulyan` / `MultiKrum` 的额外检查开关。 |
| `beta` | float | `TrimmedMean` 的截断比例。 |
| `avg_percentage` | float | `MultiKrum` 参与平均的比例。 |
| `num_iters` | int | `RFA`、`CenteredClipping`、`DnC` 等迭代次数。 |
| `epsilon` | float | `RFA`、`DeepSight`、`FoolsGold` 的数值稳定项。 |
| `bucket_size` | int | `Bucketing` 中每个桶的大小。 |
| `selected_aggregator` | 任意已注册聚合器名 | `Bucketing` 桶内使用的聚合器。 |
| `norm_threshold` | float | `CenteredClipping`、`NormClipping`、`CRFL` 等的剪裁阈值。 |
| `noise_mean` | float | 加噪均值。 |
| `noise_std` | float | 加噪标准差。 |
| `weakDP` | bool | `NormClipping` 中是否额外做弱 DP 噪声。 |
| `num_sample` | int | `FLTrust` 用于服务器根数据集采样的样本数。 |
| `subsample_frac` | float | `DnC` 中参数子采样比例。 |
| `fliter_frac` | float | `DnC` 过滤攻击者比例。注意代码中的拼写就是 `fliter_frac`。 |
| `window_size` | int | `FLDetector` 的滑动窗口长度。 |
| `start_epoch` | int | `FLDetector` 开始检测的轮次。 |
| `lower_bound` | float | `SignGuard` 范数筛选下界，按 median norm 的倍数解释。 |
| `upper_bound` | float | `SignGuard` 范数筛选上界。 |
| `selection_fraction` | float | `SignGuard` 随机选坐标比例。 |
| `clustering` | `MeanShift`, `DBSCAN`, `KMeans` 或 `MeanShift`, `DBSCAN` | `SignGuard` 支持三种；`SimpleClustering` 只支持 `MeanShift`、`DBSCAN`。 |
| `random_seed` | int | `SignGuard` 的聚类随机种子。 |
| `indicative_threshold` | float | `Auror` 指示器阈值。 |
| `indicative_find_epoch` | int | `Auror` 找指示器的轮次。 |
| `num_seeds` | int | `DeepSight` 随机噪声数据集数量。 |
| `threshold_factor` | float | `DeepSight` 计算 NEUP 阈值时的比例因子。 |
| `num_samples` | int | `DeepSight` 随机噪声样本数。 |
| `tau` | float | `DeepSight` 中 cluster 被视作 benign 的阈值。 |
| `norm_bound` | float | `LASA` 范数边界。 |
| `sign_bound` | float | `LASA` 符号边界。 |
| `sparsity` | float | `LASA` 稀疏率。 |
| `gamma` | float | `FLAME` 的噪声尺度参数。 |

### `TriGuardFL` 专用参数

`TriGuardFL` 的参数明显比其他防御多，建议单独维护：

| 字段 | 默认值 | 可选值 / 含义 |
| --- | --- | --- |
| `cos_threshold` | `0.0` | 原始阈值法时使用的余弦阈值。 |
| `cos_filter_method` | `mad` | `threshold` 或 `mad`。 |
| `cos_significance` | `0.02` | `mad` 余弦过滤时的一侧显著性阈值。 |
| `norm_filter_method` | `mad` | `none`、`mad`、`percentile`。 |
| `norm_filter_k` | `3.5` | `mad` 范数过滤的倍数。 |
| `norm_filter_percentile` | `0.95` | `percentile` 范数过滤的分位数。 |
| `candidate_rule` | `cos_or_norm` | `cos` 或 `cos_or_norm`。 |
| `delta_clip_method` | `mad` | `mad`、`percentile`、`none`。 |
| `delta_clip_k` | `3.0` | `mad` 剪裁系数。 |
| `delta_clip_percentile` | `0.95` | `percentile` 剪裁分位数。 |
| `delta_clip_value` | `0.0` | `> 0` 时直接作为固定 clip norm。 |
| `significance` | `0.02` | 第二阶段显著性检验阈值。 |
| `discount` | `0.9` | reputation 更新中的折扣因子。 |
| `reputation_threshold` | `0.6` | 阈值 gating 的保留门槛。 |
| `epochs_phase_2` | `20` | 超过该轮次后进入 phase 2。 |
| `num_items_test` | `512` | 服务器端平衡采样的测试样本数。 |
| `eval_batch_size` | `128` | 服务器端按类评估的 batch size。 |
| `rep_alpha_inc` | `1.0` | reputation alpha 增量。 |
| `rep_beta_inc` | `1.0` | reputation beta 增量。 |
| `rep_hard_zero` | `false` | 是否对劣质客户端做硬归零。 |
| `phase2_gating` | `soft` | `soft`、`threshold`、`topk`。 |
| `phase2_topk_frac` | `0.8` | `topk` gating 下保留比例。 |
| `phase2_min_keep` | `1` | `topk` gating 下至少保留的客户端数量。 |

### 各防御默认参数

- `Mean`：无参数。
- `Median`：无参数。
- `TrimmedMean`：默认 `beta: 0.1`。
- `Krum`：默认 `enable_check: false`。
- `MultiKrum`：默认 `avg_percentage: 0.2`、`enable_check: false`。
- `Bulyan`：默认 `enable_check: false`。
- `RFA`：默认 `num_iters: 3`、`epsilon: 1.0e-6`。
- `CenteredClipping`：默认 `norm_threshold: 100`、`num_iters: 1`。
- `Bucketing`：默认 `bucket_size: 2`、`selected_aggregator: Krum`。
- `DnC`：默认 `subsample_frac: 0.2`、`num_iters: 5`、`fliter_frac: 1.0`。
- `SignGuard`：默认 `lower_bound: 0.1`、`upper_bound: 3.0`、`selection_fraction: 0.1`、`clustering: MeanShift`、`random_seed: 2`。
- `SimpleClustering`：默认 `clustering: DBSCAN`。
- `Auror`：默认 `indicative_threshold: 7.0e-5`、`indicative_find_epoch: 10`。
- `FLTrust`：默认 `num_sample: 100`。会在训练集上抽一小份 server root dataset。
- `FoolsGold`：默认 `epsilon: 1.0e-5`、`topk_ratio: 0.1`。
- `NormClipping`：默认 `weakDP: true`、`norm_threshold: 3`、`noise_mean: 0`、`noise_std: 0.002`。
- `CRFL`：默认 `norm_threshold: 3`、`noise_mean: 0`、`noise_std: 0.001`。
- `DeepSight`：默认 `num_seeds: 3`、`threshold_factor: 0.01`、`num_samples: 20000`、`tau: 0.33`、`epsilon: 1.0e-6`。
- `FLAME`：默认 `gamma: 1.2e-5`。
- `FLDetector`：默认 `window_size: 10`、`start_epoch: 50`。
- `LASA`：默认 `norm_bound: 1`、`sign_bound: 1`、`sparsity: 0.3`。
- `TriGuardFL`：默认参数见上表。

## `attacks` / `defenses` 列表格式

`attacks` 和 `defenses` 的每个元素都是一条“名称 + 默认参数”的目录项：

```yaml
attack: BadNets
defense: Krum

attacks:
  - attack: BadNets
    attack_params:
      target_label: 7
      poisoning_ratio: 0.32
  - attack: MinSum
    attack_params:
      gamma_init: 10
      stop_threshold: 1.0e-5

defenses:
  - defense: Krum
    defense_params:
      enable_check: false
  - defense: TrimmedMean
    defense_params:
      beta: 0.1
```

建议把这两个列表当作共享目录或批量实验目录使用，而不是把所有参数都堆到主配置顶层。

## CLI 覆盖规则

### 基本规则

- 绝大多数 CLI 参数都直接覆盖同名 YAML 字段。
- `attack` / `defense` 是特殊处理：如果只传了 `--attack BadNets` 而没传 `--attack_params`，运行时会尝试从 `attacks` 目录项里补 `BadNets` 的默认参数；没有的话才回退到实现默认值。
- `--attack_params` 和 `--defense_params` 需要传 Python 字典字符串，代码用 `ast.literal_eval()` 解析。
- `--log_color` 使用 `BooleanOptionalAction`，所以既支持 `--log_color`，也支持 `--no-log_color`。
- 单短横线长参数会被规范化，例如 `-experiment_id=3` 会被自动转成 `--experiment_id=3`。

### 示例

```bash
python -m flpoison \
  --config configs/FedAvg_CIFAR10_vgg19.yaml \
  --attack BadNets \
  --attack_params '{"target_label": 3, "poisoning_ratio": 0.2}' \
  --defense TrimmedMean \
  --defense_params '{"beta": 0.2}' \
  --eval_interval 5 \
  --record_time
```

### 重复实验与 seed

当你设置：

- `seed: 42`
- `num_experiments: 3`
- `experiment_id: 5`

则三次实际运行的 `(experiment_id, seed)` 分别是：

1. `(5, 47)`
2. `(6, 48)`
3. `(7, 49)`

## 维护建议

- 改模型输入维度、类别数、均值方差时，优先改 `configs/presets/datasets.yaml`。
- 改攻击或防御的默认超参时，优先同步改共享目录 `configs/presets/attacks.yaml` / `configs/presets/defenses.yaml`，否则手册和基线预设会很快漂移。
- 新增攻击器、聚合器、数据集或主配置键时，也应同步更新这份手册。
