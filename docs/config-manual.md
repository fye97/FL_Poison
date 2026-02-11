**Config Manual**

本文档说明 `configs/` 目录下配置文件（`{Algorithm}_{Dataset}_config.yaml` 与 `dataset_config.yaml`）中所有可用参数、选项与默认值来源。

适用范围：
- 训练/实验配置文件：`configs/FedSGD_MNIST_config.yaml` 等
- 数据集配置文件：`configs/dataset_config.yaml`

注意：
- `main.py` 会先读取 YAML，再用命令行参数覆盖（见 `global_args.py`）。
- `attack_params` 与 `defense_params` 若未显式给出，会从 `attacks` / `defenses` 列表里匹配当前 `attack` / `defense` 填充默认值。

---

**主配置文件（{Algorithm}_{Dataset}_config.yaml）字段**

| 参数 | 类型/可选值 | 说明 |
| --- | --- | --- |
| `seed` | int | 随机种子，控制可复现性。 |
| `epochs` | int | 全局训练轮数（server 轮）。 |
| `algorithm` | `FedSGD`, `FedAvg`, `FedOpt` | 联邦学习算法。 |
| `optimizer` | `SGD`, `Adam` | 客户端优化器。 |
| `momentum` | float | 仅对 `SGD` 生效。 |
| `weight_decay` | float | 优化器权重衰减。 |
| `lr_scheduler` | `MultiStepLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR` | 学习率调度器。为空则不启用。 |
| `milestones` | list[int/float] | `MultiStepLR` 的里程碑。若元素 < 1，则按 `epochs` 比例换算成整数轮数。 |
| `num_clients` | int | 总客户端数。 |
| `batch_size` | int | 客户端本地 batch size。 |
| `learning_rate` | float | 客户端本地学习率。 |
| `local_epochs` | int | 本地训练轮数。对 `FedAvg`/`FedOpt` 生效；`FedSGD` 固定为 1。 |
| `model` | 见下方“模型选项” | 模型架构名。 |
| `dataset` | 见下方“数据集选项” | 数据集名称。 |
| `distribution` | `iid`, `class-imbalanced_iid`, `non-iid`, `pat`, `imbalanced_pat` | 数据划分方式。当前代码仅实现 `iid`/`class-imbalanced_iid`/`non-iid`。 |
| `im_iid_gamma` | float | `class-imbalanced_iid` 时的指数衰减系数，越小越不均衡。 |
| `tail_cls_from` | int | 仅用于不均衡评估统计时“尾部类别”的起始标签。 |
| `dirichlet_alpha` | float | `non-iid` Dirichlet 划分的 alpha，越小越异质。 |
| `cache_partition` | bool 或字符串 | 是否缓存划分索引。`True`/`False` 或等于某个 `distribution` 名称时启用缓存。 |
| `gpu_idx` | list[int] | GPU 索引列表，`args.gpu_idx[0]` 用作主卡。 |
| `num_workers` | int | DataLoader 的 `num_workers`。 |
| `record_time` | bool | 是否记录并输出各模块耗时。 |
| `log_stream` | bool | 是否将日志输出到 stdout（适配 tqdm）。 |
| `num_adv` | float 或 int | 对抗客户端数量或比例。`<1` 视为比例，`>=1` 视为绝对数量。 |
| `attack` | 见下方“攻击选项” | 当前实验使用的攻击方法。 |
| `defense` | 见下方“防御选项” | 当前实验使用的防御/聚合器。 |
| `attacks` | list[dict] | 攻击候选集合，供 benchmark 组合实验使用。 |
| `defenses` | list[dict] | 防御候选集合，供 benchmark 组合实验使用。 |
| `attack_params` | dict | 可选。直接指定当前攻击参数。 |
| `defense_params` | dict | 可选。直接指定当前防御参数。 |
| `root` | string | 当前未被 `load_data` 使用（数据目录固定为 `./data`）。 |
| `aug` | bool | 当前未被使用（无数据增强开关）。 |
| `partition_visualization` | bool | 当前未被使用（划分可视化未接入）。 |

---

**攻击/防御列表配置格式**

示例（与当前仓库一致）：

```yaml
attack: MinSum
defense: Krum

attacks:
  - attack: MinSum
    attack_params:
      gamma_init: 10
      stop_threshold: 1.0e-5
  - attack: BadNets
    attack_params:
      attack_model: all2one
      poisoning_ratio: 0.32
      target_label: 7

defenses:
  - defense: Krum
    defense_params:
      enable_check: false
  - defense: TrimmedMean
    defense_params:
      beta: 0.1
```

说明：
- `attack`/`defense` 表示单次实验选用的方法。
- `attacks`/`defenses` 用于 `-b/--benchmark` 模式，会两两组合执行。

---

**可选值汇总**

- `algorithm`: `FedSGD`, `FedAvg`, `FedOpt`
- `optimizer`: `SGD`, `Adam`
- `model`: `simplecnn`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `lr`, `mlp`, `fcn`, `lenet`, `lenet_bn`, `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
- `dataset`: `MNIST`, `FashionMNIST`, `EMNIST`, `CIFAR10`, `CIFAR100`, `CINIC10`, `CHMNIST`, `TinyImageNet`, `HAR`
- `distribution`: `iid`, `class-imbalanced_iid`, `non-iid`, `pat`, `imbalanced_pat`
- `lr_scheduler`: `MultiStepLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR`

---

**攻击选项**

`NoAttack`, `SignFlipping`, `Gaussian`, `IPM`, `ALIE`, `MinMax`, `MinSum`, `FangAttack`, `Mimic`, `ModelReplacement`, `BadNets`, `BadNets_image`, `LabelFlipping`, `DBA`, `EdgeCase`, `Neurotoxin`, `AlterMin`, `ThreeDFed`

说明：
- `ThreeDFed` 在代码中标记为未完成（实验性）。

---

**防御选项**

`Mean`, `Median`, `Krum`, `MultiKrum`, `TrimmedMean`, `Bulyan`, `RFA`, `FLTrust`, `CenteredClipping`, `DnC`, `Bucketing`, `SignGuard`, `LASA`, `Auror`, `FoolsGold`, `NormClipping`, `CRFL`, `DeepSight`, `FLAME`, `SimpleClustering`, `FLDetector`

---

**attack_params 通用字段（数据投毒类）**

`attack_model`: `all2one`, `all2all`, `targeted`, `random`（部分攻击只支持子集）。  
`poisoning_ratio`: 0~1，投毒样本比例。  
`target_label`: 目标标签（用于 `all2one`/`targeted`）。  
`source_label`: 源标签（用于 `targeted`）。  
`trigger_size`: 像素触发器尺寸（BadNets 系列）。  
`trigger_path`: 触发器图片路径（BadNets_image）。  
`attack_strategy`: `continuous`, `single-shot`, `fixed-frequency`。  
`single_epoch`: `single-shot` 时使用的投毒轮。  
`poison_frequency`: `fixed-frequency` 时的投毒频率（可为比例或绝对轮数，按 `frac_or_int_to_int` 处理）。  
`attack_start_epoch`: 从指定全局轮数开始投毒（BadNets/DBA 等）。  

**attack_params 通用字段（模型投毒类）**

`scaling_factor`: 更新缩放系数（IPM/ModelReplacement/DBA/EdgeCase 等）。  
`gamma_init`: MinMax/MinSum 初始搜索参数。  
`stop_threshold`: MinMax/MinSum/FangAttack 的二分停止阈值。  

---

**各攻击默认参数**

`ALIE`  
`z_max: None`, `attack_start_epoch: None`

`AlterMin`  
`attack_model: targeted`, `source_label: 3`, `target_label: 7`, `poisoned_sample_cnt: 1`, `boosting_factor: 10`, `rho: 1e-4`, `benign_epochs: 10`, `malicous_epochs: 1`

`BadNets`  
`trigger_size: 5`, `attack_model: all2one`, `poisoning_ratio: 0.32`, `target_label: 6`, `source_label: 1`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`, `attack_start_epoch: None`

`BadNets_image`  
`trigger_path: ./attackers/triggers/trigger_white.png`, `trigger_size: 5`, `attack_model: all2one`, `poisoning_ratio: 0.32`, `target_label: 6`, `source_label: 1`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`

`DBA`  
`attack_model: all2one`, `scaling_factor: 100`, `trigger_factor: [8, 2, 0]`, `poisoning_ratio: 0.32`, `source_label: 2`, `target_label: 7`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`, `attack_start_epoch: None`  
`trigger_factor` 含义：`[trigger_size, gap, shift]`。

`EdgeCase`  
`poisoning_ratio: 0.8`, `epsilon: 0.25`, `PGD_attack: True`, `projection_type: l_2`, `l2_proj_frequency: 1`, `scaling_attack: True`, `scaling_factor: 50`, `target_label: 1`  
`projection_type` 可选：`l_2`, `l_inf`。

`FangAttack`  
`stop_threshold: 1.0e-5`

`Gaussian`  
`noise_mean: 0`, `noise_std: 1`

`IPM`  
`scaling_factor: 0.1`, `attack_start_epoch: None`

`LabelFlipping`  
`attack_model: targeted`, `source_label: 2`, `target_label: 7`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`, `poisoning_ratio: 0.32`

`Mimic`  
`choice: 0`

`MinMax` / `MinSum`  
`gamma_init: 10.0`, `stop_threshold: 1.0e-5`

`ModelReplacement`  
`scaling_factor: 50`, `alpha: 0.5`, `attack_model: all2one`, `poisoning_ratio: 0.32`, `target_label: 6`, `source_label: 3`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`

`Neurotoxin`  
`num_sample: 64`, `topk_ratio: 0.1`, `norm_threshold: 0.2`, `attack_model: all2one`, `poisoning_ratio: 0.32`, `target_label: 6`, `source_label: 1`, `attack_strategy: continuous`, `single_epoch: 0`, `poison_frequency: 5`

`SignFlipping`  
无可配置参数。

`ThreeDFed`  
暂无稳定配置（代码标记为未完成）。

---

**各防御默认参数**

`Auror`  
`indicative_threshold: 0.002`, `indicative_find_epoch: 10`

`Bucketing`  
`bucket_size: 2`, `selected_aggregator: Krum`  
`selected_aggregator` 可选为任意已注册聚合器名称（见“防御选项”列表）。

`Bulyan`  
`enable_check: False`

`CenteredClipping`  
`norm_threshold: 100`, `num_iters: 1`

`CRFL`  
`norm_threshold: 3`, `noise_mean: 0`, `noise_std: 0.001`

`DeepSight`  
`num_seeds: 3`, `threshold_factor: 0.01`, `num_samples: 20000`, `tau: 0.33`, `epsilon: 1.0e-6`

`DnC`  
`subsample_frac: 0.2`, `num_iters: 5`, `fliter_frac: 1.0`

`FLAME`  
`gamma: 1.2e-5`

`FLDetector`  
`window_size: 10`, `start_epoch: 50`

`FLTrust`  
`num_sample: 100`

`FoolsGold`  
`epsilon: 1.0e-6`, `topk_ratio: 0.1`

`Krum`  
`enable_check: False`

`LASA`  
`norm_bound: 2`, `sign_bound: 1`, `sparsity: 0.3`

`MultiKrum`  
`avg_percentage: 0.2`, `enable_check: False`

`NormClipping`  
`weakDP: False`, `norm_threshold: 3`, `noise_mean: 0`, `noise_std: 0.002`

`RFA`  
`num_iters: 3`, `epsilon: 1.0e-6`

`SignGuard`  
`lower_bound: 0.1`, `upper_bound: 3.0`, `selection_fraction: 0.1`, `clustering: DBSCAN`, `random_seed: 0`  
`clustering` 可选：`MeanShift`, `DBSCAN`, `KMeans`。

`SimpleClustering`  
`clustering: DBSCAN`  
`clustering` 可选：`MeanShift`, `DBSCAN`。

`TrimmedMean`  
`beta: 0.1`

`Mean` / `Median`  
无可配置参数。

---

**dataset_config.yaml 字段**

该文件为每个数据集提供统计信息与维度配置，运行时会被 `single_preprocess` 加载并写入 `args`。

通用字段：
- `num_training_sample`: 训练样本数
- `num_channels`: 通道数
- `num_classes`: 类别数
- `mean`: 数据集均值（tuple 形式）
- `std`: 数据集标准差（tuple 形式）
- `num_dims`: 图像尺寸（仅部分数据集有，如 CHMNIST/TinyImageNet）

---

**备注**

- `NoAttack` 时建议设置 `num_adv: 0`，避免产生“攻击者客户端”。
- `distribution` 的 `pat`/`imbalanced_pat` 尚未在数据划分逻辑中实现。
- `root`/`aug`/`partition_visualization` 当前未接入到实际流程，仅存在于配置文件中。
- `lr_scheduler` 的内部超参固定在代码里：
`StepLR(step_size=80, gamma=0.5)`，`ExponentialLR(gamma=0.99)`，`CosineAnnealingLR(T_max=epochs*local_epochs)`。
- `dataset_config.yaml` 中存在 `FEMNIST` 条目，但 `load_data` 尚未支持该数据集。
