# CARAT Local 50h Experiment Plan

本计划基于当前本机环境和已有结果库：

- 代码仓库：`/home/cnlab-pu/Projects/FL_Poison`
- 已有结果库：`/home/cnlab-pu/Projects/Poisoning_Resilient_Federated_Learning_Playground`
- 本机资源：RTX 4090 24GB，i9-14900KF，约 62GB RAM
- 约束：不再使用 Compute Canada；不使用 TriGuardFL 结果；已有可复用结果不重复跑

## 当前结论

已有 CIFAR-100 / ResNet18 / non-iid / 20 clients / 20% malicious / 200 rounds 的 legacy 结果可复用：

- `Mean`、`NormClipping`、`MultiKrum`、`FLDetector`：`alpha in {1, 0.5}`、四个攻击、5 seeds 基本完整，可作为公开已发表 baseline 的 legacy rows。
- `CARAT` legacy 结果是 `T=1` pilot，不是最终 `T=8` 主结果；只作为 audit 或 appendix 对照，不作为最终 CARAT claim。
- `FLTrust` legacy 结果使用 `num_sample=100`，不满足 final matched-budget `num_sample=800`，主表不能直接使用。
- `paper_neurips_ablation_fang_alpha05.yaml` 已有结果完整，15/15，不需要重跑。
- `TriGuardFL` 文件存在，但本论文不使用，后续所有汇总和表格都排除。

现有 pilot 结果对比显示：`CARAT T=1` 已经在主表设置下明显高于 Mean / MultiKrum / FLDetector / legacy FLTrust，尤其 FLDetector 在 FangAttack 下明显退化。最终还需要证明 `T=8` 在相同协议下成立，并补齐 matched-budget FLTrust 与至少最小 TrimmedMean 行。

## 50 小时优先级

优先级按论文主张重要性排序。先跑高优先级，低优先级只在剩余时间允许时跑。

1. 必跑：最终 `CARAT T=8` 主表，24 runs。
   任务 ID：`6-8,15-17,24-26,33-35,42-44,51-53,60-62,69-71`

2. 必跑：matched-budget `FLTrust num_sample=800` 主表，24 runs。
   任务 ID：`3-5,12-14,21-23,30-32,39-41,48-50,57-59,66-68`

3. 最小补齐：`TrimmedMean` 每个 attack/alpha 先跑 seed42，8 runs。
   任务 ID：`0,9,18,27,36,45,54,63`

4. 有余量再补：`TrimmedMean` 另外两个 seeds，16 runs。
   任务 ID：`1-2,10-11,19-20,28-29,37-38,46-47,55-56,64-65`

5. 只在主表完成后再考虑 clean utility。优先跑 clean CARAT，再跑 clean matched FLTrust。
   Clean CARAT IDs：`6-8,15-17`
   Clean FLTrust IDs：`3-5,12-14`
   Clean Mean 可先复用 legacy NoAttack Mean 结果，不优先重跑。

6. 不跑：`alpha=0.1` stress test、TinyImageNet、CHMNIST、client-count scaling、backdoor/data-poisoning、TriGuardFL。

## 推荐执行命令

所有命令从仓库根目录运行：

```bash
cd /home/cnlab-pu/Projects/FL_Poison
```

先确认已有结果审计：

```bash
.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --match-config any \
  --format summary
```

跑最终 CARAT `T=8`：

```bash
CORE_CARAT_IDS="6-8,15-17,24-26,33-35,42-44,51-53,60-62,69-71"
.venv/bin/python exps/run_playground_incremental.py \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --ids "$CORE_CARAT_IDS" \
  --match-config exact \
  --cuda 0 \
  --min-free-mb 20000 \
  --max-util 30
```

跑 matched-budget FLTrust：

```bash
CORE_FLTRUST_IDS="3-5,12-14,21-23,30-32,39-41,48-50,57-59,66-68"
.venv/bin/python exps/run_playground_incremental.py \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --ids "$CORE_FLTRUST_IDS" \
  --match-config exact \
  --cuda 0 \
  --min-free-mb 20000 \
  --max-util 30
```

先跑最小 TrimmedMean：

```bash
TRIMMED_SEED42_IDS="0,9,18,27,36,45,54,63"
.venv/bin/python exps/run_playground_incremental.py \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --ids "$TRIMMED_SEED42_IDS" \
  --match-config exact \
  --cuda 0 \
  --min-free-mb 20000 \
  --max-util 30
```

如果还有时间，再补 TrimmedMean 其余 seeds：

```bash
TRIMMED_EXTRA_IDS="1-2,10-11,19-20,28-29,37-38,46-47,55-56,64-65"
.venv/bin/python exps/run_playground_incremental.py \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --ids "$TRIMMED_EXTRA_IDS" \
  --match-config exact \
  --cuda 0 \
  --min-free-mb 20000 \
  --max-util 30
```

如果主表完成后仍有 6-10 小时，再跑 clean CARAT：

```bash
CLEAN_CARAT_IDS="6-8,15-17"
.venv/bin/python exps/run_playground_incremental.py \
  exps/specs/CARAT/paper_neurips_clean_public.yaml \
  --ids "$CLEAN_CARAT_IDS" \
  --match-config exact \
  --cuda 0 \
  --min-free-mb 20000 \
  --max-util 30
```

## 监控和验收

每完成一个阶段后检查：

```bash
.venv/bin/python exps/playground_results.py missing \
  exps/specs/CARAT/paper_neurips_core_public.yaml \
  --match-config exact \
  --format summary
```

注意：`--match-config any` 可用于确认 legacy 结果覆盖情况，但最终 `CARAT T=8` 和 matched-budget `FLTrust` 必须用 `--match-config exact` 检查，因为旧 `CARAT T=1` 和旧 `FLTrust num_sample=100` 不满足最终协议。

如果第一批 `CARAT T=8` 单 run 超过 90 分钟，应停止后续低优先级任务，只保留 CARAT 全部 24 runs 和 FLTrust seed42；不要启动 clean、alpha=0.1 或额外 benchmark。
