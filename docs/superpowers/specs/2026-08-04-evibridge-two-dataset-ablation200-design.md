# EviBridge 两数据集 200 题核心消融实验设计

## 目标

在 Qasper 与 HotpotQA 上分别选取固定的 200 题样本，对 EviBridge 的八个核心组件进行单变量消融。实验用于回答各组件对答案质量、证据质量和联合质量的独立贡献，不用于继续调参。

## 实验范围

每个数据集包含完整 EviBridge 和以下八个消融变体：

1. `wo_multi_granularity_seeds`：移除多粒度 seed recall。
2. `wo_context_edges`：移除 context bridge。
3. `wo_semantic_edges`：移除 semantic bridge。
4. `wo_hierarchy_edges`：移除 hierarchy bridge。
5. `wo_typed_weights`：将 typed edge weighting 替换为非类型化权重。
6. `wo_budgeted_selector`：移除预算化 selector。
7. `wo_sufficiency_verifier`：移除充分性验证闭环。
8. `static_topk`：使用静态 top-k 路径，替代完整方法的动态证据构造流程。

不包含 citation reorder、support reranker、verifier repair 和 implicit multi-hop 四项优化组件消融。本轮不运行其他基线，也不修改 EviBridge 参数。

## 固定样本

两个数据集均使用 `seed=42`，先生成 manifest，再启动任何消融推理。八个消融和完整方法共享同一 manifest。

### Qasper 200

来源为完整 validation 1005 题。按首个 gold annotation 的答案类型进行固定分层抽样：

| 类型 | 完整集数量 | 抽样数量 |
|---|---:|---:|
| Extractive | 530 | 105 |
| Free-form | 264 | 53 |
| Yes/No | 116 | 23 |
| Unanswerable | 95 | 19 |
| 合计 | 1005 | 200 |

同一文档可以包含多个入选问题；manifest 同时记录题目 ID、文档 ID、分层类型、原始位置和随机种子。

### HotpotQA 200

来源为 fixed1000 seed42 distractor validation。按问题类型与答案类型的联合分层抽样：

| 类型 | 完整集数量 | 抽样数量 |
|---|---:|---:|
| Bridge + span | 799 | 160 |
| Comparison + span | 133 | 27 |
| Comparison + yes/no | 68 | 13 |
| 合计 | 1000 | 200 |

HotpotQA 每题对应一个文档组，因此期望覆盖 200 题、200 个文档组。

## 冻结配置与单变量原则

Qasper 消融必须继承当前完整主方法配置 `evibridge_qasper_support_pruned_seed42_100.yaml`，其完整方法后缀为 `evibridge_support_pruned`。HotpotQA 消融必须继承 `evibridge_hotpotqa_private.yaml`，其完整方法后缀为 `evibridge`。

每个消融配置只能覆盖：

- `rag_force_reprocess: false`，允许只补跑缺失输出；
- `rag.ablation_variant: <variant>`。

除上述字段外，模型、prompt、temperature、召回预算、PPR、reranker、selector、verifier、最大上下文和输出协议均继承对应数据集的冻结完整方法配置。通用的 `config/evibridge_wo_*.yaml` 不能直接用于正式实验，因为它们继承的不是两个数据集当前主表所使用的冻结配置。

## 执行顺序

采用按组件配对、数据集内顺序执行的方式：

1. Qasper 当前消融，两个推理分片。
2. Qasper 严格覆盖校验、官方评测和 bootstrap。
3. HotpotQA 同一消融，两个推理分片。
4. HotpotQA 严格覆盖校验、官方评测和 bootstrap。
5. 进入下一个消融。

同一时刻最多运行一个数据集上的一个消融，即最多两个生成模型推理分片，避免硅基流动限流并保持效率指标可比较。索引全部复用现有完整实验产物，不重建 tree、EviBridge index、BM25 或向量库。

## 覆盖校验与失败恢复

每次推理后必须执行严格校验：

- Qasper：200/200 问题完整；实际文档数与 manifest 一致；无未知或重复问题；无非法最终引用；无 verifier 状态矛盾。
- HotpotQA：200/200 问题和 200/200 文档组完整；无未知或重复问题；最终 supporting fact ID 均能映射到该题候选事实。

若覆盖不足，编排器使用同一配置自动补跑，最多三轮。因为 `rag_force_reprocess=false`，恢复轮只调用缺失题。三轮后仍不完整则停止整个 campaign，并在状态文件中记录失败方法、缺失题目和日志根因；禁止带缺失结果进入评测或主表。

## 评测指标

### Qasper

- Answer F1
- Evidence Precision
- Evidence Recall
- Evidence F1

多个 gold annotation 的 P/R/F1 继续使用正式主表规则：选择 Evidence F1 最高的同一个 annotation，F1 并列时使用数据集中的首个 annotation。

### HotpotQA

- Answer EM
- Answer F1
- SP Precision
- SP Recall
- SP F1
- Joint F1

### 统计与效率

每个消融除点估计外，还需报告：

- 单项指标的 bootstrap 95% CI；
- 相对同一 200 题完整方法的逐题 paired bootstrap 差值及 95% CI；
- 平均 token/题与秒/题；
- 严格覆盖和自动补跑次数。

完整方法直接切片复用现有逐题结果，不重新调用模型。所有 bootstrap 固定 `seed=42`、10000 次采样。

## 输出

实验产物保存在 `runs/` 下，不提交到 Git。每个数据集生成：

- 固定样本 JSON、YAML 和 manifest；
- 八个消融的逐题结果、审计报告、正式评测与 bootstrap 文件；
- 一张完整方法加八项消融的 Markdown/JSON 主表；
- 一张相对完整方法的 paired delta 表。

另生成跨数据集汇总表，以组件为行，展示：

- Qasper 的 `Δ Answer F1` 与 `Δ Evidence F1`；
- HotpotQA 的 `Δ Answer F1`、`Δ SP F1` 与 `Δ Joint F1`；
- 对应 paired bootstrap 95% CI。

## 结果解释规则

- 200 题是预先冻结的最终消融样本，看到结果后不得替换题目、修改参数或选择性重跑变体。
- 只允许恢复缺失或损坏输出；不得以结果不理想为由覆盖完整输出。
- 主要结论依据 paired delta 与置信区间，不仅依据点估计排序。
- 若某组件在两个数据集上的影响方向不同，应按任务差异如实解释，不合并成单一平均分。

## 预计成本

需要新运行 16 个消融任务，共 3200 题次；完整方法切片不产生新的模型调用。按现有 EviBridge 速度和每项两个分片估算，推理、评测和失败恢复总计约 6–10 小时，具体取决于硅基流动服务延迟。
