# KG²RAG 闭语料双数据集基线设计

## 1. 目标

在不进行任务特定训练、不使用标准答案或标准证据的条件下，基于官方 KG²RAG 源码完成 Qasper validation full（1005 题）与 HotpotQA distractor fixed-1000 的统一对比实验。实验采用闭语料 OpenIE：Qasper 仅使用当前论文全文，HotpotQA 仅使用当前问题给出的 distractor context。

论文内部方法名为 `kg2rag_official`，表格显示为 `KG²RAG`。现有 EviBridge、HippoRAG 2 和其他基线的配置、结果及运行产物不作修改。

## 2. 实现边界

官方源码固定到可核验 revision，并放置于已被 Git 忽略的 `runs/third_party/KG2RAG`。主仓库只增加适配器、公开配置、测试和 source lock，不复制或修改后提交 GPL-3.0 官方源码。

适配器采用与 HippoRAG 2 相同的隔离式三阶段结构：

1. `prepare`：从现有 dataset config、manifest 和原始映射导出闭语料 scope。
2. `run_official`：在独立环境中调用官方 KG²RAG 的图扩展与上下文组织机制。
3. `finalize`：在主项目环境中执行统一答案生成、支持证据校验、成本汇总和评测格式转换。

不修改 `main.py`、RAG discriminator 或现有策略，避免第三方依赖影响主工程。

## 3. 数据与构图

### 3.1 Qasper

- 每篇论文构成一个 scope，同一论文的多个问题共享索引。
- chunk 使用当前统一实验中的 paragraph-level evidence 单元。
- section path 可以添加到检索文本，但评测和证据提交仍使用未添加标题的原始段落。
- Qwen/Qwen3-VL-8B-Instruct 以 temperature 0 从每个段落抽取事实三元组。
- 每条三元组保存 `(head, relation, tail, paragraph_id)`，并保留原始段落映射。
- 不进行固定长度二次切分，不使用标准答案、标准证据或问题文本构图。

### 3.2 HotpotQA

- 每道问题构成一个独立 scope。
- scope 仅包含该问题提供的十个 distractor documents，不访问完整 Wikipedia。
- 每个标题对应一个 chunk，metadata 保存标题及完整的 `(title, sentence_id, sentence_text)` 映射。
- Qwen/Qwen3-VL-8B-Instruct 以 temperature 0 从每个 chunk 抽取三元组。
- 不使用 gold supporting facts 构图、实体合并、扩展、排序或回退。

### 3.3 实体与三元组规范化

仅执行确定性的空白、Unicode、大小写比较键和标点边界规范化；输出中保留原始实体字符串。相同规范化实体连接其来源三元组和 chunk，不使用 LLM 进行查询相关实体合并，避免引入额外方法模块。

OpenIE 解析失败最多重试两次；仍失败时标记 scope 失败，不静默生成空图。若某个合法 scope 经抽取后确实没有三元组，则记录 `dense_fallback`，使用语义检索结果继续，但不得读取 gold evidence。

## 4. 在线检索与组织

在线阶段保留官方 KG²RAG 的三个核心环节：

1. **语义种子检索**：使用 Qwen/Qwen3-Embedding-0.6B 计算 query 与 chunk 的余弦相似度，选择前 10 个 seed chunks。
2. **图引导扩展**：从种子关联三元组出发，在闭语料 KG 上执行一跳 BFS，读出扩展子图关联的 chunks。
3. **KG 上下文组织**：以 chunk-query 相似度作为边权，将扩展子图转为无向加权图；对每个连通分量构造最大生成树；以最高权边为根进行 DFS，形成 chunk 序列；使用 BAAI/bge-reranker-v2-m3 根据三元组表示重排连通分量。

生成上下文严格限制为最多 10 个 chunk 和 4000 tokens，不能从中间截断单个 chunk。完整有序候选列表至少保留 20 个可用 chunk，用于 Recall@1/2/3/5/10/20；最终答案仅使用预算内前 10 个 chunk。

最终生成统一使用 Qwen/Qwen3-VL-8B-Instruct、temperature 0.1。Qasper 支持证据最多 4 段，HotpotQA 支持事实最多 4 个合法 title--sentence pair。非法 ID 被拒绝；全部非法时只允许从 KG²RAG 已检索的最高排名 chunk 中进行受控回退。

## 5. 公平性与泄漏控制

- 两个数据集沿用现有锁定 manifest、问题顺序和数据哈希。
- 所有方法共享生成模型、embedding、reranker、上下文预算、输出规范和评测器。
- 构图阶段不接收问题、答案或证据标注。
- 在线阶段只接收问题与闭语料索引，不读取答案或证据标注。
- Qasper 的证据 ID 必须精确映射回原始 paragraph ID。
- HotpotQA supporting facts 必须精确映射回当前题目的合法 title--sentence pair。
- 未知、重复或跨 scope 的 source ID 视为运行错误，不用文本模糊匹配静默修复。

## 6. 输出与成本

结果写入现有评测可读取的 `eval_<dataset>_kg2rag_official` 目录。每题至少保存：

- seed chunks、相似度和排名；
- 扩展三元组、扩展 chunk 和 BFS 路径；
- 连通分量、最大生成树边、DFS 顺序和 reranker 分数；
- 最终有序候选、生成上下文和支持证据；
- citation validation、fallback 状态、在线 token 与耗时。

主实验统计：

- Qasper：Answer F1、Evidence Precision/Recall/F1、Recall@1/2/3/5/10/20。
- HotpotQA：Answer EM/F1、SP Precision/Recall/F1、Joint F1。
- 两个数据集：Tokens/Question、Seconds/Question。

在线效率包括 query embedding、图扩展、MST/DFS、reranker 和最终生成。离线 OpenIE tokens、构图墙钟时间、实体数、三元组数、边数和索引大小单独报告，不计入主效率表。

## 7. 缓存、恢复与错误处理

索引缓存键至少包含 corpus SHA、公开配置 SHA、官方源码 revision、OpenIE 模型和 embedding 模型。任一字段变化时创建新 run，不覆盖或错误复用旧索引。

已完成且校验通过的 scope 和 query 在恢复运行时跳过。缺失结果、重复问题 ID、非法证据 ID、数据哈希变化或 scope 失败会使完整实验失败。最终要求 Qasper 1005/1005 和 HotpotQA 1000/1000 预测覆盖。

## 8. 验证与运行顺序

自动测试覆盖数据映射、闭语料边界、三元组来源、图扩展、最大生成树组织、预算约束、证据 ID 校验、gold-free fallback、成本分离和断点恢复。

运行按以下阶段推进：

1. 官方示例和模型服务 smoke test。
2. Qasper 两篇论文、HotpotQA 两个 bridge 加两个 comparison 问题的适配 smoke test。
3. 核验证据映射、闭语料边界、输出字段和成本统计。
4. 构建 Qasper 281 个论文 scope 和 HotpotQA 1000 个问题 scope。
5. 完成两个数据集的统一推理、评测、Recall@k 和效率统计。

正式运行不根据 smoke test 的正确率调整参数。若 smoke test 暴露工程错误，仅修复接口、解析、恢复或映射问题。

## 9. 完成标准

- 官方源码 revision、许可证和配置均可核验。
- Qasper 完成 1005/1005，HotpotQA 完成 1000/1000。
- 所有证据均可追溯到原始 paragraph 或 title--sentence。
- Recall@k 单调不减。
- 主表指标、Recall@k 和在线成本来自同一次完整运行。
- 离线与在线成本没有混淆。
- 现有 EviBridge、HippoRAG 2 和评测测试无回归。
- 本轮只生成实验结果，不自动修改论文正文或表格。
