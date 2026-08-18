# CoSE-RAG 闭环行为与检索轨迹分析设计

## 目标

在不改变现有主实验、效率实验、消融实验和参数敏感性结果的前提下，基于已完成的 CoSE-RAG 双数据集主实验日志，新增：

1. Qasper validation full（281 篇论文、1005 个问题）与 HotpotQA fixed-1000 的闭环检索行为统计；
2. 一个真实 Qasper 案例和一个真实 HotpotQA Bridge 案例的可审计检索轨迹；
3. 对应的论文表格与分析文字。

不重新调用模型，不生成首轮答案，不报告无法由现有日志得到的首轮 Answer F1 或 Joint F1。

## 唯一论文基线与输出位置

- 输入论文：`D:/Else/Study/godot/素材/压缩包/CoSE-RAG_updated_efficiency.tex`
- 修改后论文：原路径覆盖更新，同时在修改前保留同目录备份。
- 审计输出均保存到 `D:/Else/Study/godot/素材/压缩包/`：
  - `closed_loop_query_logs.jsonl`
  - `closed_loop_behavior_summary.csv`
  - `retrieval_trajectory_cases.json`

## 数据来源

### Qasper

- 工作目录：`runs/qasper_evibridge/work_validation_full1005_unified_0417263dd/`
- CoSE-RAG 主实验：`eval_qasper_evibridge_support_pruned`
- BM25+Reranker 对照：`eval_qasper_bm25_rerank`
- 预期覆盖：1005/1005。

### HotpotQA

- 工作目录：`runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/`
- CoSE-RAG 主实验：精确目录名 `eval_hotpotqa_evibridge`，排除所有消融目录。
- BM25+Reranker 对照：`eval_hotpotqa_bm25_rerank`
- 预期覆盖：1000/1000。

每题以 `retrieval_res.json` 的 `iterations` 为闭环轮次记录，以 `result.json`、数据清单和索引映射补全问题标识、标准答案、标准证据及原文位置。标准答案和标准证据仅在检索结束后参与离线评测和案例筛选。

## 证据集合与计数定义

- 第 1 轮集合 `C1`：`iterations[0].selected` 中属于最终原子证据类型的节点。
- 最终集合 `C_final`：最后一个 iteration 的 `selected` 中属于最终原子证据类型的节点。
- 辅助节点：patch、entity、summary、title 等仅用于桥接而不允许作为最终支持证据的节点。
- 新增原子证据：`C_final - C1`；只允许来自第 2 轮候选，并且必须能经来源映射回溯到 Qasper 段落或 HotpotQA title--sentence pair。
- 上下文 token 数：对每轮全部 selected 节点文本使用主方法预算选择器的 `_token_cost` 计数函数求和，保持与 `B_tok=4000` 的执行口径一致。
- 实际检索轮数：`len(iterations)`，必须属于 `{1, 2}`。

## 闭环行为统计

逐题输出至少包含：数据集、问题 ID、`y1`、是否触发第 2 轮、`y2`、最终充分性、实际轮数、终止原因、首轮/最终原子块数、首轮/最终辅助节点数、首轮/最终 token 数、新增原子块数与标识、缺失需求、建议桥类型、受控动作、首轮/最终证据指标和来源映射检查结果。

聚合表分别报告：

- 首轮直接接受率；
- 第二轮触发率；
- 扩展后接受率（仅第二轮子集）；
- 最终仍不充分比例；
- 平均检索轮数；
- 平均新增原子块数（仅第二轮子集）；
- 第二轮子集上的首轮、最终及绝对变化：Qasper Evidence F1、HotpotQA SP F1。

CSV 同时保存分子、分母、小数比例和论文显示值。论文表不进行最优/次优标记。

## 证据指标协议

- Qasper：依据现有 Qasper 官方证据匹配协议，将每轮原子证据映射为 paragraph-level evidence；对多组 gold evidence 使用与主评测一致的最大 Evidence F1 对齐组，并由同一组产生 Precision、Recall、F1。
- HotpotQA：将每轮原子证据映射为合法 title--sentence pair，使用现有 HotpotQA Supporting Fact Precision、Recall、F1 计算逻辑。
- 只在实际触发第 2 轮的问题上比较第 1 轮和最终轮。
- 现有日志没有第 1 轮答案，因此论文明确不报告首轮 Answer F1 或 Joint F1。

## 一致性和失败处理

分析脚本必须在写出论文数据前验证：

1. Qasper 与 HotpotQA 分别恰好覆盖 1005 和 1000 个唯一问题；
2. `first_accept + second_round_trigger = N`；
3. `expanded_accept <= second_round_trigger`；
4. 最终不充分问题与非充分终止原因计数一致；
5. 所有问题轮数不超过 2；
6. 每个新增原子块属于第 2 轮候选，并能映射到原文；
7. 每个案例中的问题、答案、证据文本、证据 ID 和桥接边均来自实际数据和日志；
8. 表格数值由 JSONL/CSV 自动生成，不手工录入计算结果。

任何一项失败时停止生成最终论文片段并输出明确错误；不得跳过异常题目。

## 案例筛选

### Qasper 候选

- `y1=0` 且执行第 2 轮；
- gold evidence 涉及多个证据块，优先跨章节；
- 第 2 轮新增至少一个匹配 gold 或直接支持答案的原子块；
- 最终 Evidence F1 高于首轮；
- 最终答案按现有 Qasper token F1 计算达到 `0.8`，避免把答案错误的轨迹作为正向案例；
- 至少一个第 2 轮新增且匹配 gold 的原子块实际进入最终 `supporting_block_ids`，确保新增证据不仅被召回，而且被最终答案引用；
- 最终答案正确或最终证据覆盖更完整。

### HotpotQA 候选

- 问题类型为 Bridge；
- `y1=0` 且执行第 2 轮；
- 第 2 轮补充多跳链所需的中间或目标支持事实；
- 最终 SP F1 高于首轮；
- 最终答案正确或证据链更完整。

对各严格候选集合计算主要证据指标增益。HotpotQA 仍选择最接近本数据集严格候选增益中位数的样本；Qasper 先从严格候选中保留最终 Answer F1 不低于 `0.8` 且新增 gold 匹配块进入最终支持集的成功案例，再选择其证据增益最接近原始严格候选中位数的样本。这样既排除答案错误或新增证据未被使用的轨迹，也不因二次过滤而偏向增益最大的极端案例。距离并列时按 `question_id` 升序选择。JSON 保存严格候选数、成功候选数、原始候选中位数、所选样本增益、Answer F1、新增且被引用的 gold 证据 ID、筛选条件命中情况及选择理由。若没有完全满足者，记录缺失条件并从最接近条件的真实候选中按同一中位数规则选择。

每个案例合并同一问题的 BM25+Reranker 真实结果，展示其实际关键证据和答案；若基线正确，原样报告。

## 论文修改

仅在消融实验分析之后、参数敏感性分析之前插入：

1. `\subsection{闭环检索行为分析}`
   - 紧凑双数据集统计表，标签 `tab:closed_loop_behavior`；
   - 两至三段克制的描述性分析；
   - 表注说明子集统计口径和 gold 仅用于事后评测。
2. `\subsection{检索轨迹案例分析}`
   - 一个跨栏 `table*`，标签 `tab:retrieval_trajectory_cases`；
   - 两个案例按 Question/Gold、BM25+Reranker、Round 1、Diagnosis、Round 2、Final Output 展示；
   - 证据文本仅保留理解轨迹必需的短摘录；
   - 两至三段案例分析和基于真实失败样本的简短失败模式说明。

不修改主表、效率表、消融表、参数敏感性图、既有数值或无关正文。只有统计结果实际支持时才写入机制解释。

## 实现边界

- 新增独立分析脚本与测试，不修改 CoSE-RAG 检索、验证器、生成器和现有运行产物。
- 不访问网络，不调用模型服务，不重新生成答案。
- 不使用 gold 信息改变已完成的检索结果；gold 只用于离线度量和可复现案例筛选。
- 不提交运行产物或论文文件到仓库，不自动执行 Git commit。

## 验证与交付

- 对统计函数、证据映射、终止逻辑一致性、案例中位数选择和 LaTeX 转义添加自动测试。
- 运行分析脚本两次并比较输出哈希，确保确定性。
- 从 CSV/JSONL 重新计算所有比例与增益，并与论文表逐项比对。
- 使用 XeLaTeX 编译完整论文，检查缺失引用、未定义标签和 overfull box；渲染包含两张新增表的页面并检查双栏布局。
- 交付修改后的完整 `.tex`、三个审计文件、分析脚本与测试，以及简短修改说明。
