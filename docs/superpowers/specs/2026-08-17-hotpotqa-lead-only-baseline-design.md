# HotpotQA Lead-only 基线设计

## 目标

修正 HotpotQA 中原 `abstract_only` 因不存在 Abstract 节点而退化为无上下文问答的问题。新增独立方法 `lead_only`，在固定 1000 题样本上重新生成答案、支持事实和开销统计。Qasper 的 `abstract_only` 及既有结果保持不变。

## 实验协议

- 每道 HotpotQA 问题使用该题 distractor context 中全部候选 Wikipedia 条目。
- 每个标题仅保留原始 `sentence_id = 0` 的首句，不按问题相关性筛选或排序，不读取标准答案或标准 supporting facts。
- 每条首句保留原始节点 ID、`hotpot_title` 和 `hotpot_sent_id`，从而能够映射为合法的 title--sentence pair。
- 最终生成器沿用现有统一模型、温度和答案协议，并动态选择最多 4 个 supporting facts。
- 无效引用沿用现有 citation validation 规则丢弃；若没有有效引用，则只能从当前 Lead-only 合法首句中受控回退。
- 方法内部名为 `lead_only`。论文呈现时，Qasper 的 `Abstract-only` 与 HotpotQA 的 `Lead-only` 应明确为数据集对应的弱信息下界。

## 代码边界

1. 在 vanilla RAG 配置 discriminator 中加入 `lead_only`。
2. 在 `VanillaRAG` 中新增 Lead-only 上下文构造，按标题选取句子 0，并复用现有生成、引用校验和结果保存逻辑。
3. 新增 HotpotQA Lead-only 私有运行配置；不覆盖原 `abstract_only` 配置或输出目录。
4. 不修改 Qasper、其他 baseline、主论文或既有结果文件。

## 测试

按测试先行方式覆盖：

- 一个标题有多句时只返回 `sentence_id = 0`；
- 多个标题各返回一条首句，顺序稳定；
- 输出保留节点 ID、标题和句号 ID；
- 不存在句子 0 的标题不伪造事实；若所有标题均无合法首句，则明确返回空上下文；
- citation fallback 只使用 Lead-only 候选，且不访问 gold supporting facts；
- `abstract_only` 现有行为及资源加载测试无回归。

## 完整运行与产物

- 数据范围：当前 HotpotQA distractor validation fixed-1000，保持完全相同的问题 ID、顺序和数据哈希。
- 运行覆盖：1000/1000，缺失预测为 0。
- 主指标：Answer EM、Answer F1、SP Precision、SP Recall、SP F1、Joint F1。
- 开销指标：Tokens/Question、Seconds/Question；墙钟时间仅作为实测在线开销，不解释为理论复杂度。
- 保留逐题 `result.json`、`retrieval_res.json`、汇总 `final_results.json`、`token_cost.json`、评测结果及覆盖审计。
- 新运行使用独立方法后缀和目录，禁止覆盖旧的 `abstract_only` 结果，以便追溯差异。

## 完成标准

- 所有新增与相关回归测试通过；
- 1000 道题均有预测；
- 每个输出 supporting fact 都可回溯到当前题目的原始 title--sentence pair；
- SP 指标不再因所有检索结果为空而机械归零；
- 主指标与开销来自同一次完整运行，并保存一份机器可读汇总。
