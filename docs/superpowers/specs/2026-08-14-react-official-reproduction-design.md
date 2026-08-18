# ReAct 官方代码双数据集复现实验设计

## 1. 目标

基于 ReAct ICLR 2023 官方代码，在不改变当前统一数据清单、生成模型和评测口径的前提下，重新完成 Qasper validation full 与 HotpotQA fixed-1000 实验。复现实验用于替换当前存在异常格式修复率的本地 ReAct 结果。

本实验采用“官方控制逻辑、统一模型、固定本地语料”的闭集协议：ReAct 负责 Thought--Action--Observation 交互式检索与回答，Qwen/Qwen3-VL-8B-Instruct 作为统一语言模型，所有 Search 和 Lookup 只能访问当前问题被允许使用的本地文档内容。

## 2. 官方来源与隔离原则

- 官方仓库：`https://github.com/ysymyth/ReAct`
- 固定 commit：`6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`
- 许可证：MIT
- 官方源码保存于 Git 忽略目录 `runs/third_party/ReAct`；Git 连接失败时使用对应 commit ZIP，并记录归档哈希。
- 仓库内新增 source lock，记录仓库、commit、许可证和归档哈希，不提交官方源码。
- 新适配器位于 `Scripts/baselines/react_official/`，不修改现有 `Core/rag/react_*` 实现及旧结果。
- 新方法后缀固定为 `react_official_repro`，论文表格仍显示为 `ReAct`。

## 3. 官方控制逻辑

适配器保持官方 HotpotQA notebook 的交互协议：

- 最多执行 7 个 Thought--Action--Observation 步骤；
- Action 仅允许 `Search[...]`、`Lookup[...]` 和 `Finish[...]`；
- 每步模型调用使用 `temperature=0`、`max_tokens=100`；
- 主调用以当前步骤的 `\nObservation i:` 为停止序列；
- 若主调用不能解析出 `\nAction i:`，保留首行 Thought，并按官方代码额外调用一次模型补全 Action；
- 补全调用以换行为停止序列；
- 第 7 步后仍未结束时执行 `Finish[]`，并记录 `step_limit`；
- 服务端若忽略停止序列，客户端仅在同一停止标记处截断响应，不改写已生成文本。

HotpotQA 使用官方 `webthink_simple6` 六示例提示。Qasper 没有官方 ReAct 提示，因此使用现有六个 Qasper 示例；六个问题均来自 Qasper train，不来自 validation full。

## 4. 固定本地环境适配

### 4.1 HotpotQA

- 每个 fixed-1000 问题构成一个独立环境；
- 题目提供的每个上下文标题对应一个 page；
- page 内句子按原始 sentence ID 排列；
- `Search[entity]` 优先执行标准化后的标题精确匹配；
- 精确匹配失败时，仅在当前题目的标题集合中返回按 BM25 排序的相似标题，不检索其他问题或实时 Wikipedia；
- `Lookup[keyword]` 从当前 page 中依次返回包含关键词的下一句；
- 观察到的句子保留合法 title--sentence pair。

### 4.2 Qasper

- 每篇论文构成一个独立环境，同一论文问题共享只读语料；
- section path 对应 page，段落按论文原始顺序排列；
- `Search[query]` 优先匹配 section path，失败时仅在当前论文中执行 BM25 页面检索；
- `Lookup[keyword]` 从当前 page 中依次返回包含关键词的下一段或句子；
- 所有观察保留原始 paragraph ID 和未添加标题的原文。

## 5. 证据排序与统一输出

官方 ReAct 只返回答案，不直接输出标准证据。为接入统一评测，适配器仅使用轨迹实际观察到的内容构造证据序列：

1. `Lookup` 命中的原子证据按观察顺序排列；
2. `Search` 返回的原子证据随后按首次观察顺序排列；
3. 重复原子证据仅保留首次出现；
4. 不使用标准答案或标准证据重排、补全或回退。

Qasper 由现有 dynamic top-k 协议提交 paragraph-level evidence。HotpotQA 提交最多 4 个合法 title--sentence pair。若轨迹没有观察到合法原子证据，则提交空证据，而不读取 gold evidence 进行回退。

每题输出至少包含：

- 最终答案；
- 完整 ReAct trajectory；
- 每步原始模型响应、解析结果、Action 和 Observation；
- 观察到的证据及来源标识；
- 调用次数、修复调用数、Search/Lookup 次数和终止原因；
- 分阶段 token 与时间成本。

## 6. 数据集与统一参数

### Qasper

- validation full；
- 281 篇论文、1005 个问题；
- 使用当前锁定 manifest、问题顺序和数据哈希；
- 报告 Answer F1、Evidence Precision、Evidence Recall 和 Evidence F1。

### HotpotQA

- distractor validation fixed-1000；
- 799 个 bridge、201 个 comparison；
- 使用当前锁定 manifest、问题顺序和数据哈希；
- 报告 Answer EM、Answer F1、SP Precision、SP Recall、SP F1 和 Joint F1。

### 统一设置

- 模型：Qwen/Qwen3-VL-8B-Instruct；
- ReAct 温度：0；
- 单步最大输出：100 tokens；
- 最大交互步数：7；
- 最终答案直接来自 `Finish[...]`，不增加独立答案生成调用；
- 不进行任务特定训练或调参。

## 7. 运行阶段与恢复

适配器分为三个阶段：

1. `prepare`：读取现有数据 manifest，导出 page、证据映射和问题；
2. `run_official`：执行官方 ReAct 循环并保存逐题轨迹；
3. `finalize`：验证证据标识，转换为现有评测目录结构并汇总成本。

恢复规则：

- 数据哈希、配置哈希、源码 commit 和模型名一致时复用已完成题目；
- 任一身份字段变化时创建新 run，不覆盖旧输出；
- 已完成题目只有在 trajectory、答案、证据映射和成本文件均通过校验后才跳过；
- 网络超时按现有安全重试策略处理，TLS 证书校验保持开启；
- JSON 或 Action 解析失败只执行官方的一次 Action 修复，不增加额外语义重试。

## 8. 测试与验证

自动测试覆盖：

- 官方主调用与修复调用的 stop、temperature 和 max_tokens 参数；
- 官方 7 步循环及第 7 步后的 `Finish[]` 行为；
- 服务端忽略 stop 时的客户端截断；
- Search 精确匹配、相似标题返回和闭集边界；
- Lookup 的顺序推进；
- Qasper paragraph ID 与 HotpotQA title--sentence 映射；
- 证据排序、去重及空证据行为；
- checkpoint 恢复和运行身份校验；
- 旧 ReAct、EviBridge、KG²RAG 和 HippoRAG 2 实现不受影响。

正式运行前依次完成：

1. 官方示例级 smoke；
2. Qasper 2 篇论文 smoke；
3. HotpotQA 2 个 bridge 与 2 个 comparison smoke；
4. 双数据集完整运行；
5. 统一评测与诊断汇总。

## 9. 完成标准

- Qasper 1005/1005、HotpotQA 1000/1000 预测覆盖；
- 所有提交证据均可回溯到当前题目允许访问的原文；
- 主表指标、token、时间和诊断统计来自同一次完整运行；
- 报告 mean steps、mean calls、repair-call rate、invalid-action rate 和 step-limit rate；
- 不以修复调用率下降作为筛选配置或重跑选优依据；
- 新结果不覆盖旧 ReAct 结果，替换论文表格前保留两套结果及运行身份。

## 10. 非目标

- 不使用已停用的 `text-davinci-002`；
- 不访问实时 Wikipedia；
- 不在 Qasper validation full 上选择或改写示例；
- 不训练、微调或自动优化 ReAct prompt；
- 不修改论文正文，除非用户在实验完成后另行要求更新表格。
