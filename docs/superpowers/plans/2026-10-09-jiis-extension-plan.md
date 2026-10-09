# CoSE-RAG 面向 JIIS 的扩展与验证计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement approved engineering tasks task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本文件是研究与实施路线；研究结论须由真实实验决定，不自动 commit，不创建 worktree。

**Goal:** 将 BigData2026 投稿版 CoSE-RAG 扩展为具有明确新增研究贡献、可靠充分性验证、公平对照及可复现实证的 JIIS 稿件。

**Architecture:** 保留 EviBridge 的证据图、typed expansion 和预算选择骨架。先冻结会议版并核对实现一致性，再独立验证证据充分性和修复决策；只有诊断实验支持时才增加最小的方法模块。会议版与期刊版的配置、索引、输出及结论分开记录。

**Tech Stack:** Python、Pydantic、现有 EviBridge/DocumentTree、Qasper/HotpotQA evaluator、unittest、逐题 JSON/JSONL、paired/cluster bootstrap、静态科研图表、JIIS LaTeX。

**核对日期：** 2026-10-09，Asia/Shanghai。**现有依据：** 仓库当前代码、根目录 10 页 CoSE-RAG PDF（用户确认为 BigData2026 投稿版）及下文链接的一手资料。本轮没有复跑真实模型或执行期刊投稿；已补做两套主方法结果的一致性离线审计（第2.1节），未审核全部 baseline/消融或复算官方质量分数。

---

## 1. 研究判断与路线选择

推荐“一个聚焦的方法增量 + 严格的机制验证 + 外部效度扩展”。目前最大的风险不是表格不够多，而是：充分性代理的有效性尚未独立证明；现有组件之间的新增性受到近期工作挤压；方法收益与额外检索/重排预算未完全分开。

| 路线 | 收益 | 主要问题 | 建议 |
|---|---|---|---|
| 仅增加数据量、模型与 baseline | 快，能改善可信度和适用范围 | 很难解释相对于会议版及新文献的实质增量 | 用作必要补强，不能默认是完整期刊贡献 |
| 保留骨架，补强预算下的可信充分性与定向修复 | 能围绕当前最弱环节形成可检验贡献 | 需要标注、校准及等预算对照 | 首选；先做小规模诊断，再冻结方案 |
| 改成新 agent / 超图 / 全新训练框架 | 可能改变方法能力边界 | 工程/计算量大，贡献容易继续与他人重叠 | 暂不走；只有现有骨架无法支持研究问题时考虑 |

**不应把“加入校准”本身声称为全新贡献。** SURE-RAG 等已研究集合充分性和选择性回答。期刊版需要证明：文档结构中的具体缺口如何映射到可靠的 typed repair，在相同预算下何时继续检索、何时停止，以及错误连接/不充分证据如何影响决策。创新性最终取决于文献对比和结果，不取决于给模块换名称。

## 2. 投稿版现状与风险排序

下表中的数值是稿件报告值，尚非本轮复算结果。

| 优先级 | 风险 | 已见依据 | 首要动作 |
|---|---|---|---|
| P0 | 方法—代码—实验版本不一致 | p5 写 y=y_rule、LLM不改变接受条件；当前 verifier 在 soft missing 下允许 LLM sufficient=true | 专项审计未发现主结果接受覆盖；修复代码对约束的保障并固定运行来源，勿据潜在路径宣称旧结果受影响 |
| P0 | 充分性未被独立验证 | relevance/coverage/noise 为词面或类型代理；没有充分性误接受/误拒绝、校准结果 | 建立独立充分性标注与反事实诊断 |
| P0 | 与近期方法重叠 | structured gaps、graph expansion、stateful evidence、stopping 已有直接相关工作 | 写贡献对照表，加入最接近 baseline |
| P0 | 调参与评测分离不清 | p6 宣称配置预先固定；敏感性却在报告用 validation 上展示，未交代独立开发集 | 固定开发/最终评测 split，追溯搜索记录；不能据此直接指控泄漏 |
| P1 | verifier 消融混入计算量作用 | w/o verifier 强制一轮，full最多两轮 | 固定第二轮、随机扩展、定向扩展的等预算比较 |
| P1 | 支持证据后处理与生成证据混淆 | 当前 support_pruned 可在支持扩展后不再生成；本地主结果未发现上下文外support；ID合法不等于蕴含 | 分报 retrieval/context/support；审计证据是否实际可见 |
| P1 | 数据范围不足以支撑广义复杂文档能力 | Qasper结构化281篇1005题；Hotpot distractor固定1000题；无本稿真实PDF/视觉实验 | 加入一个真实长PDF任务及解析误差分析 |
| P1 | 证据噪声和联合质量存在取舍 | Table I：Hotpot SP Precision 61.12 vs HippoRAG2 83.88，Joint F1 49.88 vs 50.23 | 正面分析precision/recall trade-off，优化证据质量而非只扩召回 |
| P1 | 统计与成本证据不足 | 无CI/重复运行；成本未细分硬件、阶段和离线构建 | 配对/文档聚类统计，完整成本与失败率 |
| P2 | 复现和论文呈现滞后 | README仍BookRAG，无主项目完整环境锁；部分算法函数/权重不完整 | 发布最小复现说明、配置和公式—函数映射 |

具体提醒：规则 `_connectivity` 是“具有至少一条内部边的节点占比”。a-b 与 c-d 两个独立分量可得 1，不能用它证明全集连通。`_coverage` 检查 modality/scope 类型，不检查问题所需事实是否齐全。稿件 p8 Hotpot 人名案例诊断缺 table/caption，也应回查需求解析和日志，不能只展示修复成功。

投稿版已有七项组件消融、效率、四参数敏感性、两个轨迹案例。新增贡献不应再次登记为“首次补消融/效率/案例”。现有 Qasper Answer F1 相对 LongRAG 的增益为 1.63 个百分点，需统计与稳定性证据；Hotpot Joint F1 未领先，不能概括成所有指标最佳。

### 2.1 两项一致性专项审计（2026-10-09 已执行）

用户确认 BigData 投稿实验基于最新代码。审计保留该确认，并将“当前代码允许的行为”和“本地已保存结果中的行为”分开记录。审计代码基准为 `4c0d4d0fb5df49543d2c95345ba109f9e7741834`，生产代码、原稿与旧实验文件未改，未调用真实模型。

| 检查 | Qasper 主结果 | HotpotQA 主结果 |
|---|---:|---:|
| 题目 / 原文档或题内语料 | 1005 / 281 | 1000 / 1000 |
| 已复算检索轮次 | 1057 | 1004 |
| 规则拒绝、保存判定接受 | 0 | 0 |
| 五项数值指标不同于规则复算 | 0 | 0 |
| 最终支持块总次数 | 1811 | 2735 |
| 支持块位于重建生成上下文外 | 0 | 0 |
| result 与 retrieval 支持ID不一致 | 0 | 0 |
| 题ID缺失 / 重复、来源块缺失、选中块文本不符 | 0 | 0 |

**检查1：主实验未发现 LLM 改变接受判定，但当前代码并不保证稿件的约束。** 复算直接使用当前 tokenizer、schema 和 `RuleBasedSufficiencyVerifier`，逐轮使用保存的 demand、selected IDs 和源索引；规则只计 selected 内部边，使用同一索引全部边不会改变其计算。Qasper 38轮、Hotpot 3轮是可进入 LLM 分支的 soft rejection，均未转为接受。Qasper 8轮的 `missing_types` 从空列表变为 `["noise"]`，其余字段相同；不能概括成所有 verdict 字段完全一致，也不能从结果相同推断没有调用 LLM。历史日志缺 rule/LLM/raw/source/异常的独立记录，实际调用总量为 unknown。

当前 sanitizer 可以接受最简 `{"sufficient":true}`，不复验阈值；本地 mock 28个场景全部通过，其中7个人工反例出现 false→true。它们是代码反例，**不是投稿主实验的发生次数**。JIIS实现应显式区分规则接受模式与新判定模式，保留三个 verdict，按诊断集验证新模式。

**检查2：主实验的最终支持集都在重建的生成上下文中。** 每题以 `evidence_chain.json.evidence_chain` 重建初始 prompt 的证据块，交叉核对末轮 selected、顶层 selected、源索引文本及 result 中的最终 support。两套结果都无 fallback；Qasper全部明确 `answer_regeneration.used=false`。Hotpot全部缺 `answer_context_block_ids`、`answer_regeneration`、`support_controller`，因此不能直接声称新版再生成机制运行过。核心 verifier、tokenizer、demand/schema 及 legacy support 函数与7月31日相应源码相同，旧链路仍可核对。没有原始请求快照，此处是代码与日志重建的包含性结论，不是独立服务请求认证，也不是答案—证据语义蕴含结论。

当前代码仍存在 coverage_prune 扩展 support 却硬置 `regeneration_required=false`、再生成失败后保留计划 context/support、预算截断后无合法引用时沿用旧 support、fallback 保存上下文并集等路径。15个 fake-model 场景、21条断言已复现，实际传给 mock 的 prompt 可核对。上述异常路径在这两套主实验中没有被观察到；不能据此要求全部旧结果重跑。

**来源记录的具体限制。** Hotpot首/中/末样本生成时间为2026-07-31；三个新增保存字段由8月1日的 `ddb787c2e` 加入。这与“运行时采用当时最新版”相容，但不构成已重新序列化为当前版本的证明。Qasper主样本8月3日运行日志是 full/weak_only/coverage_prune/regen=false，其文档根 `rag_config_evibridge.yaml` 后被8月5日 static_topk消融覆写。`main.py`按策略名而非方法后缀保存配置；`Core/inference.py`可复用既有 result；不能用新配置文件日期倒推所有逐题结果重跑。

审计汇总与源码/稿件 hash保存在 [2026-10-09-consistency-audit.json](../../research/jiis/2026-10-09-consistency-audit.json)。忽略目录 `runs/jiis_audit_temp/root/` 保存逐题JSONL、7296项产物hash清单及只读脚本；两个独立mock子目录保存复现输入与调用。大索引、逐题文本、原始配置和临时工具不提交。

**下一步优先级调整：** 先补不可覆写的运行快照和实际成功生成阶段记录，明确代码应保持的两个约束；随后执行Task2的独立语义充分性pilot，量化“规则判充分但缺关键事实”的比例。此次0接受覆盖/0上下文外support的结果降低了这两项对既存主实验的担忧，却尚未证明充分性代理正确。暂不因这两条未触发路径复跑全量模型；只有语义/评分审计或最小修复确实改变运行输出时，才确定重跑范围。

### 2.2 可审计执行的工程准备（2026-10-09）

已按[单独实施计划](2026-10-09-cose-auditable-execution.md)增加显式模式，保留既有配置的默认行为和历史结果。`rule_only`固定规则接受、五项数值指标与noise_warning，LLM仅可修订诊断；`strict`按initial/fallback/regeneration中实际保留的成功客户端调用约束最终支持集。未参与生成的外部补充记录为posthoc；再生成失败保留draft实际上下文。新配置为 `config/evibridge_auditable.yaml`，suffix为 `evibridge_support_pruned_rule_only_strict_context`；它不是已经追溯确认的会议复现模式，也不是待验证的语义充分性期刊策略。

每轮rule/LLM/final轨迹、prompt/response hash、块ID/文本hash已接入。每次`run_rag`在实际输出目录建立独立`.runs/<UUID>/manifest.json`、逐题events及结束summary；新result链接该来源与输入指纹，复用缓存不回填今天的来源，旧来源缺失明确unknown。缓存输入明确错配时记录失败并停止，历史输入缺失记partial/unknown。实际配置/源码/输入/常见索引文件和配置指定VDB以指纹关联，私密配置脱敏，索引覆盖保守标partial/unknown。客户端调用记录不能认证服务端接收/内部截断，支持集包含关系不能证明答案蕴含；旧结果无请求记录仍为unknown。该改动没有补齐会议版全量模型版本、完整环境锁或所有baseline来源。

工程验证已通过EviBridge的121项回归和来源记录的28项新测试；Qasper相关47项通过，Hotpot相关20项中19项通过，固定采样parquet测试因本地缺PyArrow报错，未据此修改无关测试。合成小数据集成smoke确认result/retrieval/evidence_chain轨迹一致、外部支持分列、第二次缓存复用不调用模型且旧字节不变、错配停止、初始生成失败能落失败轨迹。独立spec与最终代码质量review通过；7301项历史文件hash一致。上述验证不含真实模型质量/成本测量，不产生新论文主表数字，完整边界见[验证记录](../../research/jiis/2026-10-09-auditable-execution-verification.md)。

独立诊断的下一项交付是[充分性标注规范](../../research/jiis/sufficiency_annotation.md)及开发/确认评测的分组清单。规范草案和空模板仅完成准备；120个开发源题、真实人工标注、pilot与确认评测尚未执行。优先用独立开发源题小规模预试规范，再冻结版本；不将已看validation包装为新holdout，也不因工程测试通过就增加期刊算法主张。

用户随后确认Qasper train、HotpotQA train均未用于调参/配置选择/候选结果查看。[本地开发数据核查](../../research/jiis/development_data_status.md)显示Qasper train有2593题、与历史主结果题ID/文档ID交集为0，支持下一步按文档准备开发协议；现存Hotpot ReAct简化train缺ID/context/supporting_facts，须补完整train。该清单未冻结split，未生产实际检索状态或人工标签。

## 3. 一手文献与借鉴方式

这是一份截至核对日的定向检索，不是穷尽综述。用户未指定关注的新论文清单，后续可补入。

| 工作/可核实状态 | 与本稿的关系 | 借鉴或比较方式 |
|---|---|---|
| [DocNavRAG](https://arxiv.org/abs/2608.01565)，2026-08-03 arXiv预印本 | 文档层级和跨区域关系、持续证据状态、收集至充分，研究定位非常接近 | 优先全文对照；比较结构导航与typed预算修复；未核实作者完整代码可用性，不能直接宣称已官方复现 |
| [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/)，ACL2026 | 明确充分性判定、结构化缺口和下一轮检索；会议稿已引用但未实验比较 | 优先控制器baseline，区分其trained judge与本方法训练成本；[作者代码](https://github.com/nianaaa/S2G-RAG)可核实 |
| [Relink](https://ojs.aaai.org/index.php/AAAI/article/view/40382)，AAAI2026 | query-specific evidence graph、补断路与筛干扰 | 路径修复baseline；借鉴断链/干扰关系诊断；[作者代码](https://github.com/DMiC-Lab-HFUT/Relink)有path scorer/数据库依赖，记录训练和适配成本 |
| [SURE-RAG](https://arxiv.org/abs/2605.03534)，2026-05预印本、7月v2 | 集合充分性、支持/反驳/不足及选择性回答；会议稿已引用 | 借鉴独立验证、冲突和risk-coverage评测；主要作verifier对照，不能当成与整条检索系统完全等价 |
| [Sufficient Context](https://arxiv.org/abs/2411.06037)，ICLR2025 | 区分检索证据不充分与生成器未利用充分证据 | 沿用问题—上下文充分性的诊断思路，评测judge独立于在线决策 |
| [GeAR](https://aclanthology.org/2025.findings-acl.624/)，Findings ACL2025 | graph expansion + 多步agent；会议稿已引用 | 对比图扩展和多轮证据记忆；预算受限时先于追加泛相关旧baseline |
| [HGRAG](https://ojs.aaai.org/index.php/AAAI/article/view/40623)，AAAI2026 | 跨粒度实体/段落与hypergraph diffusion | P2结构表示对照；不因此直接把现有图改为超图 |
| [When Should Multi-Round RAG Stop?](https://arxiv.org/abs/2608.13237)，2026-08预印本 | 首次STOP产生序列选择偏差，分类准确不等于部署停机质量 | 借鉴整条轨迹评测、独立阈值选择和总调用成本统计 |

JIIS 已有 [adaptive path generation 的多跳QA研究](https://link.springer.com/article/10.1007/s10844-025-00945-5)，也有 [faithful multi-source QA](https://link.springer.com/article/10.1007/s10844-026-01084-1)。它们支持题材适配，也要求本稿区别于一般的自适应路径/反馈/忠实性叙述。

**读文献之后的产出必须是对照矩阵。** 每篇记录：原始证据单位、结构来源、线上信息可见范围、充分性判定对象、修复动作、停止规则、训练数据、预算和source commit。不能以文章是否“用了graph”决定要不要借鉴。

## 4. 期刊版拟回答的研究问题

- RQ1：在词面相关且结构看似连通时，现有充分性判定有多少误接受？哪种缺口最常见？
- RQ2：在候选、上下文及模型调用预算可比时，缺口驱动的typed repair是否优于固定扩展和随机扩展？
- RQ3：可靠关系与真实文档结构分别贡献什么？收益是否仍存在于断边、误连和解析损坏条件？
- RQ4：停止/继续/证据不足的决策在不同模型与任务上是否校准，能否改善质量—风险—成本取舍？
- RQ5：从结构化文本迁移到真实长PDF时，哪些误差来自解析，哪些来自检索/选择/生成？

新增贡献候选控制为三项：可审计的缺口与关系可靠性表示；预算下可验证的修复/停止策略；独立充分性与机制/真实PDF实验。若第一阶段发现现有机制已经足够且新策略没有可靠收益，收缩算法主张，以明确的新实证发现扩展，不编造方法优越性。

## 5. 实验协议与资源分配

### 5.1 数据与 split

| 数据 | 用途 | 最小设计 | 增强设计 |
|---|---|---|---|
| Qasper | 长论文、跨章节、答案类型、paragraph evidence | 原validation1005作为历史复核；开发采用独立train文档；最终使用未用于选择的新官方可用评测split | 完整独立split，按文档长度/章节跨度/答案类型分层 |
| HotpotQA distractor | 多跳、bridge/comparison、sentence supporting facts | 保留seed42固定1000历史基准；开发从train选；最终扩大到预先冻结、未用于调参的validation题目 | 可承受时完整distractor validation；分报历史1000题、新增确认集及全集描述性结果，混合集不称未见holdout或full-wiki |
| MMLongBench-Doc真实PDF | 跨页、table/figure、无答案和解析误差 | 冻结数据修订版本及文档级开发/评测划分；按来源/跨页/可回答性抽固定至少300题评测子集，报告实际覆盖 | 完整可用评测集，使用同版官方协议 |
| HRI | 领域迁移 | P2小规模独立案例、人工证据说明 | 作为补充材料，不代替公开数据主实验 |

MMLongBench-Doc 的真实PDF、跨页、视觉来源和无答案问题来自[官方数据仓库](https://github.com/mayubo2333/MMLongBench-Doc)。数据曾修订，不能混用不同版本答案。其 page-level evidence 不等于 paragraph gold，不与Qasper Evidence F1跨单位直接比较；视觉结论要求生成器真正接收对应图像。若只能处理OCR文本，报告 text-only 设置。

已经用于选参数或看结果的题目不能重新命名为“未见holdout”。Qasper按文档拆分，充分性扰动的原题及所有变体进入同一split；Hotpot重复源文档需另作交集审计，并单列完全独立源文档子集与存在共享来源的子集。训练/验证/校准/确认评测的ID、顺序、hash与用途全部写入manifest。

### 5.2 baseline与比较矩阵

主生成模型先固定为会议版Qwen3-VL-8B-Instruct的明确checkpoint和temperature=0.1，用于隔离方法差异。第二个独立backbone按本机可部署资源选择并冻结版本；不要求追逐最新模型。

| 组 | 必要方法 | 执行范围 |
|---|---|---|
| 简单强对照 | BM25+reranker、Hybrid RRF、LongRAG/Full-document | 三任务，明确reranker和预算 |
| 历史图方法 | 官方HippoRAG2调用、KG2RAG论文适配、BookRAG/GBC | Qasper/Hotpot；PDF路线能公平适配者进入PDF表 |
| 直接相关新工作 | S2G-RAG；DocNavRAG；Relink或GeAR | 优先同预算开发pilot；至少覆盖充分性控制与图修复两类；不可运行项披露原因而不填伪结果 |
| 自身版本 | 固定会议版、期刊候选、等预算控制组 | 三任务；所有版本保持相同评测单位 |
| verifier对照 | 当前规则、LLM-only、Sufficient Context/SURE-style或作者实现、期刊候选 | 独立充分性标注与轨迹评测，勿混成整套RAG排名 |

DocNavRAG 若无可核实源码，按论文适配须明确标签、接口与偏离，不能挂“official”名称。S2G/Relink若用trained scorer/judge，审核训练与评测交集，单列训练成本；不使用gold构造的中间状态作为本方法线上输入。

全矩阵先在一个backbone完成。第二backbone只复核候选、会议版、最强简单对照及最接近新工作；重复模型运行先覆盖这些核心方法，避免所有旧弱baseline乘上所有模型形成无效开销。

### 5.3 指标、统计和成本

- 答案：Qasper official Answer F1；Hotpot Answer EM/F1、Joint EM/F1；PDF任务按固定版本官方答案协议。
- 证据：Qasper Evidence PRF/Recall@K；Hotpot SP PRF；PDF按page/source标注单位。retrieved/context/support三个集合分别统计。
- 充分性：confusion matrix、precision/recall/Macro-F1；误接受率 `P(gold insufficient | decision accept)` 与 false-positive rate `P(decision accept | gold insufficient)` 分别报告，注明分母。
- 校准：有可解释概率时报告Brier/ECE；硬二元规则不伪造概率。risk-coverage用答案错误率，另列unsupported answer风险，两者不混称。
- 轨迹：首次错误STOP率、浪费检索率、每轮缺口变化、增量证据/答案收益、修复成功率、budget-exhausted/no-progress终止率。
- 关系：最大连通分量比例、需求端点路径覆盖、路径可追溯率及人工关系有效率；辅助routing节点不充当原子支持证据。
- 成本：离线解析/建图/embedding/训练，与在线召回/rerank/verifier/生成/再生成分开；记录输入/输出token、真实调用数、均值/p50/p95延迟、硬件、并发、缓存、失败/重试和峰值内存。

最终上下文预算之外，增加reranker候选数、retrieval rounds和LLM调用数匹配的对照；另外画“真实总成本—质量”曲线。现有selector的词项token近似与模型tokenizer计数分别保存，不将4000代理token直接当作真实API输入token。

核心生成实验使用3个固定运行seed（42/43/44），保存实际服务返回与模型版本；先报告各seed结果，再对每题seed平均分做配对/文档cluster bootstrap，不能把同题三次输出当独立新增样本；若API不能保证seed行为，如实注明。bootstrap10000次用于CI，不能当作10000次模型重复。

Qasper配对bootstrap按文档cluster重采样，保留文档内全部问题及原先题目加权estimand；同时给按题目配对结果作为敏感性分析。Hotpot按题配对并检查共享语料相关性。主终点及对照预先登记，多项确认比较做Holm校正，开发pilot只作为探索性结果。现有bootstrap输出字段的统计定义也要审计，不把未核实的bootstrap tail概率自动称为严格显著性p值。

## 6. 可执行任务与验收条件

下文标注“拟新建”的文件仍是后续路径；已完成的审计与工程准备见第2.1/2.2节和单独实施计划。代码任务展开为单独spec和实现计划后执行；下面先规定接口、测试行为和研究验收，避免在尚未确认假设时写死整套算法。

### Task 0：冻结会议版，审计描述一致性（P0，第1周）

**Files:** 新建 `docs/research/jiis/conference_journal_delta.md`、`docs/research/jiis/conference_snapshot.md`；拟新建 `Scripts/analysis/jiis_artifact_audit.py`、`tests/test_jiis_artifact_audit.py`；读取 verifier/rag/config、源日志与官方eval文件。

- [ ] 记录稿件PDF SHA256、当时commit（或明确不可追溯）、config hash、dataset/index hash、模型checkpoint、执行命令和逐题覆盖。
- [ ] 把论文公式/符号映射到实际函数：Rel/Cov/Noise/H、typed weights、score normalization、selector权重、Rmax、support completion/regeneration。
- [x] 回查两套主方法结果的LLM接受覆盖：2061轮未发现覆盖；记录潜在代码反例、8轮诊断标签差异与来源限制，见第2.1节。baseline/消融不纳入本次完成范围。
- [x] 两套主方法结果的context/support离线包含性与来源一致性审计：4546个支持块均在重建context中，result/retrieval一致，无未知来源或题ID重复。结论限于日志重建，见第2.1节。
- [x] 为后续新运行增加客户端成功/失败调用、generation_provenance.prompt_block_ids和prompt hash、initial/fallback/regeneration来源及不可覆写manifest；strict区分实际与planned context，缓存来源不回填，见第2.2节。
- [ ] 完成真实新运行的引用合法性与来源单位重复独立审计；不能仅用旧answer_context_block_ids证明模型见到；历史缺失请求仍为unknown，新增代码不补造历史。
- [ ] 回查Hotpot人名案例的table/caption诊断；核对原query、demand、verdict、动作与新增证据，不按成功结果倒推诊断正确。
- [ ] 为审计模块覆盖三种行为：未知版本明确返回unknown；缺题/重复qid直接失败；support-context差异被记录且不静默修补。

**验收：** 每个会议主表行能链接到指定结果集/协议；不能溯源项明确标unknown。可观测代码—论文差异全部有处置：说明后续改动、修论文描述、或同协议复跑；不改历史数字掩盖差异。

### Task 1：锁定贡献与开发/确认评测协议（P0，第1–2周）

**Files:** 新建 `docs/research/jiis/related_work_matrix.md`、`docs/research/jiis/evaluation_protocol.md`；拟新建 `Scripts/preprocess/jiis_dataset_manifest.py`、`tests/test_jiis_dataset_manifest.py`；按需修改 `Core/configs/dataset_config.py` 和 replay入口。

- [ ] 完成第3节工作逐篇全文对照，冻结待验证的核心新主张；“与新论文相似”必须指出具体机制，不能按标题判断。
- [ ] 将train开发、calibration、confirmatory评测按文档/原问题组隔离；保存题目ID和来源文档交集审计。
- [ ] replay强制核查输入确属开发集合；tuning manifest与DatasetConfig manifest保持明确schema，不能混用。
- [ ] 预先登记模型、参数选择范围、主终点、等预算控制、缺题处理、多重比较和停止准则。
- [ ] 测试：篡改hash、跨split原题/扰动、将holdout送给tuning、错误manifest schema均被拒绝；同配置同输入得到同manifest。

**验收：** 已看过的validation不能称新holdout；评测集合在看候选结果前冻结；需要改变协议时记新版本及原因，而非覆盖旧文件。

### Task 2：独立充分性诊断集与小规模pilot（P0，第2–3周）

**Files:** 拟新建 `Scripts/analysis/jiis_sufficiency_states.py`、`Scripts/eval/jiis_sufficiency_eval.py`、`tests/test_jiis_sufficiency_eval.py`；标注规范存 `docs/research/jiis/sufficiency_annotation.md`；实际文本/标注存runs或获授权的数据发布位置。

- [x] 准备v0.1标注规范与空JSON模板，明确盲审、语义标签、源题分组、反事实核验和undetermined；不代表规范已冻结或有人类标签。
- [ ] 用独立开发来源的10–20个源题预试规范，保留分歧及修订记录，冻结v1后再正式标注。
- [ ] 从开发数据选120个源问题（Qasper/Hotpot各60），按类型、跨度、当前accept/reject和回答对错分层；全量统计权重与分层样本结果分开。
- [ ] 每个源问题生成原始状态、关键原子证据替换、缺中间事实、词面相近干扰、冲突/无答案测试，共最多600个状态；缺适用条件记录原因。
- [ ] 删除证据时用长度相近干扰替换，控制上下文块数/token；变体的语义不足必须人工确认，不能仅凭删除gold ID贴标签。
- [ ] 由两名标注者独立阅读question/evidence，判断sufficient/insufficient/undetermined、缺口及支持位置；争议仲裁并报告一致性。源文档无答案与当前上下文缺证据分别标注；确定标签子集计算判定指标，undetermined比例及其决策另报，不将它们静默删除后宣称全量质量；端到端QA保留全部题目。gold/全文仅用于离线核验，线上verifier不可见。
- [ ] **只删图边且证据文本相同时，语义充分性标签保持原判。** 图断路是结构鲁棒性干预，不能自动标成语义不充分。合成冲突与真实源文档冲突分列。
- [ ] 对当前规则、LLM判定及相关verifier做pilot；评测用judge独立于在线judge，人工子集审核其误差。pilot用于诊断与方案选择，不充当确认结果；另从已冻结确认集选120个源问题及同规则变体，使用冻结标注规范，参数冻结后仅作最终充分性评测。

**验收：** 能回答RQ1，报告各类误接受/误拒绝与失败原因，不以回答正确代替证据充分。若主要风险来自生成/支持后处理而非verifier，先修复那一环，不无条件增加新judge。

### Task 3：最小方法增量与可追溯决策（P0/P1，第3–4周，依赖Task2）

**Files:** 拟新建 `Core/rag/evibridge_sufficiency_signals.py`、`Core/rag/evibridge_repair_policy.py`；修改 demand/verifier/selector/rag/config；新增 `config/evibridge_jiis.yaml`；测试 `tests/test_evibridge_jiis_policy.py` 并扩现有模块/wiring测试。

候选接口：保留旧 `SufficiencyVerdict` 字段，追加version、decision_source、semantic_gap_signals、calibration_id、remaining_budget、stop_reason。gap至少包含自然语言需求、已见原子证据ID、缺失关系/事实、允许桥类型；不得含gold答案。

- [ ] 先把规则代理与语义支持信号分开输出；对claim/facet/hop级信号保留来源ID和undetermined，不把LLM自报confidence视为已校准概率。
- [x] 增加`legacy_hybrid`与`rule_only`显式接受模式，固定规则接受与独立诊断轨迹；旧接口和suffix兼容，新模式suffix隔离。只有Task0追溯历史快照后才命名conference-reproduction，不能先验假定规则模式就是投稿实验。
- [ ] 根据Task2证据确定journal-candidate的最小语义策略和冻结规则；现有rule_only不代替校准或语义判定。
- [ ] 用calibration集合选阈值/分数映射，目标操作点例如误接受风险5%；在确认评测上报告实际值与区间。该目标不是保证，不满足时诚实报告。
- [ ] 将缺口映射到context/semantic/hierarchy候选动作；按缺口收益、来源关系可靠性、增量真实成本排序，限制搜索次数、候选及上下文预算。
- [ ] 停止状态区分sufficient、budget_exhausted、no_progress、insufficient/abstain；标准QA主表仍报告全部问题，选择性回答单独报告coverage，不能丢弃拒答题后再称全量F1提升。
- [ ] 对routing桥与事实支持关系分开评估；不要求每个context/hierarchy邻接边都构成语义蕴含。
- [ ] 测试：两个独立边对不会被新连通诊断称全集连通；代理覆盖高但缺关键hop不能直接通过语义判定；非法来源ID被拒；所有动作受预算约束；hard失败/LLM异常可审计回退。
- [x] 测试strict支持集增补：提交支持证据在保留成功生成上下文内；外部证据触发可审计再生成或存posthoc，覆盖失败保留draft、预算裁剪、fallback和无效引用。legacy仅追加诊断，不改变旧支持行为；仍需独立验证语义支持。

**验收：** 新主张精确对应实现和测试。只保留pilot证明有用的模块；若新策略不优于简单阈值/固定扩展，记录负结果，缩减期刊主张。

### Task 4：等预算机制实验与关系干预（P1，第4–5周）

**Files:** 拟新建 `Scripts/analysis/jiis_mechanism_analysis.py`、`Scripts/analysis/jiis_graph_interventions.py`、对应tests；为关键控制新增独立配置并冻结hash。

- [ ] 对照：一轮；固定两轮无诊断；随机扩展；普通query扩展；缺口驱动typed repair。机制实验共享冻结首轮状态、同一触发题集合和生成设置，逐题匹配新增候选、rerank和调用预算，不仅匹配总体触发率。端到端策略允许各自触发，但单独报告总体效果与真实总成本。
- [ ] 分两种协议：固定轮数/调用上限用于比较诊断质量；按实际总成本匹配用于比较部署性价比。随机对照的额外诊断调用开销也记账。
- [ ] 分别移除typed graph、gap diagnosis、support pruner、LLM接受覆盖、校准、关系可靠性；已有会议消融作为历史背景，新增实验用于隔离交互和混杂。
- [ ] 在固定文本/候选下做随机删边、度数尽量保持重连、typed labels shuffle；报告扰动强度0/10/30/50%与实际图结构变化。
- [ ] 再做端到端检索图扰动，将候选变化与选择/判定变化分开。人工检查路径端点/关系是否真实相关，不能只测自动connectivity。
- [ ] 新运行保存首轮和各轮候选答案，仅用于离线分析且不反馈未授权gold；由此度量证据增加是否产生答案收益。旧日志缺答案不能补造历史首轮成绩。

**验收：** 可以区分诊断、关系、扩量和支持后处理的贡献；CI不支持的收益不写为确定改进；不能只挑触发第二轮的成功题汇总总体效果。

### Task 5：直接相关baseline与真实PDF迁移（P1，第4–7周，可并行）

**Files:** 按需新建 `Scripts/baselines/s2grag/`、`Scripts/baselines/docnavrag/`、`Scripts/baselines/relink/` 的adapter/README/source lock/tests；新建 `Scripts/preprocess/mmlongbench_jiis.py`、对应测试；沿用PDF tree builder与MMLong evaluator。

- [ ] baseline先做20题接口smoke，再固定不少于100题开发pilot；smoke不参与主表。核对上下文、提示词、训练来源、实际预算和官方/适配身份。
- [ ] 三数据集的推理结果先完整性验证，再评测；失效、超时、无引用计入失败率，不仅报告成功题。
- [ ] PDF先按文档拆分和源类型挑20题解析smoke，再冻结评测子集/全量；测试page ID、table/figure来源与unanswerable保留。
- [ ] 将相同文档的结构化/人工清洁文本与MinerU输出配对，评估标题层级丢失、OCR错误、表格/图注缺失对结果的影响。
- [ ] 视觉证据若进入主张，检查img_path、VLM实际输入、图表裁切和页码映射；否则明确text-only结果及边界。
- [ ] 分层报告跨章节/跨页、table/figure、bridge/comparison、no-answer；subgroup样本不足时显示数量与区间。

**验收：** 至少新增一个真实长文档/PDF的独立评测；最近直接相关方法有实证对照或具体不可复现说明；不得把仅paragraph文本实验写成多模态优势。

### Task 6：统计、总成本与可复现报告（P1，第6–8周）

**Files:** 扩展 `Eval/utils/paired_bootstrap.py` 或新建 `Eval/utils/cluster_bootstrap.py`；拟新建 `Scripts/analysis/jiis_cost_analysis.py`、`Scripts/analysis/jiis_report.py` 与tests；修改provider统计只追加字段；更新README和 `docs/research/jiis/reproduction.md`。

- [ ] 逐题配对校验、文档cluster bootstrap、3次模型运行汇总、预登记主终点及多重比较；无概率校准时不绘制虚构reliability diagram。
- [ ] 成本仪表记录所有stage/calls/input-output token/elapsed/cache/retry；无usage字段明确标missing或tokenizer估算。
- [ ] 输出质量—成本Pareto曲线与固定coverage操作点，单列index构建和trained baseline开销。
- [ ] 测试：qid不配对报错；cluster样本不会拆散同文档题；统计seed确定；所有调用token/时间能对账；失败或缺usage不被当0。
- [ ] 表格/图从逐题结果自动生成，每个数字存provenance；缓存replay与真实新生成结果分别标注。
- [ ] 记录主项目最小运行环境及可选依赖；source lock与环境锁分别描述，不上传真实数据/密钥/大索引。

**验收：** 每个主表数字和图点可回溯并重算；至少对最强对照给效应量与95%CI；成本与质量使用同一coverage/输入集合。

### Task 7：JIIS重写、扩展披露与投稿检查（P0/P1，第8–10周）

**Files:** 更新 `docs/research/jiis/conference_journal_delta.md`；用户指定稿件位置后创建期刊TeX/source/figures、cover letter草稿、数据可用性声明。默认不覆盖BigData原PDF/源稿，不自动提交。

- [ ] 引言从“提出typed graph+loop”改为明确研究缺口、已有方法边界及本次新发现；避免unsupported first/SOTA/全指标最佳主张。
- [ ] 方法补完整函数/阈值/边权与需求映射、预算、失败回退、停止逻辑和来源边界；公式—代码—配置逐项一致。
- [ ] 结果正面讨论SP Precision/Joint F1取舍、无答案与误接受、解析误差、失败率；案例同时给成功、中性和失败。
- [ ] conference-to-journal表区分原贡献、修正描述、真正新算法、新评测、新实证发现；不以增页数或换标题作为实质扩展证明。
- [ ] 按实际BigData状态准备披露：仍在审则准备研究但不默认提交重叠稿；接收后引用/披露会议版和重用权限；拒稿/正式撤稿则如实处理稿件状态。
- [ ] 检查JIIS最新规则、作者/资助/利益/数据/AI使用声明、源文件及编译PDF、参考文献和重用图表许可。
- [ ] 编译并渲染稿件，检查图表字号、截断、交叉引用和总页数。最终投稿动作留给用户。

**验收：** 新贡献有独立证据；无版本/统计/披露矛盾；不存在已看holdout调参、伪官方结果或unsupported综合最优表述。结果不支持假设时先收缩结论，不为满足“验收”筛除负结果。

## 7. 时间、依赖与停止点

时长是资源可用条件下的工作量估计，不是审稿/录用时间承诺。

| 周 | 核心工作 | 可并行工作 | 决策点 |
|---|---|---|---|
| 1 | Task0版本审计 | 文献全文/对照矩阵 | 能否追溯会议结果，哪些需重跑 |
| 2 | Task1协议冻结、Task2标注准备 | 新baseline环境核查 | 是否有独立确认评测数据 |
| 3 | Task2 pilot | PDF parser smoke | 主要失败在哪一环，是否需方法增量 |
| 4 | Task3最小候选 | baseline adapter | 冻结一个方案，避免反复追榜 |
| 5 | Task4机制对照 | Task5真实PDF与新baseline | 是否超越等预算简单控制 |
| 6–7 | 主矩阵与第二backbone | 成本/重复运行/统计 | 是否支撑核心主张及边界 |
| 8 | 重算与复现包 | JIIS正文/差异表 | 修复所有证据和描述不一致 |
| 9–10 | 稿件精修/渲染/披露 | 补有理由的验证 | 达到下面的投稿准备门槛 |

算力紧张时先保证：版本审计、独立充分性诊断、等预算对照、最接近新baseline、一个真实PDF任务、CI和总成本。可后置HGRAG、HRI扩量、第三backbone和大规模开放语料。若省略PDF，应收缩标题/结论到已验证的结构化文本任务，并明确剩余外部效度风险。

**最先执行的五项：** 冻结会议快照；核对LLM接受覆盖；审计context/support和异常诊断；锁开发/最终评测split；启动独立充分性pilot。不能用“先加更多模型”替代这些任务。

## 8. 复用命令与完成门槛

以下现有命令已从源码核对。真实路径通过PowerShell变量提供；文件必须来自已冻结同协议实验，模板中的TODO不直接运行。

```powershell
$datasetConfig = 'runs/jiis/local/qasper_confirmatory.yaml'
$methodSuffix = 'evibridge_support_pruned'
python main.py -c config/evibridge_support_pruned.yaml -d $datasetConfig --nsplit 1 --num 1 index --stage evibridge
python main.py -c config/evibridge_support_pruned.yaml -d $datasetConfig --nsplit 1 --num 1 rag
python -m Scripts.eval.qasper_run_validator --dataset-config $datasetConfig --method $methodSuffix
python -m Scripts.eval.qasper_official --dataset-config $datasetConfig --method $methodSuffix
```

这是现有版本复核命令；未来 `evibridge_jiis.yaml` 与suffix仅在Task3实现后使用。

```powershell
python -m unittest discover -s tests -p 'test_evibridge*.py'
python -m unittest discover -s tests -p 'test_qasper*.py'
python -m unittest discover -s tests -p 'test_hotpotqa*.py'
python -m unittest tests.test_dataset_manifest tests.test_paired_bootstrap tests.test_closed_loop_analysis tests.test_closed_loop_paper
```

```powershell
$baselineDetail = 'runs/jiis/analysis/baseline_detail.json'
$candidateDetail = 'runs/jiis/analysis/candidate_detail.json'
python -m Eval.utils.paired_bootstrap --baseline $baselineDetail --candidate $candidateDetail --metric answer_f1 --resamples 10000 --seed 42 --output runs/jiis/analysis/paired_ci.json
```

该CLI目前按题目重采样；Qasper文档cluster版需要Task6另行实现。现有closed_loop CLI硬编码1005/1000及会议后缀，期刊新split不得未经修改直接套用。

投稿准备门槛：

- [ ] 会议状态明确，版本差异与重用披露完整；主表可追溯。
- [ ] 与S2G/DocNav/Relink等直接相关工作的差异能够具体解释。
- [ ] 至少一项实质新增研究贡献有证据，而非仅参数/页数/表格增加。
- [ ] 充分性、连接和答案支持的概念/指标不混用；关键误判可独立测量。
- [ ] 等预算对照能支撑所声称的机制，或明确给出其失败边界。
- [ ] 调参与确认评测分离；指标单位、coverage、CI、失败率及总成本可复算。
- [ ] 真实PDF/视觉/跨文档主张只覆盖确实验证的设置；不利指标完整保留。
- [ ] 文档、代码、配置、稿件描述一致；可选依赖/模型版本/官方适配说明清楚。

## 9. JIIS要求与会议衔接

题材与智能检索、知识表示/融合、复杂查询及系统实证方向相符，这是范围判断，不能据此量化接收概率。[JIIS aims and scope](https://link.springer.com/journal/10844/aims-and-scope)

当前官方指南要求LaTeX、最多25页（含参考文献/图表）、150–250词摘要、4–6关键词，并提供源码/编译PDF及数据可用性等声明。允许既有工作扩展但须透明披露重用；投稿声明同时要求工作未在其他地方审理。[JIIS submission guidelines](https://link.springer.com/journal/10844/submission-guidelines)

BigData2026主会官方通知日期为2026-10-24；实际稿件状态以作者系统/通知为准。现在可以开展期刊研究，但仍在审的核心重叠稿不能默认并行投稿。[BigData2026 CFP](https://bigdataieee.org/BigData2026/calls/papers/)、[IEEE conference author responsibilities](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/your-responsibilities-and-rights/)

如会议稿仍在审且拟提交核心重叠期刊稿，建议先向编辑及会议组织方准确说明状态和新增贡献，取得明确意见；这是审慎建议，不是所有已接收待发表扩展稿均需额外许可的统一规定。已接收待发表版本按实际状态披露和引用。本计划没有代发任何信息。本次查阅的JIIS通用指南与BigData2026主会CFP中未查到固定新增比例或通向JIIS的自动邀请安排，不能将经验比例当官方门槛。[Springer journal policies](https://link.springer.com/brands/springer/journal-policies)

最终建议：借鉴新工作的诊断、对照与评价设计，优先比较最接近方法；保持本方法主线集中。丰富实验是必要条件，本项目更需要先证明充分性与定向修复的独立价值。无法根据当前材料负责任地给出个人稿件的数值接收概率。
