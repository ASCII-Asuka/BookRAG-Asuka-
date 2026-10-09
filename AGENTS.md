# AGENTS.md

## 适用范围与工作约束

本文件适用于 BookRAG-Asuka 整个仓库；更近的 `AGENTS.md` 优先。最后一次按代码和投稿稿件核对：2026-10-09。

- 直接在 `dev` 分支修改，不新建 worktree，不自动 commit、push 或提交论文，由用户自行操作。
- 修改前先执行 `git status --short` 和 `git branch --show-current`，保留用户已有改动、本地数据与实验产物；不因状态变化自动切分支或回滚。
- `runs/` 为被忽略的本地实验目录。索引、日志、缓存、模型调用结果、fake SDK、临时稿件提取文本及大数据不提交。
- 不把 API key、本机绝对路径、模型服务内网地址写入通用配置或文档。环境变量和本地配置仍需保护，不输出凭据。
- 本次指南更新及 JIIS 计划不代表已经实现期刊扩展、复跑模型或验证全部历史结果。

## 项目定位与论文版本

仓库源自 **BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents**，保留 BookRAG/GBC、HRI 和多种 baseline。

当前研究主线的正式投稿名称为 **CoSE-RAG: Connectivity- and Sufficiency-Aware Evidence Retrieval for Complex Document Question Answering**；用户已确认根目录 CoSE-RAG PDF 是 BigData2026 投稿版。论文状态目前只确认“已投稿”，不能写成“已接收”。

`EviBridge` / `evibridge` 是现有代码、配置、索引文件和结果目录使用的内部名称，早期文档曾称 EviBridge-RAG。论文用 CoSE-RAG，代码继续保留内部名称以兼容既有索引、实验和评测；不要仅为统一命名批量重命名接口。

方法主线：evidence demand parsing -> 多粒度 seed recall -> typed PPR / shortest-path bridge expansion -> budgeted evidence selection -> sufficiency diagnosis / retrieval repair -> answer generation -> supporting evidence validation / completion / pruning。

目标是构造相关、覆盖充分、连接可解释、预算可控且来源可追溯的证据集。结构连通、证据来源合法与语义充分是不同概念，不得相互替代。

投稿版实验范围是 Qasper validation 的 281 篇论文 / 1005 个问题，以及 HotpotQA distractor validation 的 seed=42 固定 1000 题。稿件已有组件消融、效率、参数敏感性和闭环轨迹；这些不是未来 JIIS 版自动成立的“新增贡献”。代码支持 MMLongBench、M3DocVQA、HRI，不等于本次投稿已评测这些任务或真实 PDF / 视觉路线。

JIIS 扩展计划见 [2026-10-09-jiis-extension-plan.md](docs/superpowers/plans/2026-10-09-jiis-extension-plan.md)。计划中的新模块、校准机制与实验均为后续任务，实施前按最新证据修订。

## 核心架构

- `main.py`：`index` / `rag` 入口；按 `doc_uuid`、`doc_path` 分组，支持分片。**`--nsplit`、`--num` 默认均为 2，单进程完整实验必须显式传 `--nsplit 1 --num 1`。**
- `Core/construct_index.py`、`Core/inference.py`：离线构建和在线编排。
- `Core/Index/Tree.py`：`DocumentTree` / `TreeNode`，保留父子层级、页码、PDF block ID、文本、图片路径、表格等来源信息。
- `Core/Index/Graph.py`、`GBCIndex.py`：实体关系图、`tree2kg` / `source_ids` 与树图向量库组合。
- `Core/Index/HRIIndex.py`：水利规范证据锚点；HRI 为领域迁移案例，非当前论文主方法。
- `Core/Index/EvidenceBridgeIndex.py`：证据块与 typed bridges、JSON 持久化、BM25、vector 文档和 typed graph 导出。
- `Core/pipelines/`：PDF / MinerU 树构建、HRI 构建、EviBridge 构建及 baseline 所需流水线。
- `Core/rag/`：各策略及需求解析、扩展、选择、验证、支持证据处理。
- `Core/configs/`：Pydantic 系统模型、部分 dataclass 子配置、RAG discriminator；`Core/provider/` 封装 LLM/VLM/embedding/reranker/向量库/MinerU/token 统计。
- `Core/utils/resource_loader.py`：策略依赖加载；`Core/configs/dataset_config.py`：数据集配置及可选 manifest 校验。
- `Eval/evaluation.py`：按数据集名懒加载评测器；`Scripts/eval/`：Qasper 官方协议导出、覆盖校验、支持集 replay、诊断与汇总。
- `Scripts/preprocess/`：数据下载与转换；`Scripts/baselines/`：独立上游复现/适配；`Scripts/analysis/`：闭环和轨迹分析。
- `config/`：系统/方法/消融示例；`Scripts/cfg/`：数据集模板与本地配置；`tests/`：单元、wiring、评测、预处理和分析测试。

## 离线索引

`index --stage` 支持 `tree`、`graph`、`vdb`、`hri`、`evibridge`、`mm_reranker`、`rebuild_graph_vdb`、`all`。

**当前 `all` 执行 tree / graph / vdb，不包括 HRI 或 EviBridge；构建主方法必须显式使用 `--stage evibridge`。**

- PDF 路线：PDF -> MinerU -> `DocumentTree` -> 对应索引。`tree.pkl` 已存在时，树构建优先加载，不应强制导入/调用 MinerU。
- Qasper 结构化路线：raw JSON 的全文 -> pseudo `DocumentTree` -> EviBridge；无需 PDF parser，但不能宣称已检验真实 PDF 结构恢复或多模态能力。
- HotpotQA 路线：题目提供的 distractor context -> 题内封闭语料 -> pseudo tree / evidence graph。不能称 full-wiki 开放检索；所有方法须使用相同候选语料。

BookRAG/GBC 常见产物为 `tree.pkl`、`tree.json`、`graph_data*.json`、`kg_vdb` / `kg_vdb_basic`、`Tree_vdb`。EviBridge 主要为 `evibridge_index.json`、`evibridge_bm25.pkl`，可选向量库。

`EvidenceBlock` 类型包括 paragraph/table/figure/caption/title/summary/entity/patch/equation/unknown。bridge 顶层类型固定 context/semantic/hierarchy，细分关系放在 `relation_type`，如 parent_child、same_section、caption_of、mentions_entity、entity_cooccur、summary_of、patch_contains、shared_terms。保留块到源树节点、章节、页码及原始证据单位的映射，加载旧索引时考虑兼容性。

## 在线策略与 baseline 身份

工厂策略 discriminator：`gbc`、`graph`、`gbcvanilla`、`vanilla`、`mmr`、`traverse`、`hri`、`evibridge`、`lightrag`、`hipporag`。

- `gbc`：BookRAG 的规划、实体映射、章节/子树检索、rerank/skyline、map/reduce 生成。
- `graph`：仓库的 GBC 图/PPR baseline；不能仅凭名称称为 Microsoft GraphRAG 官方实现。
- `gbcvanilla`：树与图向量库 top-k；`mmr`：多模态检索生成；`traverse`：逐层树导航。
- `vanilla` 的 `retrieval_method` 支持 vanilla/bm25/raptor/pdf_vanilla/hybrid/bm25_rerank/abstract_only/lead_only/full_document/longrag/ircot/react。IRCoT 和仓库 ReAct 不是独立策略 discriminator。Qasper 使用 paragraph corpus 时同步设置 `corpus_unit` / `bm25_corpus`。
- `lightrag`：LightRAG 集成及本地诊断；`hipporag`：仓库轻量实现，不等同于官方 HippoRAG 2。

独立 baseline 在 `Scripts/baselines/`，**不通过新增 `main.py` discriminator 运行**：

- `hipporag2/`：固定上游源码版本，调用官方 retrieval，经 prepare / run_official / finalize 适配统一语料和输出。
- `kg2rag/`：按论文 generic triples graph 进行封闭语料适配；上游 source lock 用于来源记录，不能据此声称所有代码逐行调用官方实现。
- `react_official/`：官方控制流与 prompt 复现，适配统一模型和 closed corpus；与 `vanilla.retrieval_method=react` 路线区分。

运行前阅读各自 README、`official-source-lock.json` 和环境脚本；论文逐项披露 model/corpus/prompt/budget/evidence export 的改动。baseline 失败、空图、无效引用或适配偏差不得静默混入“官方结果”。

新增或修改策略同步更新配置联合类型、工厂、resource loader、示例 YAML 和 wiring 测试；按需懒加载可选依赖，避免轻量评测被无关 SDK 或 PDF 依赖阻塞。

## CoSE-RAG / EviBridge 当前实现

- `evibridge_demand.py`：rule/llm/hybrid parser，Qasper conservative mode、HotpotQA profile 和 implicit multi-hop 处理。
- `evibridge_ppr.py`：typed PPR、分类型贡献、shortest-path connector 和路径导出。
- `evibridge_selector.py`：预算化选择，兼顾相关性、覆盖、连接、多样性、冗余和 token cost。
- `evibridge_verifier.py`：规则与可选 LLM verdict、缺失类型/桥/下一动作和 sanitization。
- `evibridge_support_pruner.py`：归一化相关性、facet coverage、冗余筛除和 supporting evidence 裁剪。
- `evibridge_rag.py`：检索闭环、repair、答案生成、引用校验、支持集补全/重排、可选再生成和全部输出。
- `evibridge_config.py`：完整策略配置与 `method_suffix`；不要凭配置文件名猜输出目录。

当前机制必须准确描述：

1. 规则 `_coverage` 是 modality/scope 类型覆盖代理；`_noise` 使用 query 词项重叠代理；不能直接称为语义需求满足率或人工判断的噪声率。
2. 规则 `_connectivity` 计算有内部邻接边的节点比例；多个不连通分量也可能得到 1，不是最大连通分量比例，也不保证逻辑推理链成立。shortest path 和 selector 的连接奖励不提供全局语义充分性保证。
3. 规则判 sufficient 时直接返回；hard missing 也直接返回；其余情况可调用 LLM，sanitizer 在允许条件下可以接受其 sufficient=true。**投稿版 p5 写 LLM 不改变 acceptance，与当前代码存在需追溯的差异。**先查实验 commit/config/logs，不能断言历史实验执行了当前逻辑，也不能直接覆盖旧结果。
4. 支持证据可以在生成后补全、answer-conditioned rerank 或裁剪。ID 属于合法来源只证明可追溯，不证明答案由该证据蕴含；区分生成上下文与后处理支持集。
5. `config/evibridge_support_pruned.yaml` 使用 coverage_prune，且 `regenerate_on_support_expansion=false`；基础 evibridge 配置当前开启支持扩展再生成。核查具体运行的配置，不把不同 pipeline 混为同一版本。

当前消融除三类边、multi-seed、typed weights、selector、verifier、static_topk 外，还包括 citation_reorder、support_reranker、verifier_repair、implicit_multihop；精确名称以 `ablation_variant` 为准。`wo_sufficiency_verifier` 将检索轮数限制为 1，因此单独此对照不能隔离 verifier 诊断与第二轮候选/计算量的作用。

## 输出与方法后缀

常见路径：`eval_<dataset>_<method>/query_XXX/{result,retrieval_res,evidence_chain}.json`、`final_results.json`、`token_cost.json`。

- 保留旧 `retrieved_node_ids`，新增字段不得破坏旧评测。
- `retrieved_block_ids` / selected 记录检索/选择，`supporting_block_ids` 记录提交的支持集；`answer_context_block_ids` 是 controller 记录的答案上下文，可能先于再生成更新，须结合 `answer_regeneration.used`、调用记录或实际 prompt 判断最终模型是否见到。不能仅凭此字段证明 support 属于真实生成上下文；保留 paragraph / title-sentence 来源映射。
- `retrieval_res.json` 保存 demand/provenance、seed、typed PPR/score parts、connector paths/edges、selected/bridges、verification、iterations、stopping_reason。
- 支持处理相关字段包括 supporting_evidence/budget、citation_validation、support_controller、answer_extraction、fallback、answer_regeneration；`evidence_chain.json` 保留关联信息。
- verdict 的 missing_types、missing_bridge_types、next_action、next_bridge 与 Pydantic schema / structured prompt 保持一致。

`method_suffix` 规则：从 evibridge 开始，依次追加非 full variant、`_fallback`，再追加 full 对应的 support suffix。无 fallback 时 full + coverage_prune 为 evibridge_support_pruned，full + 非 always completion 为 evibridge_support_controller；组合例为 evibridge_fallback_support_pruned。基础 `config/evibridge.yaml` 当前并非保证输出裸 `evibridge`。推理、评测、分析都使用实际 suffix。

## 数据、预处理与评测协议

统一 JSON 为列表，每题至少含 question、answer、doc_uuid、doc_path。保留 qasper_question_id / hotpotqa_question_id、来源证据和数据 split 信息；无 doc_uuid/doc_path 会破坏分组。

- Qasper：`Scripts/preprocess/download_qasper.py`、`qasper_evibridge.py`；pseudo tree 按全文组织，不把 gold answers/evidence 送入索引、检索、demand 或 verifier。
- HotpotQA：`download_hotpotqa.py`、`hotpotqa_evibridge.py`、`hotpotqa_fixed_sample.py`；固定采样记录 question IDs、顺序、seed、SHA256 与语料范围。`lead_only` 为摘要捷径对照。
- `DatasetConfig.manifest_path` 可校验统一 JSON SHA256、题目顺序与数量；这是输入完整性校验，不是调参与最终评测分离的证明。
- `Scripts/cfg/example-*.yaml` 为模板，可能仍含 TODO。真实数据、机器路径、模型服务放本地配置，不能将模板当可直接运行的实验。
- `Scripts/data/` 本地 HRI/smoke 和 `data/documents/hydro/` 不误删或覆盖。

评测必须区分协议：

- `Eval/evaluation.py` 支持 Qasper、HotpotQA、MMLongBench、M3DocRAG、HRI_*。传统 Qasper evaluator 含 LLM answer extraction，不与官方确定性评测混称；当前 `--skip_llm_judge` 不保证 Qasper 跳过该调用。
- `Scripts/eval/qasper_run_validator.py` 校验题目覆盖、结果完整性/ID及 LightRAG 图/fallback 诊断；不要称它已通用校验所有证据ID/文本来源；`qasper_official.py` 导出 predictions.jsonl、official_eval.json、official_eval_detail.json、coverage_manifest.json，可使用外部官方 evaluator。主结论优先按明确的官方协议报告。
- Qasper 多标注证据 PRF 按同一个最大 F1 标注组对齐；Recall@K 也须说明选组方式，不能逐指标挑不同 gold 组。
- `Eval/utils/hotpotqa_eval.py`：Answer EM/F1、SP EM/Precision/Recall/F1、Joint EM/F1；检查缺失预测和数据集重复题ID，并恢复 title-sentence 映射后评分，写汇总和 coverage 文件。它未强制校验所有显式支持事实属于当前语料，单结果ID回退也须单独审计。
- `Eval/utils/paired_bootstrap.py` 已支持题目配对 CI；存在工具不等于稿件已报告显著性。Qasper 同文档问题相关，期刊评测应补文档聚类重采样和多次模型运行，bootstrap 次数不能当作模型重复次数。

`qasper_support_manifest.py` / `qasper_support_replay.py` 用于 tuning/holdout 和支持集缓存重放。replay 不调用 LLM，但可能首次调用 reranker；当前 `--manifest` 主要记录 hash，不能假定它已强制 dataset 等于 tuning 子集。必须核查实际输入、split 和数据交集；holdout 不用于配置挑选。tuning/holdout manifest 与 DatasetConfig 输入manifest的schema不同，不能直接互换。

`closed_loop_analysis.py` / `render_closed_loop_paper.py` 基于既有日志统计闭环与生成稿件片段。当前 CLI 专用于 Qasper1005/Hotpot1000 和指定方法后缀，非任意子集分析器。gold 仅用于事后评测和可复现案例筛选；历史日志缺首轮答案时不得补报首轮 Answer/Joint F1。

## 常用命令与验证

以下命令在已准备本地数据/模型配置后运行；模板 TODO 必须先填充。单分片显式 1/1。

```powershell
python main.py -c config/evibridge_support_pruned.yaml -d Scripts/cfg/example-Qasper.yaml --nsplit 1 --num 1 index --stage evibridge
python main.py -c config/evibridge_support_pruned.yaml -d Scripts/cfg/example-Qasper.yaml --nsplit 1 --num 1 rag
python -m Scripts.eval.qasper_run_validator --dataset-config Scripts/cfg/example-Qasper.yaml --method evibridge_support_pruned
python -m Scripts.eval.qasper_official --dataset-config Scripts/cfg/example-Qasper.yaml --method evibridge_support_pruned
```

```powershell
python -m unittest discover -s tests -p 'test_evibridge*.py'
python -m unittest discover -s tests -p 'test_qasper*.py'
python -m unittest discover -s tests -p 'test_hotpotqa*.py'
python -m unittest tests.test_dataset_manifest tests.test_paired_bootstrap tests.test_closed_loop_analysis tests.test_closed_loop_paper
```

baseline 独立运行按各自 README，不使用内部 lightweight 方法冒充官方模型。smoke 可用临时 fake SDK 验证工程链路，不能作为模型质量、真实成本或论文主结果。Windows 使用 PowerShell 路径与后台方式，不直接照搬 bash；后台 Start-Process 使用隐藏窗口。

## 后续 JIIS 工作原则

- 先固定投稿稿件、代码版本、配置、dataset manifest 和逐题结果，建立 conference-to-journal 差异表；原会议结果保留，扩展结果单独存放。
- 方法增量围绕可验证的缺失需求、可靠连接、预算下的修复/停止展开；先做诊断试验，避免仅追新论文名称堆模块。
- 比较更新到直接相关文档结构/证据图/充分性方法；官方调用、论文适配、本地实现三类 baseline 身份准确披露。
- 所有调参只在独立开发集；同模型、同语料、同最终上下文预算之外，还要对齐候选召回/reranker/验证/再生成调用预算。
- 同时报告 Answer、Evidence/SP、Joint、充分性误接受/误拒绝、成本和失败率；保留不利指标与案例，不能只选获胜列。
- 真实 PDF / 多模态 / 跨文档主张必须由对应数据和实验支撑；Qasper structured 与 Hotpot distractor 的结果不能直接外推。
- 新功能按修改层补索引、RAG、wiring/resource loader、预处理/官方指标测试及小样本 smoke；prompt/schema、存取兼容和来源映射同步验证。
- 不声称存在统一 requirements.txt / pyproject.toml / 完整锁文件。当前有部分 baseline 的源码版本锁，但不等于主项目完整环境锁；README 仍是原始 BookRAG，需后续专门更新。
- 不做无关乱码注释机械清理或大规模重构；优先沿用 Core/pipelines、Core/rag、Core/provider 与 Eval 的模块边界。
- JIIS 投稿时核实当时官方规则，披露会议版和新增贡献。BigData 在审不能默认同时提交核心重叠的期刊稿；不要编造固定扩展比例、接收率或保证录用。
