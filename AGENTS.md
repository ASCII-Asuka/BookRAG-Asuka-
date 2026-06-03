# AGENTS.md

## 适用范围

本文件是 BookRAG-Asuka 仓库的项目级开发指南，适用于整个仓库。若未来某个子目录中出现更近的 `AGENTS.md`，以更近文件中的说明为准。

## 项目概览

本仓库最初来自论文 **BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents** 的方法与实验源码，面向书籍、手册、论文、报告等具有层级结构的复杂文档问答任务。

当前仓库已经在 BookRAG/GBC 与 HRI 的基础上新增 **EviBridge-RAG: Sufficiency-Guided Evidence Bridging for Complex Document Question Answering**。后续论文主线应以 EviBridge-RAG 为核心：复杂文档问答的关键不是检索若干相似 chunk，而是构造一个相关、连通、覆盖充分、噪声可控、可追溯的证据集，并由充分性验证器决定是否继续桥接检索。

项目主流程分为两步：

- 离线索引构建：解析或转换文档，构建 `DocumentTree`、BookRAG/GBC 图索引、HRI 索引或 EviBridge 证据桥接图。
- 在线问答推理：按配置选择 RAG 策略，加载索引或向量库，逐条处理数据集问题并保存生成结果、检索证据和评测中间产物。

## 核心架构

- `main.py`：命令行入口，提供 `index` 和 `rag` 两个子命令；`index --stage` 支持 `tree`、`graph`、`vdb`、`hri`、`evibridge`、`mm_reranker`、`rebuild_graph_vdb` 和 `all`。
- `Core/Index/`：核心索引结构。`Tree.py` 定义 `DocumentTree` 和 `TreeNode`；`Graph.py` 定义实体关系图；`GBCIndex.py` 组合树、图和实体向量库；`HRIIndex.py` 保存水利规范证据锚点；`EvidenceBridgeIndex.py` 保存 EviBridge 的 evidence block 与 typed bridge。
- `Core/pipelines/`：离线构建流水线。PDF 路线走 MinerU 与 `doc_tree_builder.py`；HRI 路线走 `hri_builder.py`；EviBridge 路线走 `evibridge_builder.py`。
- `Core/rag/`：在线 RAG 策略实现。`gbc_*` 文件实现 BookRAG/GBC；`hri_*` 文件实现 HRI；`evibridge_*` 文件实现 EviBridge 的 demand parsing、typed PPR、selector、verifier 和 answer generation。
- `Core/configs/`：系统、数据集、图、树、向量库、RAG 策略、LLM/VLM/embedding/reranker 等配置模型。新增策略时必须同步更新 RAG config discriminator。
- `Core/provider/`：外部能力封装，包括 LLM、VLM、embedding、reranker、ChromaDB 向量库、MinerU PDF 解析和 token 统计。
- `Eval/`：实验评测入口和各数据集指标实现。`Eval/evaluation.py` 按数据集名懒加载具体评测器，避免单个数据集评测被其他数据集的可选依赖阻塞。
- `Scripts/`：示例运行脚本、数据预处理 notebook、数据集配置模板和预处理脚本。EviBridge 的 Qasper 结构化预处理脚本为 `Scripts/preprocess/qasper_evibridge.py`。
- `config/`：论文方法、baseline 和消融实验的系统配置示例。EviBridge 主配置为 `config/evibridge.yaml`，消融配置包括 `config/evibridge_wo_*.yaml` 和 `config/evibridge_static_topk.yaml`。

## 方法流程

### BookRAG / GBC 离线索引

`main.py index` 调用 `Core.construct_index` 中的构建函数：

- `tree`：`build_tree_from_pdf()` 使用 MinerU 解析 PDF，清洗内容，抽取章节层级，再构建 `DocumentTree`。如果 `save_path/tree.pkl` 已存在，当前实现会优先直接加载已有树，避免不必要地导入或调用 MinerU。
- `graph`：`build_knowledge_graph()` 从文档树节点抽取实体和关系，再通过 `KGRefiner` 做实体归并，保存为 `graph_data*.json`。
- `vdb`：`construct_vdb()` 为树节点、普通文本块、RAPTOR 或 BM25 baseline 构建检索索引。
- `mm_reranker`：预计算树节点和问题的多模态 embedding。
- `rebuild_graph_vdb`：从已有 GBC 索引重建实体向量库。

主要索引产物通常保存在每个文档的 `save_path` 下，包括 `tree.pkl`、`tree.json`、`graph_data*.json`、`kg_vdb`/`kg_vdb_basic`、`Tree_vdb`、日志和 token 成本统计。

### EviBridge 离线索引

`main.py index --stage evibridge` 调用 `construct_evibridge_index()`，再进入 `Core/pipelines/evibridge_builder.py`。

EviBridgeIndex 的核心结构是多视图证据桥接图：

- `EvidenceBlock`：证据节点，类型包括 `paragraph`、`table`、`figure`、`caption`、`title`、`summary`、`entity`、`patch`、`equation`、`unknown`。
- `EvidenceBridge`：证据边，顶层类型固定为 `context`、`semantic`、`hierarchy`。
- 细分 `relation_type` 写在 bridge 中，例如 `parent_child`、`same_section`、`caption_of`、`mentions_entity`、`entity_cooccur`、`summary_of`、`patch_contains`、`shared_terms`。
- 持久化产物包括 `evibridge_index.json` 和 `evibridge_bm25.pkl`。

EviBridge 索引可以来自两条路线：

- PDF 路线：PDF -> MinerU -> `DocumentTree` -> `EvidenceBridgeIndex`。
- 结构化数据路线：例如 Qasper raw JSON -> pseudo `DocumentTree` -> `EvidenceBridgeIndex`。这条路线用于公开数据集实验时可避免强依赖 PDF 解析。

### 在线 RAG 推理

`main.py rag` 会按 `Scripts/cfg/*.yaml` 中的数据集配置读取 JSON 数据，按 `doc_uuid` 和 `doc_path` 分组处理每个文档。`Core.inference.inference()` 通过 `prepare_rag_dependencies()` 加载策略所需资源，再由 `create_rag_agent()` 创建具体 RAG agent。

每个问题的常见输出保存在：

- `eval_<dataset>_<method>/query_XXX/result.json`
- `eval_<dataset>_<method>/query_XXX/retrieval_res.json`
- `eval_<dataset>_<method>/query_XXX/evidence_chain.json`
- `eval_<dataset>_<method>/final_results.json`
- `eval_<dataset>_<method>/token_cost.json`

EviBridge 额外要求：

- `result.json` 保留 `retrieved_node_ids`，并新增 `retrieved_block_ids`，避免破坏旧评测。
- `retrieval_res.json` 保存 seed、typed PPR 总分、typed PPR 分类型贡献、connector path、selector 分数和 verifier 轮次。
- `evidence_chain.json` 保存 ordered evidence、连接边、connector path 和 verifier 信息。

## 主要索引结构

### BookIndex / GBC

BookIndex 可理解为 `B = (T, G, M)`：

- `T` 是 `DocumentTree`，以 `TreeNode` 表示标题、正文、图片、表格、公式等 PDF 内容块，并保留父子层级、页码、PDF block ID、图片路径、表格正文等元信息。
- `G` 是 `Graph`，以实体节点和关系边保存跨章节、跨模态的细粒度语义关系。
- `M` 是 Graph-Tree Link，在代码中主要体现为 `Graph.tree2kg` 和实体的 `source_ids`。
- `GBC` 将 `DocumentTree`、`Graph` 和实体向量库组合起来，支持实体映射、图检索、树节点定位和索引持久化。

### EvidenceBridgeIndex

EviBridgeIndex 可理解为 `G_EB=(V_b,V_p,V_e,V_h,E_c,E_s,E_h,M)`：

- `V_b`：原始证据块，例如 paragraph/table/figure/title。
- `V_p`：patch 证据块，聚合一个章节内的局部上下文。
- `V_e`：entity 证据块，作为 semantic bridge 的轻量中间节点。
- `V_h`：summary/title 等层级证据块。
- `E_c`：context bridge，例如前后文、同章节、引用表格、图表说明。
- `E_s`：semantic bridge，例如共享术语、实体提及、实体共现。
- `E_h`：hierarchy bridge，例如属于章节、章节包含、summary/patch 与原块的层级关系。
- `M`：证据块到原始 `TreeNode`、页码、标题路径、PDF 元信息的映射。

## RAG 策略

- `gbc`：BookRAG 主方法。使用 `TaskPlanner` 将查询分为 simple、complex、global；通过实体抽取与映射、章节选择、子树投影、图/文本 reranker、skyline 过滤、map/reduce 式生成完成回答。
- `graph`：基于 GBC 图和实体向量库的 GraphRAG baseline，使用查询实体映射和 Personalized PageRank 选择相关实体与树节点。
- `gbcvanilla`：直接从树向量库和图向量库取 top-k，再交给 LLM 生成，作为去掉规划与结构化算子的 baseline。
- `vanilla`：文本向量检索、BM25、RAPTOR、PDF vanilla 等常规 RAG baseline。
- `mmr`：多模态 vanilla 检索，使用多模态 embedding 召回文本或图像证据，再用 LLM/VLM 生成。
- `traverse`：树遍历 agent，由 LLM 根据当前节点摘要和子节点候选逐层选择路径。
- `hri`：水利规范垂直策略，保留为领域迁移案例。HRI 不再作为 EviBridge 论文主方法，但可将规范关系映射为 EviBridge 的 domain-specific semantic edges。
- `evibridge`：EviBridge-RAG 主策略。流程为 evidence demand parsing -> 多粒度 seed recall -> typed PPR / shortest-path bridge expansion -> budgeted selector -> sufficiency verifier loop -> answer generation。

新增或修改策略时，应同步更新：

- `Core/configs/rag/*.py` 中的配置模型和 discriminator 字段。
- `Core/configs/rag/__init__.py` 中的策略联合类型。
- `Core/rag/__init__.py` 中的 `create_rag_agent()` 分发逻辑。
- `Core/utils/resource_loader.py` 中的依赖加载逻辑。
- 对应 `config/*.yaml` 示例。
- 对应单元测试和 smoke 测试。

## EviBridge 当前实现要点

EviBridge-RAG 已有实现文件：

- `Core/Index/EvidenceBridgeIndex.py`：定义 `EvidenceBlock`、`EvidenceBridge`、`EvidenceBridgeIndex`，支持 JSON 持久化、BM25 构建、vector 文档导出和 typed graph 导出。
- `Core/pipelines/evibridge_builder.py`：从 `DocumentTree` 构建 EviBridgeIndex，并按配置可选构建向量库。
- `Core/configs/rag/evibridge_config.py`：EviBridge 策略配置和 ablation 配置字段。
- `Core/rag/evibridge_demand.py`：规则/LLM/hybrid 的 evidence demand parser。
- `Core/rag/evibridge_ppr.py`：typed PPR、分类型 PPR 贡献、shortest-path connector 和路径导出。
- `Core/rag/evibridge_selector.py`：预算化证据选择，包含 relevance、coverage、connectivity、diversity、bridge diversity、redundancy 和 token cost。
- `Core/rag/evibridge_verifier.py`：规则 + 可选 LLM 的证据充分性验证器，输出 `missing_types`、`missing_bridge_types`、`noise_warning`、`next_action`。
- `Core/rag/evibridge_rag.py`：完整 EviBridge RAG agent，负责检索闭环、生成 prompt 和输出文件。
- `Scripts/preprocess/qasper_evibridge.py`：Qasper raw JSON 到统一 JSON 的转换，以及 Qasper 结构化论文到 pseudo `DocumentTree` / EviBridgeIndex 的构建。

EviBridge 消融配置包括：

- `config/evibridge_wo_multiseed.yaml`
- `config/evibridge_wo_context.yaml`
- `config/evibridge_wo_semantic.yaml`
- `config/evibridge_wo_hierarchy.yaml`
- `config/evibridge_wo_typed_weights.yaml`
- `config/evibridge_wo_selector.yaml`
- `config/evibridge_wo_verifier.yaml`
- `config/evibridge_static_topk.yaml`

## 配置与数据

系统配置位于 `config/*.yaml`，数据集配置位于 `Scripts/cfg/*.yaml`。运行前必须填充配置中的 `TODO`，尤其是：

- `pdf_path`、`save_path`
- 数据集的 `dataset_path`、`working_dir`、`dataset_name`
- LLM/VLM 的 `model_name`、`api_key`、`api_base`、`backend`
- MinerU 的 `backend`、`server_url`、`lang`
- embedding/reranker 的模型、设备和服务地址

统一数据集 JSON 采用列表格式，每条样本至少应包含：

```json
{
  "question": "问题文本",
  "answer": "标准答案或答案结构",
  "doc_uuid": "文档唯一 ID",
  "doc_path": "PDF 文档路径或文档标识路径"
}
```

不同数据集可以保留额外字段，例如 `answer_format`、`qasper_question_id`、`evidence_block_ids`、`evidence_paragraph_ids`、证据信息或原始元数据。`main.py` 依赖 `doc_uuid` 和 `doc_path` 分组，因此这两个字段缺失会破坏批量索引和批量推理流程。

当前仓库的数据状态：

- `data/documents/hydro/` 中有本地水利 PDF，用于 HRI/水利案例实验。
- `Scripts/data/` 中有 HRI 问答 JSON 和 smoke 数据。注意不要误删或覆盖这些本地数据文件。
- `Scripts/cfg/hri_*.yaml` 是本地 HRI 实验配置。
- `Scripts/cfg/example-Qasper.yaml`、`example-MMLongBench.yaml`、`example-m3docVQA.yaml` 只是模板，默认仍含 `TODO`，不是可直接跑的公开数据集配置。
- 公开 Qasper/MMLongBench/M3DocVQA 原始数据通常不应提交到仓库，需要在本地下载后通过 `Scripts/preprocess/` 转为统一 JSON。
- `runs/` 是本地运行产物目录，已被 `.gitignore` 忽略。不要把 `runs/` 下的索引、日志、结果、fake SDK、临时 smoke 数据提交。

## 运行命令

常用命令如下：

```bash
python main.py -c config/gbc.yaml -d Scripts/cfg/example-m3docVQA.yaml index --stage all
python main.py -c config/gbc.yaml -d Scripts/cfg/example-m3docVQA.yaml rag
python Eval/evaluation.py -d Scripts/cfg/example-m3docVQA.yaml --method gbc_standard
```

分片运行使用 `--nsplit` 和 `--num`：

```bash
python main.py -c config/gbc.yaml -d Scripts/cfg/example-m3docVQA.yaml --nsplit 2 --num 1 index --stage graph
python main.py -c config/gbc.yaml -d Scripts/cfg/example-m3docVQA.yaml --nsplit 2 --num 1 rag
```

EviBridge 常用命令：

```bash
python main.py -c config/evibridge.yaml -d Scripts/cfg/example-Qasper.yaml --nsplit 1 --num 1 index --stage evibridge
python main.py -c config/evibridge.yaml -d Scripts/cfg/example-Qasper.yaml --nsplit 1 --num 1 rag
python Eval/evaluation.py -d Scripts/cfg/example-Qasper.yaml --method evibridge --max_workers 1
```

Windows/PowerShell 环境运行时注意路径分隔符和后台服务启动方式，不能直接照搬 Linux shell 脚本。

运行前请确认：

- MinerU 已安装并可解析 PDF；若走 Qasper pseudo `DocumentTree` 路线，可以先生成 `tree.pkl`，再让 `index --stage evibridge` 加载已有树。
- 所需 LLM/VLM/embedding/reranker 服务已经启动，且配置中的 `api_base`、GPU `device` 与本机资源一致。
- `Eval/utils/api.txt` 或评测用 API 配置已准备好，评测阶段会使用 LLM 做答案抽取。
- 不要把本地绝对路径、API key、模型服务内网地址或大体积运行产物提交为通用配置。

## Qasper 与 EviBridge 实验流程

### Smoke 流程

Smoke run 只用于验证链路是否可跑通，不代表真实模型质量。

已验证过的最小流程是：

1. 构造极小 Qasper raw JSON。
2. 使用 `Scripts/preprocess/qasper_evibridge.py` 转为统一 JSON。
3. 从 Qasper `full_text` 生成 pseudo `DocumentTree` 并保存 `tree.pkl`。
4. 运行 `main.py index --stage evibridge`，生成 `evibridge_index.json` 和 `evibridge_bm25.pkl`。
5. 运行 `main.py rag`，生成 `result.json`、`retrieval_res.json`、`evidence_chain.json` 和 `final_results.json`。
6. 运行 `Eval/evaluation.py --method evibridge`，生成 `eval.json`、`final_eval_*.json` 和 `.score.json`。

若当前机器没有真实 `openai`/`ollama` SDK 或模型服务，可以用临时 fake SDK 验证流程，但这种 smoke 只能证明工程链路，不证明答案质量。

### 真实 Qasper 实验流程

真实实验应按以下顺序执行：

1. 下载真实 Qasper raw JSON 和对应 PDF。
2. 决定索引路线：
   - 结构化路线：Qasper raw JSON -> pseudo `DocumentTree` -> EviBridgeIndex。
   - PDF 路线：PDF -> MinerU -> `DocumentTree` -> EviBridgeIndex。
3. 预处理为统一数据：

```bash
python Scripts/preprocess/qasper_evibridge.py --raw <qasper_raw.json> --output <qasper_unified.json> --pdf-dir <qasper_pdf_dir>
```

4. 准备数据集配置，例如 `Scripts/cfg/Qasper.yaml`，字段为：

```yaml
dataset_path: <qasper_unified.json>
working_dir: <experiment_working_dir>
dataset_name: qasper
```

5. 准备系统配置，例如复制 `config/evibridge.yaml`，填入真实 LLM/VLM/embedding 服务。第一轮建议先设置 `enable_vector_recall: false`，优先验证 BM25 + EviBridge 图扩展。
6. 构建索引：

```bash
python main.py -c <evibridge_config.yaml> -d <qasper_dataset_config.yaml> --nsplit 1 --num 1 index --stage evibridge
```

7. 运行推理：

```bash
python main.py -c <evibridge_config.yaml> -d <qasper_dataset_config.yaml> --nsplit 1 --num 1 rag
```

8. 运行评测：

```bash
python Eval/evaluation.py -d <qasper_dataset_config.yaml> --method evibridge --max_workers 1
```

9. 跑 baseline 和消融，至少包括 BM25/Dense/Hybrid、BookRAG/GBC、EviBridge、`wo_context`、`wo_semantic`、`wo_hierarchy`、`wo_multiseed`、`wo_selector`、`wo_verifier`、`static_topk`。

真实实验优先关注：

- Answer F1 / Accuracy
- Evidence F1 / Evidence Recall
- Path Connectivity
- Bridge Coverage
- Noise Ratio
- Verifier Iterations
- Token Cost / Time Cost

## 实验与评测

论文实验涉及 MMLongBench、M3DocVQA、Qasper 和 HRI 案例。评测入口是 `Eval/evaluation.py`：

- `MMLongBench`：使用 EM、F1、token F1 和 LLM 抽取分数等结果字段。
- `M3DocVQA`：使用 EM、F1 和 LLM 抽取分数。
- `Qasper`：使用 Accuracy、F1 和 LLM 抽取分数，并额外读取 EviBridge 的 evidence metrics。若 gold evidence id 存在，优先用 `evidence_block_ids`/`evidence_paragraph_ids` 对齐，否则退化为文本匹配。
- `HRI_*`：使用水利案例评测逻辑，可作为 EviBridge 的垂直迁移案例补充。

评测会读取每个文档目录下 `eval_<dataset>_<method>/final_results.json`，写回单文档 `eval.json`，并在 `working_dir/0_results/` 下保存汇总结果和 `.score.json`。评测方法名必须和推理输出目录中的 method 后缀一致，例如 `gbc_standard`、`vanilla`、`bm25`、`hri`、`evibridge`、`evibridge_wo_context_edges` 等。

## 开发注意事项

- 当前仓库未提供 `requirements.txt`、`pyproject.toml` 或依赖锁文件；不要在文档中声称存在固定依赖清单。README 明确提示完整环境仍待补充，PDF 解析相关问题优先参考 MinerU。
- 部分依赖是可选依赖。入口文件应尽量懒加载数据集评测器、PDF parser、策略实现和模型后端，避免运行某个轻量流程时被无关依赖阻塞。
- 保持现有配置风格：顶层系统配置使用 Pydantic `BaseModel`，部分子配置使用 dataclass，RAG 策略通过 Pydantic discriminator 分发。
- 运行索引和评测会生成大量文件，包括 pickle、ChromaDB、JSON、日志、图片、embedding 和 token cost；除非任务明确要求，不要提交这些产物。
- 修改索引结构时，要同时考虑保存/加载兼容性：`tree.pkl`、`tree.json`、`graph_data*.json`、`kg_vdb`、`Tree_vdb`、`hri_index.json`、`evibridge_index.json` 的读取逻辑分散在多个模块中。
- 修改 prompt 或 structured output schema 时，要检查对应的 Pydantic 模型和 `llm.get_json_completion()` 调用，避免输出字段不匹配。
- 表格和图片节点通常既有文本描述也有 `img_path`，回答阶段会分别路由到 LLM 或 VLM；改动时不要破坏多模态证据路径。
- 文件中存在少量历史乱码注释和日志字符串；非任务相关时不要做大规模机械清理，避免引入无关 diff。
- `Scripts/example-*.sh` 是 Linux/bash 风格示例；在 Windows/PowerShell 环境运行时需要改写路径和后台执行方式。
- 当前仓库可能存在未提交的本地实验产物和配置改动。修改文件前先看 `git status --short`，不要回滚不是自己造成的改动。

## 代码变更原则

- 优先沿用现有模块边界：索引构建放在 `Core/pipelines/`，运行编排放在 `Core/construct_index.py` 或 `Core/inference.py`，策略实现放在 `Core/rag/`，外部模型封装放在 `Core/provider/`。
- 新增实验配置时，复制最接近的 `config/*.yaml` 并只改必要字段；不要把本机私有路径和密钥写入默认配置。
- 新增数据集时，先在 `Scripts/preprocess/` 生成统一 JSON，再新增 `Scripts/cfg/*.yaml`，最后在 `Eval/evaluation.py` 中接入评测逻辑。
- 新增 baseline 时，明确它依赖的是树、图、文本向量库、BM25 还是多模态向量库，并在资源加载、agent factory 和示例配置中保持一致。
- 新增或修改输出字段时，优先保持向后兼容。EviBridge 可以新增 `retrieved_block_ids`、`typed_ppr_score_parts`、`connector_paths` 等字段，但不应删除旧的 `retrieved_node_ids`。
- 新增 EviBridge 功能时，应至少补充：
  - 索引层单元测试。
  - RAG 模块单元测试。
  - wiring/resource loader 测试。
  - Qasper evidence metric 或 preprocess 测试。
  - 一个小样本 smoke 流程说明或可复现命令。
