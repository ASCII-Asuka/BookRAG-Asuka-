# AGENTS.md

## 适用范围

本文件是 BookRAG-Asuka 仓库的项目级开发指南，适用于整个仓库。若未来某个子目录中出现更近的 `AGENTS.md`，以更近文件中的说明为准。

## 项目概览

本仓库是论文 [BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents](https://arxiv.org/abs/2512.03413) 的方法与实验源码。BookRAG 面向书籍、手册、论文、报告等具有层级结构的复杂文档问答任务，核心思想是构建结构感知的 `BookIndex`，再通过 agent-based retrieval 在文档层级、实体关系和多模态证据之间动态检索。

项目主流程分为两步：

- 离线索引构建：解析 PDF，构建文档树、知识图谱、实体向量库和可选的多模态检索资源。
- 在线问答推理：按配置选择 RAG 策略，加载索引或向量库，逐条处理数据集问题并保存生成结果与检索证据。

## 核心架构

- `main.py`：命令行入口，提供 `index` 和 `rag` 两个子命令；支持通过 `--stage` 分阶段构建 `tree`、`graph`、`vdb`、`mm_reranker` 或重建图向量库。
- `Core/Index/`：BookIndex 的核心数据结构。`Tree.py` 定义文档树节点、节点类型和树结构；`Graph.py` 定义实体、关系、知识图谱和 Tree-KG 映射；`GBCIndex.py` 将树、图和实体向量库组合为 GBC 索引。
- `Core/pipelines/`：离线构建流水线。包括 PDF 解析与清洗、目录/章节结构抽取、树节点构建、节点摘要、知识图谱抽取与精炼、向量库构建。
- `Core/rag/`：在线 RAG 策略实现。`__init__.py` 负责根据配置创建具体 agent；`gbc_*` 文件实现 BookRAG/GBC 的规划、检索和回答；其他文件实现 baseline 或消融策略。
- `Core/provider/`：外部能力封装，包括 LLM、VLM、embedding、reranker、ChromaDB 向量库、MinerU PDF 解析和 token 统计。
- `Core/configs/`：系统、数据集、图、树、向量库、RAG 策略、LLM/VLM/embedding/reranker 等配置模型。
- `Eval/`：实验评测入口和各数据集指标实现。`Eval/evaluation.py` 根据数据集名分发到 MMLongBench、M3DocVQA、Qasper 的评测逻辑。
- `Scripts/`：示例运行脚本、数据预处理 notebook 和数据集配置模板。
- `config/`：论文方法、baseline 和消融实验的系统配置示例。

## 方法流程

### 离线索引构建

`main.py index` 调用 `Core.construct_index` 中的构建函数：

- `tree`：`build_tree_from_pdf()` 使用 MinerU 解析 PDF，经过 `pdf_info_refiner()` 清洗内容，使用 LLM 抽取章节层级，再构建 `DocumentTree`。可选生成节点摘要。
- `graph`：`build_knowledge_graph()` 从文档树节点抽取实体和关系，再通过 `KGRefiner` 做基础或高级实体归并，保存为 `graph_data*.json`。
- `vdb`：`construct_vdb()` 为树节点、普通文本块、RAPTOR 或 BM25 baseline 构建检索索引。
- `mm_reranker`：预计算树节点和问题的多模态 embedding，保存为 `mm_node_metadata.json`、`mm_embeddings.npy`、`mm_question_metadata.json`、`mm_question_embeddings.npy`。
- `rebuild_graph_vdb`：从已有 GBC 索引重建实体向量库。

主要索引产物通常保存在每个文档的 `save_path` 下，包括 `tree.pkl`、`tree.json`、`graph_data.json` 或变体图文件、`kg_vdb`/`kg_vdb_basic`、`Tree_vdb`、日志和 token 成本统计。

### 在线 RAG 推理

`main.py rag` 会按 `Scripts/cfg/*.yaml` 中的数据集配置读取 JSON 数据，按 `doc_uuid` 和 `doc_path` 分组处理每个文档。`Core.inference.inference()` 通过 `prepare_rag_dependencies()` 加载策略所需资源，再由 `create_rag_agent()` 创建具体 RAG agent。每个问题的输出保存到：

- `eval_<dataset>_<method>/query_XXX/result.json`
- `eval_<dataset>_<method>/query_XXX/retrieval_res.json` 或检索节点 JSON
- `eval_<dataset>_<method>/final_results.json`
- `eval_<dataset>_<method>/token_cost.json`

## BookIndex 要点

BookIndex 可理解为 `B = (T, G, M)`：

- `T` 是 `DocumentTree`，以 `TreeNode` 表示标题、正文、图片、表格、公式等 PDF 内容块，并保留父子层级、页码、PDF block ID、图片路径、表格正文等元信息。
- `G` 是 `Graph`，以实体节点和关系边保存跨章节、跨模态的细粒度语义关系。
- `M` 是 Graph-Tree Link，在代码中主要体现为 `Graph.tree2kg` 和实体的 `source_ids`，用于从实体回到证据节点。
- `GBC` 将 `DocumentTree`、`Graph` 和实体向量库组合起来，支持实体映射、图检索、树节点定位和索引持久化。

## RAG 策略

- `gbc`：BookRAG 主方法。使用 `TaskPlanner` 将查询分为 simple、complex、global；通过实体抽取与映射、章节选择、子树投影、图/文本 reranker、skyline 过滤、map/reduce 式生成完成回答。配置类为 `GBCRAGConfig`，支持 `standard`、`wo_plan`、`wo_selector`、`wo_graph`、`wo_text`、`wo_er`、`wo_map` 等变体。
- `graph`：基于 GBC 图和实体向量库的 GraphRAG baseline，使用查询实体映射和 Personalized PageRank 选择相关实体与树节点。
- `gbcvanilla`：直接从树向量库和图向量库取 top-k，再交给 LLM 生成，作为去掉规划与结构化算子的 baseline。
- `vanilla`：文本向量检索、BM25、RAPTOR、PDF vanilla 等常规 RAG baseline。
- `mmr`：多模态 vanilla 检索，使用多模态 embedding 召回文本或图像证据，再用 LLM/VLM 生成。
- `traverse`：树遍历 agent，由 LLM 根据当前节点摘要和子节点候选逐层选择路径。

新增或修改策略时，应同步更新：

- `Core/configs/rag/*.py` 中的配置模型和 discriminator 字段。
- `Core/configs/rag/__init__.py` 中的策略联合类型。
- `Core/rag/__init__.py` 中的 `create_rag_agent()` 分发逻辑。
- `Core/utils/resource_loader.py` 中的依赖加载逻辑。
- 对应 `config/*.yaml` 示例。

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
  "doc_path": "PDF 文档路径"
}
```

不同数据集可以保留额外字段，例如 `answer_format`、证据信息或原始元数据。`main.py` 依赖 `doc_uuid` 和 `doc_path` 分组，因此这两个字段缺失会破坏批量索引和批量推理流程。

## 运行命令

示例脚本位于 `Scripts/`，但其中路径和配置名需要按本地环境调整。常用命令如下：

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

运行前请确认：

- MinerU 已按 README 指向的官方说明安装并可解析 PDF。
- 所需 LLM/VLM/embedding/reranker 服务已经启动，且配置中的 `api_base`、GPU `device` 与本机资源一致。
- `Eval/utils/api.txt` 或评测用 API 配置已准备好，评测阶段会使用 LLM 做答案抽取。
- 不要把本地绝对路径、API key、模型服务内网地址或大体积运行产物提交为通用配置。

## 实验与评测

论文实验涉及 MMLongBench、M3DocVQA 和 Qasper 三个复杂文档 QA 数据集。评测入口是 `Eval/evaluation.py`：

- `MMLongBench`：使用 EM、F1、token F1 和 LLM 抽取分数等结果字段。
- `M3DocVQA`：使用 EM、F1 和 LLM 抽取分数。
- `Qasper`：使用 Accuracy、F1 和 LLM 抽取分数，并区分 answerable 样本。

评测会读取每个文档目录下 `eval_<dataset>_<method>/final_results.json`，写回单文档 `eval.json`，并在 `working_dir/0_results/` 下保存汇总结果和 `.score.json`。评测方法名必须和推理输出目录中的 method 后缀一致，例如 `gbc_standard`、`vanilla`、`bm25` 等。

## 开发注意事项

- 当前仓库未提供 `requirements.txt`、`pyproject.toml` 或依赖锁文件；不要在文档中声称存在固定依赖清单。README 明确提示完整环境仍待补充，PDF 解析相关问题优先参考 MinerU。
- 保持现有配置风格：顶层系统配置使用 Pydantic `BaseModel`，部分子配置使用 dataclass，RAG 策略通过 Pydantic discriminator 分发。
- 运行索引和评测会生成大量文件，包括 pickle、ChromaDB、JSON、日志、图片、embedding 和 token cost；除非任务明确要求，不要提交这些产物。
- 修改索引结构时，要同时考虑保存/加载兼容性：`tree.pkl`、`tree.json`、`graph_data*.json`、`kg_vdb`、`Tree_vdb` 的读取逻辑分散在多个模块中。
- 修改 prompt 或 structured output schema 时，要检查对应的 Pydantic 模型和 `llm.get_json_completion()` 调用，避免输出字段不匹配。
- 表格和图片节点通常既有文本描述也有 `img_path`，回答阶段会分别路由到 LLM 或 VLM；改动时不要破坏多模态证据路径。
- 文件中存在少量历史乱码注释和日志字符串；非任务相关时不要做大规模机械清理，避免引入无关 diff。
- `Scripts/example-*.sh` 是 Linux/bash 风格示例；在 Windows/PowerShell 环境运行时需要改写路径和后台执行方式。

## 代码变更原则

- 优先沿用现有模块边界：索引构建放在 `Core/pipelines/`，运行编排放在 `Core/construct_index.py` 或 `Core/inference.py`，策略实现放在 `Core/rag/`，外部模型封装放在 `Core/provider/`。
- 新增实验配置时，复制最接近的 `config/*.yaml` 并只改必要字段；不要把本机私有路径和密钥写入默认配置。
- 新增数据集时，先在 `Scripts/preprocess/` 生成统一 JSON，再新增 `Scripts/cfg/*.yaml`，最后在 `Eval/evaluation.py` 中接入评测逻辑。
- 新增 baseline 时，明确它依赖的是树、图、文本向量库、BM25 还是多模态向量库，并在资源加载、agent factory 和示例配置中保持一致。
