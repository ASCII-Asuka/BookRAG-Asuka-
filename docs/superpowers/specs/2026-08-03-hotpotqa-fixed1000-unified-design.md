# HotpotQA fixed1000 统一 11 方法实验设计

## 目标

在固定的 HotpotQA distractor validation seed 42、1000 题样本上，完成与 Qasper 主表一致的 11 个方法：Abstract-only、BM25、Dense、Hybrid、BM25+Reranker、RAPTOR、Full-document、LongRAG、EviBridge、IRCoT、ReAct。所有方法共享同一数据 manifest、答案生成模型、输出规范和 HotpotQA 官方评测器。

## 方案选择

采用增量复用方案。BM25+Reranker、Full-document、LongRAG、EviBridge、IRCoT 已有 1000/1000 完整输出；ReAct 有 998/1000 输出；其余五个方法尚未运行。已有输出先经过严格覆盖校验，校验通过则复用；ReAct 仅恢复缺失两题；只对 Abstract-only、BM25、Dense、Hybrid、RAPTOR 做完整推理。

未采用的方案：全部 11 方法重新推理会提供最强的运行时间一致性，但重复模型调用成本高且不会改变固定代码、模型与数据下的公平性；直接拼接旧表而不做严格校验成本最低，但不能保证逐题覆盖和当前评测版本一致。

## 数据与方法身份

- 数据集配置：`runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml`
- 数据 manifest：1000 题，799 bridge/hard、201 comparison/hard，gold supporting facts 映射覆盖 2435/2435。
- 工作目录：`runs/hotpotqa_evibridge/work_validation_fixed1000_seed42`
- 统一方法后缀：`abstract_only`、`bm25`、`vanilla`、`hybrid`、`bm25_rerank`、`raptor`、`full_document`、`longrag`、`evibridge`、`ircot`、`react`。
- 所有新增推理沿用 `Qwen/Qwen3-VL-8B-Instruct`、temperature 0.1 和现有 SiliconFlow 服务配置。

## 索引与推理数据流

BM25、BM25+Reranker、LongRAG、IRCoT 和 ReAct 复用每题已有的 sentence-level BM25 索引。Dense 与 Hybrid 共享 `dense_paragraph_vdb`；RAPTOR 使用独立 `raptor_vdb`。Dense 与 RAPTOR 索引均以 HotpotQA sentence metadata（title、sentence id）为证据映射来源，构建完成后必须达到 1000/1000。

编排器按“索引校验 → 缺失索引构建 → ReAct 两题恢复 → 五个新方法推理 → 每方法严格评测 → bootstrap → 主表”的顺序运行。每一步写原子状态文件，并根据已有完整输出跳过已完成任务，因而可以安全续跑而不重复模型调用。

## 评测与验收

论文主表固定报告 Answer EM、Answer F1、Supporting Fact Precision、Supporting Fact Recall、Supporting Fact F1、Joint F1，以及逐题 bootstrap 95% CI（10000 次、seed 42）。底层官方评测文件仍保留 SP EM 和 Joint EM，但二者不进入论文主表。严格校验要求：1000/1000 问题、1000/1000 文档、零缺失预测、每题 question id 唯一、supporting fact 均可映射为合法 title/sentence pair。

最终生成 Markdown、JSON 和逐方法评测文件。若任一方法覆盖不足或评测失败，campaign 标记为 failed 并停止后续主表生成；已完成的方法和索引保留用于恢复。

## 安全与兼容性

不建立 worktree，直接使用 `dev` 分支。实验配置、日志和结果保存在已忽略的 `runs/` 下；不覆盖已有五方法的逐题输出。新配置只继承现有私有配置，不向跟踪文件写入 API key。
