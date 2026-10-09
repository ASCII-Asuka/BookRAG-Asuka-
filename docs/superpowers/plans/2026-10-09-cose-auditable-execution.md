# CoSE-RAG 可审计判定与生成上下文实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. 不创建worktree，不自动commit；直接dev，保留已有文档改动及旧实验结果。

**Goal:** 在显式新模式中保持规则接受约束、最终support属于保留答案的实际生成上下文，并记录每次运行及缓存来源。

**Architecture:** 沿用现有索引/检索/预算/支持控制器。兼容模式保留现有判定与支持行为，新模式使用独立method suffix。verifier记录rule/LLM/final；generation记录initial/fallback/regeneration成功与失败，只有成功响应对应上下文才能成为保留答案的来源；运行manifest按UUID不可覆写。

**Tech Stack:** Python、Pydantic、unittest、stdlib SHA256/JSON/UUID，无真实模型调用。

## 范围与设计

- `verifier_acceptance_mode=legacy_hybrid`、`support_context_policy=legacy`为缺省兼容值；新增`rule_only`与`strict`。新mode的suffix追加`_rule_only`/`_strict_context`，不能写到旧主结果目录。
- `rule_only`允许LLM修订诊断，接受值、五项数值指标及noise_warning保留规则结果；记录调用跳过/成功/失败，不以最终值相同推断未调用。
- `generation_provenance`保存schema版本、各call的stage、prompt hash、按序block IDs/text hashes、响应hash、success/error_type，以及保留答案的stage和上下文；不保存完整prompt/响应或异常消息。
- `strict`最终support仅保留当前成功响应上下文内的合格块。未用于生成的补充证据记录为posthoc，不能进入正式support。coverage_prune在允许再生成且出现外部补充时真正发起再生成；失败保留draft实际上下文；预算裁剪、无效引用、fallback并集都在最终保存前校验。
- `answer_context_block_ids`在strict下对应实际保留答案上下文；另存planned context，避免把计划当调用。legacy下保持旧字段语义，新generation_provenance独立反映真实调用。
- 每次run_rag写`.runs/<UUID>/manifest.json`与逐题事件；新result指向该来源。复用旧result不得回填今天runid，记录reused/origin unknown。配置/数据/源码以hash关联；凭据、地址、绝对路径不进入manifest明文。
- 不改规则代理的研究定义，不引入语义judge/校准，不声称新研究贡献或质量提升。不重写旧PDF/结果，不复跑真实模型。

## Task 1：判定模式与配置

Files: `Core/rag/evibridge_verifier.py`、`Core/configs/rag/evibridge_config.py`、`config/evibridge.yaml`、新`config/evibridge_auditable.yaml`、新`tests/test_evibridge_verifier_modes.py`。

- [x] RED：最简LLM接受、软缺失组合、规则指标保持、异常来源、legacy接受覆盖、配置/后缀隔离。
- [x] GREEN：实现acceptance_mode及每次清空的last_trace；不改变旧SufficiencyVerdict字段。
- [x] 验证：14项verifier新测试及既有模块/config wiring测试通过。

## Task 2：实际生成上下文与支持集约束

Files: `Core/rag/evibridge_rag.py`、可选新`Core/rag/evibridge_generation_provenance.py`、新`tests/test_evibridge_generation_context.py`。

- [x] RED：coverage_prune开/关再生成、失败保留draft、预算裁剪后无效引用/非JSON、fallback上下文替换与失败、legacy保留旧输出、逐题记录不残留、prompt hash与实际mock输入相同。
- [x] GREEN：统一记录实际LLM尝试，只有成功更新retained context；strict finalization裁剪提交support并保存posthoc/validation。
- [x] 接线：RAG向verifier传模式；每轮deepcopy verification_trace；持久化generation/support记录并提供last_generation_provenance与last_support_context_validation给inference。
- [x] 验证：12项生成上下文新测试及完整EviBridge回归通过；实际chain保留metadata，Hotpot导出facts恰对应正式support。

## Task 3：不可覆写运行来源

Files: 新`Core/utils/run_provenance.py`、`Core/inference.py`、`main.py`、新`tests/test_run_provenance.py`。

- [x] RED：secret/path/URL保护、未知revision、未commit源码hash、不同调用不同folder、ordered输入hash、cache字节不变、freshresult关联、failed事件不泄漏异常消息。
- [x] GREEN：纯stdlib来源模块与main/inference兼容接线，按实际输出suffix保存；不从旧缓存推断新配置执行。明确输入错配停止，缺历史身份为partial/unknown，mkdir/cache/finalize失败记录完整。
- [x] 验证：28项来源新测试及wiring/resource loader/数据manifest相关检查通过；配置自定义/外部VDB纳入hash，路径按实际loader原字符串解析，索引覆盖保守partial/unknown。

## Task 4：集成与review

- [x] 逐项spec review通过后做独立code-quality review；修正并复核缓存身份、Hotpot映射、索引路径和失败记录问题，最终质量复核通过。
- [x] 完整EviBridge121项、来源28项、Qasper47项、可运行Hotpot19项、manifest/bootstrap/closed-loop18项均通过；唯一Hotpot parquet固定采样测试因本地缺PyArrow未通过，已明确记录，无模型调用。
- [x] 用临时fake SDK和小数据验证新模式输出、manifest、引用来源、有效缓存与错配停止；产物只存ignored runs。
- [x] 更新AGENTS及JIIS总计划，记录legacy与strict、实际验证范围及未知项；增加独立标注规范草案与开发数据候选核查。
- [x] `git diff --check`及语法检查通过；7301项历史文件hash一致（7296实验产物、4审计文件、投稿PDF），无commit/push。

详细证据与工程边界见 [验证记录](../../research/jiis/2026-10-09-auditable-execution-verification.md)。本步骤完成工程准备；PyArrow依赖问题和真实研究任务仍单独记录，不将其称为全部项目测试通过。

## 后续研究交付

本步骤仅提供可信记录与代码约束。独立充分性pilot需要真实标注者或明确独立judge；先冻结开发源题与标注规范，再生产标注包。不能把当前已有validation题重新称为未见开发/确认集，也不能以gold evidence被删自动标不充分。
