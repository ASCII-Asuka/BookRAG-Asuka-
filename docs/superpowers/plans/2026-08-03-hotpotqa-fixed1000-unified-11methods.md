# HotpotQA fixed1000 Unified 11-Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 HotpotQA 1000 题上完成并严格评测 11 个统一方法，复用已完成结果且不重复昂贵模型调用。

**Architecture:** 使用现有 `main.py` 索引/推理入口和 `Eval/evaluation.py` 官方 HotpotQA 评测器；新增仅位于 `runs/` 的可恢复 campaign 编排与 bootstrap/主表工具。Dense/Hybrid 共享 sentence-level dense index，RAPTOR 独立构建层次索引。

**Tech Stack:** PowerShell、Python 3、Pydantic YAML 配置、ChromaDB、HotpotQA official-style metrics、paired bootstrap。

---

### Task 1: 固化 campaign 输入与覆盖审计

**Files:**
- Create: `runs/_scripts/hotpotqa_fixed1000_unified_campaign.ps1`
- Create: `runs/_scripts/hotpotqa_unified_audit.py`
- Test: `runs/_scripts/test_hotpotqa_unified_audit.py`

- [ ] **Step 1: 写覆盖审计失败测试**

构造包含缺失 question id、重复预测和非法 supporting fact 的临时结果，断言审计器分别返回明确错误。

- [ ] **Step 2: 运行测试并确认因审计器尚不存在而失败**

Run: `python -m unittest runs._scripts.test_hotpotqa_unified_audit -v`
Expected: FAIL，因为 `hotpotqa_unified_audit` 尚未实现。

- [ ] **Step 3: 实现最小审计器**

审计器读取固定数据集和 `eval_hotpotqa_<method>/final_results.json`，输出 expected/completed documents、questions、duplicate ids、missing ids、invalid facts 和 errors。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m unittest runs._scripts.test_hotpotqa_unified_audit -v`
Expected: 所有覆盖审计测试 PASS。

### Task 2: 准备五方法配置与 ReAct 恢复清单

**Files:**
- Create: `runs/_configs/hotpotqa_fixed1000_baseline_common.yaml`
- Create: `runs/_configs/hotpotqa_fixed1000_abstract_only.yaml`
- Create: `runs/_configs/hotpotqa_fixed1000_bm25.yaml`
- Create: `runs/_configs/hotpotqa_fixed1000_dense.yaml`
- Create: `runs/_configs/hotpotqa_fixed1000_hybrid.yaml`
- Create: `runs/_configs/hotpotqa_fixed1000_raptor.yaml`
- Create: `runs/_datasets/hotpotqa/fixed1000/hotpotqa_react_missing2.json`
- Create: `runs/_datasets/hotpotqa/fixed1000/hotpotqa_react_missing2.yaml`

- [ ] **Step 1: 让公共配置继承现有冻结模型配置**

设置 paragraph/sentence corpus、top-k 10、supporting evidence top-k 4、HotpotQA short answer 输出，并保持 `rag_force_reprocess: true`。

- [ ] **Step 2: 为五个方法写最小派生配置**

方法仅覆盖 `index_type`、`retrieval_method` 与对应 vdb 目录，Dense/Hybrid 使用 `dense_paragraph_vdb`，RAPTOR 使用 `raptor_vdb`。

- [ ] **Step 3: 从固定 1000 题精确抽取 ReAct 缺失的两个 question id**

Expected ids: `5a81f10055429903bc27ba06`、`5a81fa59554299676cceb1b0`；工作目录仍指向 fixed1000 主工作目录。

- [ ] **Step 4: 解析全部 YAML 并验证方法 discriminator**

Run: `python -c "from Core.configs import load_config; ..."`
Expected: 六个配置均可加载，方法后缀与设计一致。

### Task 3: 构建并校验 Dense 与 RAPTOR 索引

**Files:**
- Output: `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/*/dense_paragraph_vdb/chroma.sqlite3`
- Output: `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/*/raptor_vdb/chroma.sqlite3`

- [ ] **Step 1: 两分片构建 Dense sentence index**

Run: `python main.py -c runs/_configs/hotpotqa_fixed1000_dense.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num <1|2> index --stage vdb`
Expected: 两个 worker exit 0。

- [ ] **Step 2: 严格检查 Dense collection 数量与非空状态**

Expected: 1000/1000 `dense_paragraph_vdb/chroma.sqlite3`，每个 collection 至少一个 sentence。

- [ ] **Step 3: 两分片构建 RAPTOR index**

Run: `python main.py -c runs/_configs/hotpotqa_fixed1000_raptor.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --nsplit 2 --num <1|2> index --stage vdb`
Expected: 两个 worker exit 0。

- [ ] **Step 4: 严格检查 RAPTOR collection 数量与非空状态**

Expected: 1000/1000 `raptor_vdb/chroma.sqlite3`，无空 collection。

### Task 4: 恢复 ReAct 并运行五个新方法

**Files:**
- Output: `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/*/eval_hotpotqa_<method>/`
- Output: `runs/_logs/hotpotqa_fixed1000_unified_*.log`

- [ ] **Step 1: 只对 missing2 数据配置运行 ReAct**

Run: `python main.py -c runs/_configs/hotpotqa_react_private.yaml -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_react_missing2.yaml --nsplit 1 --num 1 rag`
Expected: ReAct 从 998/1000 补到 1000/1000。

- [ ] **Step 2: 依次以两分片运行 Abstract-only、BM25、Dense、Hybrid、RAPTOR**

每个 worker 使用独立 stdout/stderr，完成后立即调用覆盖审计；已有完整方法由审计器跳过。

- [ ] **Step 3: 失败即停止并记录原子状态**

`full1000_unified_campaign_status.json` 记录 phase、current_method、completed_methods、failed_methods、active_pids、错误摘要与更新时间。

### Task 5: 统一官方评测、bootstrap 与主表

**Files:**
- Create: `runs/_scripts/bootstrap_hotpotqa.py`
- Create: `runs/_scripts/hotpotqa_unified_table.py`
- Test: `runs/_scripts/test_bootstrap_hotpotqa.py`
- Output: `runs/hotpotqa_evibridge/work_validation_fixed1000_seed42/0_results/hotpotqa_fixed1000_unified_11methods/`

- [ ] **Step 1: 先写 bootstrap 指标聚合失败测试**

固定三题逐题指标，断言 Answer EM、Answer F1、SP P、SP R、SP F1、Joint F1 的 point estimate 与 seed 42 区间字段结构；先确认缺少实现时测试失败。

- [ ] **Step 2: 实现六指标逐题 bootstrap**

对 Answer EM、Answer F1、SP Precision、SP Recall、SP F1、Joint F1 使用 10000 次有放回抽样，输出 point estimate、95% percentile CI、sample count 和 seed。SP EM 与 Joint EM 只保留在底层官方评测文件中。

- [ ] **Step 3: 对 11 方法运行 `Eval/evaluation.py`**

Run: `python Eval/evaluation.py -d runs/_datasets/hotpotqa/fixed1000/hotpotqa_distractor_validation_fixed1000_seed42.yaml --method <method> --max_workers 1`
Expected: 每方法 `missing_predictions=0`、`total_samples=1000`。

- [ ] **Step 4: 生成 Markdown/JSON 主表**

表中仅包含 11 方法的 Answer EM、Answer F1、SP P、SP R、SP F1、Joint F1，另附 bootstrap 95% CI、平均 token/题、平均时间/题和覆盖状态。

- [ ] **Step 5: 做最终一致性验证**

Run: `python -m unittest runs._scripts.test_hotpotqa_unified_audit runs._scripts.test_bootstrap_hotpotqa -v`
Expected: 全部 PASS；11/11 方法、每方法 1000/1000、零错误，主表 JSON 与各 score JSON 数值一致。
