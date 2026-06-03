# Qasper 数据集准备与 EviBridge 实验流程

本文档用于第一次准备公开数据集实验。当前阶段只跑 Qasper validation split，路线为：

```text
Qasper full_text -> pseudo DocumentTree -> EviBridgeIndex -> EviBridge-RAG -> Qasper/Evidence metrics
```

第一轮不使用 PDF，不调用 MinerU，也不下载论文原文 PDF。Qasper 官方数据页是 [allenai/qasper](https://huggingface.co/datasets/allenai/qasper)。该数据集包含论文正文、问题、答案和 supporting evidence，适合先验证 EviBridge-RAG 的 evidence recall、path connectivity 和 answer quality。

## 1. 本地目录

公开数据放在仓库外，默认：

```powershell
$DATA_ROOT = "D:\Else\Study\datasets\public"
$QASPER_ROOT = "$DATA_ROOT\qasper"
```

推荐目录结构：

```text
D:\Else\Study\datasets\public\
  qasper\
    hf_cache\
    raw\
    processed\
  hotpotqa\
  2wikimultihopqa\
  mmlongbench_doc\
```

实验产物放在仓库内已忽略的 `runs/`：

```text
runs\qasper_evibridge\
  processed\
  config\
  work\
```

## 2. 安装下载依赖

仓库没有固定依赖锁文件。下载 Qasper 需要 Hugging Face datasets：

```powershell
pip install datasets huggingface_hub pyarrow
```

如果只是跑已有单元测试，不需要安装这些包；下载脚本会懒加载 `datasets`。

## 3. 下载 Qasper

```powershell
python Scripts\preprocess\download_qasper.py --output-root D:\Else\Study\datasets\public\qasper
```

脚本会：

- 调用 `load_dataset("allenai/qasper")`。
- 保存 `train`、`validation`、`test` 到 `D:\Else\Study\datasets\public\qasper\raw\qasper_<split>.json`。
- 写出 `D:\Else\Study\datasets\public\qasper\raw\manifest.json`。
- 打印每个 split 的 paper 数、question 数，以及 answer/evidence 字段是否存在。

第一轮实验只使用：

```text
D:\Else\Study\datasets\public\qasper\raw\qasper_validation.json
```

## 4. 生成 3 篇 paper 小样本

先用 3 篇 paper 检查字段与 evidence 对齐：

```powershell
python Scripts\preprocess\qasper_evibridge.py `
  --raw D:\Else\Study\datasets\public\qasper\raw\qasper_validation.json `
  --output runs\qasper_evibridge\processed\qasper_validation_sample_3docs.json `
  --sample-docs 3 `
  --tree-dir runs\qasper_evibridge\work `
  --dataset-config-output runs\qasper_evibridge\config\qasper_validation_sample.yaml `
  --system-config-output runs\qasper_evibridge\config\evibridge_qasper.yaml
```

该命令会生成：

```text
runs\qasper_evibridge\processed\qasper_validation_sample_3docs.json
runs\qasper_evibridge\config\qasper_validation_sample.yaml
runs\qasper_evibridge\config\evibridge_qasper.yaml
runs\qasper_evibridge\work\<paper_id>\tree.pkl
runs\qasper_evibridge\work\<paper_id>\tree.json
```

统一 JSON 中每条样本至少包含：

```json
{
  "question": "...",
  "answer": [],
  "doc_uuid": "paper_id",
  "doc_path": "qasper://paper_id",
  "qasper_question_id": "...",
  "evidence_block_ids": []
}
```

## 5. 构建 EviBridge 索引

```powershell
python main.py `
  -c runs\qasper_evibridge\config\evibridge_qasper.yaml `
  -d runs\qasper_evibridge\config\qasper_validation_sample.yaml `
  --nsplit 1 --num 1 `
  index --stage evibridge
```

这里会读取每篇 paper 已生成的 `tree.pkl`，再生成：

```text
runs\qasper_evibridge\work\<paper_id>\evibridge_index.json
runs\qasper_evibridge\work\<paper_id>\evibridge_bm25.pkl
```

第一轮建议保持 `enable_vector_recall: false`，先验证 BM25 + EviBridge 图扩展链路。

## 6. 运行 RAG

运行前确认 `runs\qasper_evibridge\config\evibridge_qasper.yaml` 里的 LLM 配置已经改成你本机可用的服务。

```powershell
python main.py `
  -c runs\qasper_evibridge\config\evibridge_qasper.yaml `
  -d runs\qasper_evibridge\config\qasper_validation_sample.yaml `
  --nsplit 1 --num 1 `
  rag
```

每个 query 会产生：

```text
result.json
retrieval_res.json
evidence_chain.json
```

重点看 `retrieval_res.json` 里的 seed、typed PPR 分数和 selector 分数，以及 `evidence_chain.json` 中的 ordered evidence、连接边和 verifier 轮次。

## 7. 评测

```powershell
python Eval\evaluation.py `
  -d runs\qasper_evibridge\config\qasper_validation_sample.yaml `
  --method evibridge `
  --max_workers 1
```

汇总结果位于：

```text
runs\qasper_evibridge\work\0_results\
```

Qasper 评测会输出 Answer F1/Accuracy，并读取 EviBridge 额外指标：

- Evidence F1
- Evidence Recall
- Path Connectivity
- Bridge Coverage
- Noise Ratio
- Verifier Iterations
- Token Cost / Time Cost

## 8. 人工检查 5 个 query

第一轮不要急着跑完整 validation。先人工打开 5 个 query 的输出，检查：

- `retrieval_res.json` 的 seed 是否包含 gold evidence 段落或邻近段落。
- `typed_ppr_score_parts` 是否体现 `context`、`semantic`、`hierarchy` 差异。
- `evidence_chain.json` 是否记录连接边和 verifier 补检索动作。
- `evidence_recall` 是否合理，尤其是有 gold evidence id 时是否按 id 对齐。
- `noise_ratio` 是否明显过高。

## 9. 后续扩展

3 篇 paper 小样本稳定后，再跑完整 validation：

```powershell
python Scripts\preprocess\qasper_evibridge.py `
  --raw D:\Else\Study\datasets\public\qasper\raw\qasper_validation.json `
  --output runs\qasper_evibridge\processed\qasper_validation_unified.json `
  --tree-dir runs\qasper_evibridge\work `
  --dataset-config-output runs\qasper_evibridge\config\qasper_validation.yaml `
  --system-config-output runs\qasper_evibridge\config\evibridge_qasper.yaml
```

Qasper validation 跑稳后，再接 HotpotQA、2WikiMultiHopQA 或 MMLongBench-Doc。相关入口：

- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [MMLongBench](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench)
- [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc)
