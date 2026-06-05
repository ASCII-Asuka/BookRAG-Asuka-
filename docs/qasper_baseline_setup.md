# Qasper 基础 Baseline 运行说明

本说明用于 Qasper validation 100Q 的基础 baseline 对比。所有方法默认使用同一份数据集配置，例如：

```powershell
$DATA = "runs\qasper_evibridge\config\qasper_validation_100q.yaml"
```

系统配置模板位于 `config/`，运行前复制到 `runs/qasper_evibridge/config/` 并填好 `TODO`，不要把本机 API key 或服务地址提交到仓库。

## 使用硅基流动 Embedding

Dense paragraph 和 Hybrid BM25+Dense 可以直接使用硅基流动的 OpenAI-compatible embedding 接口。把相关配置中的 `embedding_config` 改成：

```yaml
embedding_config:
  type: text
  model_name: Qwen/Qwen3-Embedding-0.6B
  backend: openai
  max_length: 32768
  device: cpu
  api_base: https://api.siliconflow.cn/v1
  api_key: TODO
```

可选模型包括 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Embedding-4B`、`Qwen/Qwen3-Embedding-8B` 和 `BAAI/bge-m3`。第一轮 Qasper 100Q 建议先用 `Qwen/Qwen3-Embedding-0.6B`：成本低、速度快，足够作为 dense baseline；若它明显弱于 BM25，再补一次 `Qwen/Qwen3-Embedding-4B` 或 `BAAI/bge-m3` 作为强 dense baseline。

## Baseline 配置

| 方法 | 配置模板 | 依赖索引 |
|---|---|---|
| BM25 paragraph | `config/qasper_bm25_paragraph.yaml` | `bm25_vdb/bm25_index.pkl` |
| Dense paragraph | `config/qasper_dense_paragraph.yaml` | `dense_paragraph_vdb` |
| Hybrid BM25+Dense | `config/qasper_hybrid_rrf.yaml` | `bm25_vdb` + `dense_paragraph_vdb` |
| BM25+Reranker | `config/qasper_bm25_rerank.yaml` | `bm25_vdb` + reranker 服务 |
| Abstract-only | `config/qasper_abstract_only.yaml` | `tree.pkl` |

## 构建索引

BM25 paragraph：

```powershell
python main.py -c runs\qasper_evibridge\config\qasper_bm25_paragraph.yaml -d $DATA --nsplit 1 --num 1 index --stage vdb
```

Dense paragraph：

```powershell
python main.py -c runs\qasper_evibridge\config\qasper_dense_paragraph.yaml -d $DATA --nsplit 1 --num 1 index --stage vdb
```

Hybrid 复用上面两个索引，不需要单独构建。BM25+Reranker 复用 BM25 paragraph 索引。Abstract-only 只需要 Qasper 预处理阶段已经生成的 `tree.pkl`。

## 运行推理

```powershell
python main.py -c runs\qasper_evibridge\config\qasper_bm25_paragraph.yaml -d $DATA --nsplit 1 --num 1 rag
python main.py -c runs\qasper_evibridge\config\qasper_dense_paragraph.yaml -d $DATA --nsplit 1 --num 1 rag
python main.py -c runs\qasper_evibridge\config\qasper_hybrid_rrf.yaml -d $DATA --nsplit 1 --num 1 rag
python main.py -c runs\qasper_evibridge\config\qasper_bm25_rerank.yaml -d $DATA --nsplit 1 --num 1 rag
python main.py -c runs\qasper_evibridge\config\qasper_abstract_only.yaml -d $DATA --nsplit 1 --num 1 rag
```

Vanilla baseline 的结果目录后缀等于 `retrieval_method`，例如 `eval_qasper_bm25`、`eval_qasper_hybrid`、`eval_qasper_bm25_rerank`。

## 官方格式导出与评测

主口径使用原始生成回答 `--answer-source output`，paragraph-only evidence，默认导出 top4：

```powershell
python Scripts\eval\qasper_official.py --dataset-config $DATA --method bm25 --answer-source output --top-k-evidence 4 --output-dir runs\qasper_evibridge\work\0_results\qasper_official_bm25_output
python Scripts\eval\qasper_official.py --dataset-config $DATA --method vanilla --answer-source output --top-k-evidence 4 --output-dir runs\qasper_evibridge\work\0_results\qasper_official_dense_output
python Scripts\eval\qasper_official.py --dataset-config $DATA --method hybrid --answer-source output --top-k-evidence 4 --output-dir runs\qasper_evibridge\work\0_results\qasper_official_hybrid_output
python Scripts\eval\qasper_official.py --dataset-config $DATA --method bm25_rerank --answer-source output --top-k-evidence 4 --output-dir runs\qasper_evibridge\work\0_results\qasper_official_bm25_rerank_output
python Scripts\eval\qasper_official.py --dataset-config $DATA --method abstract_only --answer-source output --top-k-evidence 4 --output-dir runs\qasper_evibridge\work\0_results\qasper_official_abstract_only_output
```

## 汇总表格

```powershell
python Scripts\eval\qasper_baseline_table.py --root runs\qasper_evibridge\work\0_results --output runs\qasper_evibridge\work\0_results\qasper_baseline_table.md --json-output runs\qasper_evibridge\work\0_results\qasper_baseline_table.json
```

推荐论文主表至少包含：

| Method | Answer F1 | Evidence F1 | Missing |
|---|---:|---:|---:|
| Abstract-only |  |  |  |
| BM25 paragraph |  |  |  |
| Dense paragraph |  |  |  |
| Hybrid BM25+Dense |  |  |  |
| BM25+Reranker |  |  |  |
| EviBridge-RAG |  |  |  |
