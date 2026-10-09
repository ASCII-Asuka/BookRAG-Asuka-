# CoSE-RAG 可审计执行验证记录

日期：2026-10-09。范围是[可审计执行实施计划](../../superpowers/plans/2026-10-09-cose-auditable-execution.md)的工程准备；没有真实模型调用、论文质量复测、commit/push或旧实验覆写。基础HEAD为 `4c0d4d0fb5df49543d2c95345ba109f9e7741834`，工作区仍在dev且有未提交改动。

## 实现与独立review

`config/evibridge_auditable.yaml` 使用rule_only/strict，suffix为 `evibridge_support_pruned_rule_only_strict_context`。rule_only保留规则接受和数值指标，LLM诊断单独记录；strict的提交support属于保留成功生成上下文，外部补充存posthoc，再生成失败保留draft。客户端尝试保存prompt/response hash与块ID/文本hash；各轮rule/LLM/final独立记录。默认legacy的判定和支持行为保持兼容。

每次run_rag创建独立manifest/events/summary。新result链接来源和输入指纹；有效缓存原字节不变，明确输入错配记录失败并停止，缺历史身份为partial/unknown。配置指定VDB和常见索引文件参与hash，索引覆盖保守partial/unknown，配置及位置脱敏，异常只在新provenance记录类型。

独立spec review和最终质量review均通过。review发现并经失败测试、修复、复核的问题：

- 缓存结果为非对象、读取失败、输入错配或目录创建失败时，来源事件/失败summary缺失或关联错误。
- strict初始chain缺metadata，使Hotpot评测可能从正式support回退到全部selected；现在完整保留来源映射。
- 固定目录扫描遗漏配置指定的自定义/外部VDB；现在加入指纹，并将覆盖保守标partial。
- 提前Path归一化与实际loader的原字符串路径规则不一致；现在沿用loader解析，`./`和Windows混合斜杠测试确认只跟实际索引目录变化。

## 实际验证

| 检查 | 结果 |
|---|---|
| 完整`test_evibridge*.py` | 121项通过，包含14项verifier与12项生成上下文新测试 |
| `tests.test_run_provenance` | 28项通过；与wiring联跑42项通过 |
| `test_qasper*.py` | 47项通过 |
| Hotpot评测/预处理/下载/分析及可用采样测试 | 19项通过 |
| Hotpot `test_prepare_fixed_sample_writes_hashed_manifest_and_trees` | 本地缺PyArrow，报ModuleNotFoundError；未安装依赖或改无关测试 |
| dataset manifest、paired bootstrap、closed-loop analysis/paper | 18项通过 |
| 修改模块py_compile、git diff --check | 通过 |
| 既有实验/审计/PDF指纹 | 7301项一致、0差异（7296主实验产物＋4审计文件＋投稿PDF） |

离线集成smoke在合成题和fake SDK上使用真实EviBridge generation与run_rag；确认三种结果文件中的generation轨迹一致、支持集包含性、posthoc分列、有效缓存0模型调用/旧字节不变、错配缓存停止且保留文件、初始生成失败留下当前安全轨迹。最后一次smoke的Core/main源码聚合SHA256为 `9681e438effe7cbe639e2f045f807b915f1b1624a208dea05fc6caf53fc4adc6`；这不是整个环境或仓库的锁文件。

临时产物均在忽略目录：集成smoke及不可覆写运行记录位于 `runs/jiis_audit_temp/auditable_smoke/e58f52a60d15413b949e497747b2d566/`，入口脚本为 `runs/jiis_audit_temp/root/auditable_smoke.py`；历史保存核查脚本与汇总为该root目录下 `verify_artifact_preservation.py` 和 `artifact_preservation_summary.json`。不把mock质量数字、token计数或成功响应用于论文。

## 仍待完成

这一步没有验证语义充分性、答案蕴含、校准或新策略收益，也不能认证服务端实际收到prompt或其内部截断。历史缺请求仍为unknown，会议完整运行commit/模型版本不能由新manifest补造。实际索引覆盖与模型版本分别报告，不将partial包装为完整可复现锁。

下一步依[标注规范v0.1](sufficiency_annotation.md)开展10–20个独立开发源题的两人标注预试，再冻结规范和分组协议；[开发候选清单](development_data_status.md)已核对Qasper train与历史主实验题/文档ID无交集。Hotpot现存简化train缺ID/context/supporting_facts，须先准备完整语料。120开发源题、反事实状态、真实人工标签、模型pilot和确认评测尚未执行。
