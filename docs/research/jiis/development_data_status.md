# 独立开发数据准备状态

日期：2026-10-09。用户已确认 Qasper train、HotpotQA train 均未用于调参、配置选择或查看候选方法结果。本记录只核对本地候选文件；开发、calibration、确认评测的抽样和 split 尚未冻结，尚无充分性标签或模型 pilot。

| 候选文件 | 本地核查 | 后续用途与限制 |
|---|---|---|
| `runs/_datasets/qasper/official/qasper-train-v0.3.json` | 888篇文档、2593题；题ID无缺失/重复；887篇有非空full_text；与历史1005题及全部本地official dev的题ID、文档ID交集均为0 | 可准备按文档分组的独立开发协议；空全文文档须记录处置。ID交集为0不代替正文重复、来源完整性或全部历史用途审计。 |
| `runs/third_party/ReAct/data/hotpot_train_v1.1_simplified.json` | 90447行，只有question/answer/type；无题ID、context、supporting_facts | 无法用于封闭语料证据检索/充分性评测。需准备带原始ID、题内context和支持事实的完整train，不能拿简化问题列表或历史1000题代替。 |

本地 Qasper train SHA256：`9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4`。Hotpot简化train SHA256：`39cceef6f332af585cc7eb4ca302982b7c9b26a589a392770b68be41c9c3c37e`。只读统计脚本和完整候选清单保存于忽略目录 `runs/jiis_audit_temp/root/development_candidate_inventory.{py,json}`；清单不包含问题/答案正文。

下一阶段先依[标注规范](sufficiency_annotation.md)从独立开发来源选10–20个源题预试，再扩展到Qasper/Hotpot各60个开发源题。实际检索上下文、两人标注和仲裁仍待执行；不会用gold证据人为构造“原始检索结果”。抽样前需记录题型/跨度、来源文档组、排除原因及确认数据的隔离协议。Qasper可先做规范准备，跨数据集pilot须在Hotpot完整train就绪后进行。

现有 `Scripts/preprocess/download_hotpotqa.py` 支持train/validation下载与JSON导出，依赖datasets/pyarrow；当前虚拟环境缺PyArrow。数据准备应使用独立本地输出目录，保留现有validation与会议产物，下载后核对原题ID、语料范围、重复/共享源文档及文件hash，再按[JIIS计划Task1/2](../../superpowers/plans/2026-10-09-jiis-extension-plan.md)冻结分组清单。本步骤未下载数据、安装依赖或调用真实模型。
