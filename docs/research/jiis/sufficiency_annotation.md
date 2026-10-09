# CoSE-RAG 独立充分性标注规范 v0.1

版本日期：2026-10-09。状态：**标注准备草案，尚未完成规范预试、数据划分、人工标注或模型 pilot**。本文件对应 [JIIS 扩展计划 Task 1/2](../../superpowers/plans/2026-10-09-jiis-extension-plan.md)，不代表 Task 2 已完成。

## 1. 标注对象与判断边界

标注单位为一个问题和一份确切的候选上下文 `question + context`。上下文包含按实际顺序展示的证据块及其可见标题、表头、图注、来源定位；须保存块文本、顺序和 SHA256。仅凭块 ID、图连通性、query 词项重叠或系统自报分数不能判断充分性。答案是否碰巧正确也不能代替本项判断。

第一阶段进行盲审：标注者仅见问题、候选上下文和完成阅读所需的来源定位。隐藏方法名称、verifier 判定/分数、自动解析的 demand、生成答案、gold 答案/证据、变体类型与预期标签。块使用中性展示 ID；保留到真实源单位的离线映射。不得点击上下文之外的全文或检索外部材料补足事实。

标注者先独立列出问题需要的原子事实、比较维度或推理关系，再寻找上下文中的支持位置。允许有多个有效答案或多套替代证据；不要求匹配某一个 gold 证据集合。问题未要求的背景信息、格式要求或某一种模态，不自动成为必需证据。

第二阶段离线核验源语料可回答性及变体有效性。它与第一阶段标签分开存储，由核验者查看冻结的源语料、原始问题、多个 gold 标注和干预记录。线上 verifier、检索器和调参接口不可见这些资料。第一阶段标签若需更改，保留原标签、修订标签、原因和规范版本，不静默覆盖。

## 2. 三类 context 标签

| 标签 | 判断条件 | 必须记录 |
|---|---|---|
| `sufficient` | 仅凭可见 context，可给出符合问题范围的答案；所有关键原子需求及必要关系都有支持，可解释推导步骤。 | 至少一套可定位的支持位置；有推理时列出前提和结论。 |
| `insufficient` | 可以明确指出回答所需的关键事实、限定条件、关系或可读证据在 context 中缺失，且现有替代证据不能补足。 | 缺失需求及为何现有证据不足；不靠猜测填空。 |
| `undetermined` | 问题歧义、证据不可读、范围不明或无法消解的冲突，使标注者不能可靠确定充分或不足。 | 不确定来源及需要何种资料才能裁决。 |

不完整证据应标 `insufficient`；“我没有把握”须进一步检查是否有明确缺口或真正的判定障碍，不能把所有困难题都放入 `undetermined`。反之，不得强迫有歧义或不可读的状态取得二元标签。

这里的充分性关注有依据的回答能力。context 明确支持“否”“没有发生”等否定结论时，可以为 `sufficient`。仅因 context 没写答案而输出“无法回答”，不算已经证明充分；原任务评测中的拒答正确性另行测量。必要但不可读的图表应记为 `undetermined` 和 `unreadable_evidence`，不能凭标题推测图表内容。

## 3. 源语料可回答性与缺口分开

离线核验限定在本题冻结的合法语料范围：Qasper 的源论文全文，HotpotQA distractor 的题内候选语料。不得用外部网页回答 HotpotQA 后将它改称题内可回答。对清洁全文和 parser 输出有差异的情况，分别记录源语料状态与解析缺失。

| `source_answerability` | 含义 |
|---|---|
| `answerable` | 冻结源语料存在至少一套能回答问题的证据。 |
| `unanswerable` | 经离线核验，源语料不能支持问题要求的答案；记录问题前提、标注范围及核验理由。 |
| `undetermined` | 源材料不完整、冲突或问题歧义使可回答性无法核定。 |

源语料 `answerable`、context `insufficient` 表示当前上下文缺证据；源语料 `unanswerable` 是任务本身的无答案状态。二者的错误原因和改进方向分别统计。源语料无答案不自动把当前 context 改为 `sufficient`，也不把无答案任务从端到端 QA 主评测删除。gold 中的 unanswerable 标记是核验资料，不是人工充分性标签的自动赋值器。

## 4. 原子需求、hop 与支持位置

每条原子需求用一句话描述可核验的事实或关系，并标记其为必需或辅助。比较问题须列出双方、相同维度、必要限定和比较操作；桥接问题须列出中间实体如何与前后事实对应。无需人为固定 hop 数：直接证据可以消除原本预计的中间 hop，记录实际可成立的证明路径即可。

支持位置以展示块 ID 和精确文本跨度定位，文本偏移采用 Unicode 字符的 `[start, end)` 区间。保留可选的原文引用片段；表格使用行/列/表头定位，图或图注使用可见区域/图注定位。PDF 页码、树节点 ID 或 title-sentence ID 是来源字段，不能单独当作蕴含说明。

对每条必需需求，记录 `supported`、`missing` 或 `unclear`，给出支持块/跨度或缺口理由。有推理时，单列前提需求 ID、推导出的关系及允许的操作。图边 `context/semantic/hierarchy` 可以用于导航；其存在不自动证明某条原子关系。

允许基本语言理解、同文明确的代词/实体对应、单位换算、简单算术及题目要求的比较；计算过程要可复查。不允许用个人记忆、专业常识、世界知识或互联网补出缺失实体事实、实验结果、日期及因果关系。若推理确实依赖外部知识，记录该依赖并视其是否关键判为 `insufficient`；推理规则本身存在合理争议时可标 `undetermined`。

替代证据只要实际支持所需事实即可采用，包括非 gold 段落及不同有效推导路径。删除某一 gold 块后，若剩余 context 有替代证据，标签仍可能为 `sufficient`。语义相关、相同实体或词面相近不足以证明支持；证据表达的是相关性时，不得自行升级为因果结论。

## 5. 数据划分与最多五个变体

开发 pilot 的目标为 120 个源问题，Qasper/HotpotQA 各 60。来源须是经 Task 1 审计的独立开发数据；按题型、证据跨度、系统接受/拒绝及回答对错分层，抽样工作由数据管理员完成，分层字段不展示给标注者。系统判定和答案仅用于离线抽样，不参与人工贴标签。保存各层总体数量、抽样数量、纳入概率及不适用原因；分层诊断结果与加权总体估计分开报告，未知纳入概率时不声称总体代表性。

**现有 BigData 的 Qasper validation 1005 题和 HotpotQA validation 固定 1000 题已被研究者查看，不能重命名为独立开发集、新 holdout 或未看确认集。** 可以另做历史结果探索，必须标 `historical_exploration` 并与本 pilot、确认结果分表；若独立开发数据尚未确定，保持 pilot 待执行，不用历史 validation 凑齐目标数量。

每个源问题最多五个 context 状态，120 源问题最多 600 状态；不适用的变体留缺项和原因，不为凑数造状态：

| 状态槽位 | 构造方式与核验要求 |
|---|---|
| 原始状态 | 冻结选定检索阶段的实际 context；不要预设原始一定充分。 |
| 关键原子证据替换 | 用长度相近的干扰块替换候选关键证据，核验是否仍有替代支持。 |
| 缺中间事实 | 移除或替换一个必要桥接事实；直接证据已足够的题记录不适用。 |
| 词面相近干扰 | 加入或替换高词项重叠、语义不支持的证据；不得自动贴不足标签。 |
| 冲突或无答案测试 | 按预登记规则选一种适用的测试，记录选择理由与真实/合成来源。 |

尽量控制 context 块数、token 数和展示格式，记录原始/变体的实际差值及 token 计数器版本。替换材料来源合法，干预记录可追溯；控制长度不会保证语义干预成功，仍须两人盲审和离线核验。反事实状态只有在人工确认被干预的需求、剩余替代证据和实际语义标签后才进入对应分析；标签与构造预期相反也保留。

图边干预另属结构鲁棒性实验：问题、文本、顺序、可见来源信息完全相同时，只删边、重连或改 typed label，沿用原语义充分性标签并记录原状态 ID。若边干预改变最终检索 context，则新 context 需要重新盲审，不能沿用标签。不得将“图断路”自动改成 `insufficient`。

真实源文档中的冲突与人工合成冲突分别记录、分别统计。无法消解的冲突通常为 `undetermined`；若问题要求识别冲突、或 context 明确给出版本/范围而能解决冲突，可为 `sufficient`。合成内容保留修改前后 hash 和定位，标记为 `synthetic_conflict`，不混入自然分布的主 QA 结果。

同一源问题的原始状态和所有变体始终属于同一 split。Qasper 同论文的问题按文档组隔离；HotpotQA 按原问题组隔离，并审计跨 split 的共享来源文档，不能将同一源文档的事实通过另一题泄漏到确认评测。必要时按共享文档形成分组并记录最终可用数量。开发、calibration 与确认评测之间保存源问题 ID、文档 ID 和文本重复交集审计。

确认阶段另从事先冻结、研究者未查看候选结果的确认集选 120 个源问题及同规则变体，数量目标同样为两数据集各 60。使用冻结的 v1 规范、抽样规则及模型/阈值，只做最终评测。参数、prompt、阈值和候选策略选择仅使用独立开发数据；校准映射只用专设 calibration 分区，不使用确认集。规范或协议必须调整时建立新版本、记录原因和已看数据，不覆盖旧版本后继续称原确认评测。

## 6. 两人独立标注、仲裁与规范冻结

1. 从独立开发数据选 10–20 个源问题及适用变体作规范预试，覆盖比较/桥接、明确缺口、替代证据、冲突与无答案边界。标注者 A/B 独立提交，互不可见标签及理由；状态顺序随机化，同题变体尽量不连续展示。
2. 在预试后汇总分歧与漏项，修订定义、来源显示和定位方式；保持修订记录。预试用于规范训练，其题目仍属开发，不进入确认集。需用 v1 分析时按冻结规范重新独立标注，不用讨论后的共识冒充初次独立标注。
3. 冻结 v1 规范、模板、展示方式、采样 manifest 和版本 hash，然后进行完整开发 pilot 的 A/B 双标。两人即使二元标签一致但关键支持/缺口相矛盾，也进入核验；同标签不能替代证据定位质量。
4. 仲裁者先看 question/context，再查看两份独立标注；记录最终标签、支持/缺口及裁决理由。必要时查源语料做第二阶段核验。争议无法可靠解决时最终为 `undetermined`；保留两份原标签，不用空缺顶替争议。
5. 报告仲裁前的三类标签分布、三类混淆表、原始一致率和 Cohen's κ（无法定义时说明原因）；分源题/干预类型呈现分歧。仲裁后的最终标签才是诊断参考，不能把仲裁一致率报告成人类独立一致性。

模型辅助评测若后续引入，评测 judge 的输入、prompt、版本及调用记录独立于在线 verifier，禁止读取在线 verdict、生成答案或方法身份来决定充分性。冻结人工参考标签，并人工审核模型 judge 的错误；使用相同模型或模型家族时披露相关偏差，不能只因换 prompt 就声称模型误差已独立。人工参考是本 pilot 的主要依据，本规范没有调用或校准任何 judge。

## 7. 可复制的空 JSON 模板

下面是单状态的记录壳，所有人工标签与结果保持 `null`。管理员在准备真实样本时填 ID/hash/context；盲审导出只含 `blind_view`，其余管理员、核验和模型字段不可展示给标注者。`context_sha256` 对精确展示载荷计算，保证顺序/标题等可见信息也纳入校验。实际文本、标注和 manifest 存于 `runs/jiis/` 或另获授权的数据位置，不把真实文本填入本模板提交仓库。

```json
{
  "schema_version": "0.1",
  "protocol_version": "0.1",
  "state_id": null,
  "admin_only": {
    "dataset": null,
    "dataset_version": null,
    "split_role": null,
    "source_question_id": null,
    "source_document_ids": [],
    "source_group_id": null,
    "source_corpus_sha256": null,
    "sampling_stratum": null,
    "inclusion_probability": null,
    "variant_type": null,
    "parent_state_id": null,
    "intervention_kind": null,
    "intervention_record_sha256": null,
    "conflict_origin": null,
    "token_counter_version": null,
    "context_tokens": null,
    "context_blocks": null,
    "delta_tokens_from_original": null,
    "delta_blocks_from_original": null
  },
  "blind_view": {
    "question": null,
    "question_sha256": null,
    "context_sha256": null,
    "context": [
      {
        "display_block_id": null,
        "visible_source_locator": null,
        "text": null,
        "text_sha256": null,
        "visible_nontext_reference": null
      }
    ]
  },
  "independent_annotations": [
    {
      "annotator_id": null,
      "annotation_version": null,
      "context_label": null,
      "reason_codes": [],
      "rationale": null,
      "atomic_requirements": [
        {
          "requirement_id": null,
          "description": null,
          "necessity_label": null,
          "support_status": null,
          "support_spans": [
            {
              "display_block_id": null,
              "char_start": null,
              "char_end": null,
              "quote": null,
              "table_or_figure_locator": null
            }
          ],
          "gap_reason": null
        }
      ],
      "inference_steps": [
        {
          "premise_requirement_ids": [],
          "inferred_relation": null,
          "operation": null,
          "external_knowledge_required": null
        }
      ],
      "alternative_support_sets": [],
      "uncertainty_reason": null
    }
  ],
  "adjudication": {
    "adjudicator_id": null,
    "final_context_label": null,
    "rationale": null,
    "retained_support_requirement_ids": [],
    "unresolved_reason": null,
    "label_revision_history": []
  },
  "offline_source_check": {
    "reviewer_id": null,
    "source_answerability": null,
    "evidence_source_locators": [],
    "gold_sets_checked": null,
    "source_parser_gap": null,
    "intervention_validity_label": null,
    "intervened_requirement_ids": [],
    "alternative_evidence_checked": null,
    "rationale": null
  },
  "evaluation_only": {
    "system_version": null,
    "system_decision": null,
    "decision_source": null,
    "failure_type": null
  }
}
```

真实记录中 `independent_annotations` 必须包含 A/B 两条分别完成的记录；不得复制第一条冒充第二名标注者。必需字段留空的记录归为未完成，不等于 `undetermined`。`context_label` / `final_context_label` 的允许值仅为 `sufficient`、`insufficient`、`undetermined`；`source_answerability` 的允许值见第3节。

建议缺口/不确定原因码包括 `missing_atomic_fact`、`missing_intermediate_relation`、`missing_scope_constraint`、`external_knowledge_dependency`、`unreadable_evidence`、`ambiguous_question`、`unresolved_conflict`、`source_unanswerable`。可多选，但附自然语言理由；`source_unanswerable` 只在离线核验后赋值。变体有效性可填 `confirmed`、`not_effective`、`undetermined`，它与充分性标签分开。

## 8. 判定指标、分母和报告要求

以下以仲裁后确定标签为参考，系统决策拆为 `accept`、`reject`、`abstain`、`error`；不得将空预测、异常或未完成标注静默转成某个类。`rule_only` 与其他候选 verifier 分别计算；只报告真实执行过且记录可审计的判定。

- 误接受率：`#(人工 insufficient 且系统 accept) / #(人工 insufficient)`。
- 误拒绝率：`#(人工 sufficient 且系统 reject) / #(人工 sufficient)`；系统 `abstain` 和 `error` 在该分母内另报比例。若另报“充分状态未被接受率”，其分子明确为 `reject + abstain + error`，不得混称误拒绝率。
- 接受错误占比：`#(人工 insufficient 且系统 accept) / #(确定人工标签且系统 accept)`，与误接受率同时注明不同分母。
- 确定标签上的准确率：`#(sufficient/accept 或 insufficient/reject) / #(人工 sufficient 或 insufficient)`；分母内 abstain/error 计未正确。条件于成功二元决策的指标可另报，同时给出决策覆盖率。
- `undetermined` 占比：`#(人工 undetermined) / #(完成仲裁的全部状态)`；在这些状态上另报系统 accept/reject/abstain/error 分布，不计入二元正确率后宣称全量准确率。

每项报告分子、分母和区间；分母为零时写不可计算，不填 0。报告总源题数、总状态数、缺失/未完成标注、无效干预、模型异常以及各类标签数量。`not_effective` 变体保留实际人工标签和构造失败率，不能仅因未变不足而删去；不可用状态的排除规则须提前冻结并报告数量。

按原始/干预状态、两数据集、源可回答性及真实/合成冲突分层。每个源题的多个状态相关，按源题组估计不确定性；Qasper 还需以文档为聚类单位。统计重采样次数不等于模型重复运行次数。分层抽样诊断不能不加权冒充自然数据的误接受风险；若做加权估计，披露权重与实际覆盖。

充分性判定与端到端 Answer/Evidence/Joint 指标分开；端到端主评测保留全部冻结题目，包括人工 `undetermined`、无答案及系统失败。没有实际生成答案的状态只报告充分性诊断，不能补造 QA 分数。每个结论注明属于开发 pilot、历史探索或确认评测。

## 9. 启动与完成门槛

- [ ] Task 1 的独立开发/calibration/确认来源、分组和交集审计已冻结；当前历史 validation 没有被改称独立集合。
- [ ] 10–20 源题规范预试已由两人独立完成，分歧/修订可追溯；v1 规范及展示 hash 已冻结。
- [ ] 120 个开发源题的抽样 manifest 已保存，最多五状态、变体不适用原因及长度控制可核验。
- [ ] 盲审载荷不含方法身份、gold、答案、verdict 或干预预期；两人原标签与仲裁均保留。
- [ ] 源可回答性、缺失 context、替代证据、反事实有效性和真实/合成冲突分开核验。
- [ ] 确定/不确定状态及系统异常均有明确分母；开发选择和确认评测有独立记录。

只有上述流程执行并产生真实可审计标注、判定和报告后，才能写“独立充分性 pilot 已完成”。目前本文件提供的是可执行规范与空模板。
