"""Render audited closed-loop results into the latest CoSE-RAG TeX source."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


ANCHOR = r"\subsection{参数敏感性分析}"
ANALYSIS_START = r"\subsection{闭环检索行为分析}"
TRAJECTORY_START = r"\subsection{检索轨迹案例分析}"


def _latex(value: Any) -> str:
    text = " ".join(str(value or "").split())
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _short(value: Any, limit: int = 150) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _rate(row: Mapping[str, Any], prefix: str) -> str:
    count = int(float(row[f"{prefix}_count"]))
    total = int(float(row["total_questions"]))
    return f"{100 * float(row[f'{prefix}_rate']):.1f}\\% ({count}/{total})"


def _expanded_rate(row: Mapping[str, Any]) -> str:
    count = int(float(row["expanded_accept_count"]))
    denominator = int(float(row["expanded_accept_denominator"]))
    return f"{100 * float(row['expanded_accept_rate']):.1f}\\% ({count}/{denominator})"


def _gold_text(case: Mapping[str, Any]) -> str:
    value = case.get("gold_answer")
    if isinstance(value, list):
        return "；".join(_short(item, 90) for item in value)
    return _short(value, 180)


def _fact_key(block: Mapping[str, Any]) -> tuple[str, int] | None:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    title = metadata.get("hotpot_title") or block.get("hotpot_title")
    sent_id = metadata.get("hotpot_sent_id", block.get("hotpot_sent_id"))
    if title is None or sent_id is None:
        return None
    return str(title), int(sent_id)


def _representative_round_blocks(case: Mapping[str, Any], limit: int = 2) -> list[Mapping[str, Any]]:
    blocks = list((case.get("round_1") or {}).get("selected_atomic_blocks") or [])
    if case.get("dataset") == "qasper":
        gold = {item for group in case.get("gold_evidence") or [] for item in group}
        matching = [item for item in blocks if item.get("block_id") in gold]
    else:
        gold = {tuple(item) for item in case.get("gold_evidence") or []}
        matching = [item for item in blocks if _fact_key(item) in gold]
        matching.sort(key=lambda item: (-len(str(item.get("text") or "")), str(item.get("block_id") or "")))
    selected = matching[:limit]
    for item in blocks:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _evidence_text(
    items: Sequence[Mapping[str, Any]], limit: int = 2, text_limit: int = 150
) -> str:
    parts = []
    for item in list(items)[:limit]:
        block_id = _latex(item.get("block_id"))
        section = _latex(_short(item.get("section_id") or item.get("hotpot_title") or "", 55))
        text = _latex(_short(item.get("text") or item.get("content") or "", text_limit))
        parts.append(rf"\texttt{{[{block_id}]}}（{section}）{text}")
    return r"\newline ".join(parts) if parts else "无可展示原子证据"


def _baseline_cell(case: Mapping[str, Any]) -> str:
    baseline = case.get("bm25_reranker") or {}
    answer = _latex(_short(baseline.get("answer"), 100))
    evidence = _evidence_text(baseline.get("supporting_evidence") or [])
    f1 = 100 * float(baseline.get("evidence_f1") or 0.0)
    return rf"{evidence}\newline 答案：{answer}；证据集 F1={f1:.2f}\%。"


def _round_one_cell(case: Mapping[str, Any]) -> str:
    round_one = case.get("round_1") or {}
    evidence = _evidence_text(_representative_round_blocks(case))
    f1 = 100 * float(round_one.get("evidence_f1") or 0.0)
    return rf"{evidence}\newline 首轮证据集 F1={f1:.2f}\%。"


def _diagnosis_cell(case: Mapping[str, Any]) -> str:
    diagnosis = case.get("diagnosis") or {}
    missing = _latex(", ".join(str(item) for item in diagnosis.get("missing_demand") or []) or "none")
    bridges = _latex(", ".join(str(item) for item in diagnosis.get("suggested_bridge_types") or []) or "none")
    action = _latex(diagnosis.get("controlled_action") or "none")
    return rf"$\mathcal{{M}}$：{missing}；$\mathcal{{L}}$：{bridges}；$\xi_{{\mathrm{{next}}}}$：{action}。"


def _round_two_cell(case: Mapping[str, Any]) -> str:
    added_evidence = list(case.get("round_2_added_evidence") or [])
    if case.get("dataset") == "qasper":
        cited_gold = set(case.get("round_2_added_gold_support_ids") or [])
        added_evidence.sort(key=lambda item: item.get("block_id") not in cited_gold)
    text_limit = 200 if case.get("dataset") == "qasper" else 150
    evidence = _evidence_text(added_evidence, limit=3, text_limit=text_limit)
    relations = []
    for edge in case.get("round_2_selected_bridges") or []:
        relation = (str(edge.get("bridge_type") or ""), str(edge.get("relation_type") or ""))
        if relation not in relations:
            relations.append(relation)
    relation_text = "、".join(f"{_latex(a)}/{_latex(b)}" for a, b in relations[:4]) or "日志未记录直接边"
    return rf"{evidence}\newline 相关桥：{relation_text}。"


def _final_cell(case: Mapping[str, Any]) -> str:
    final = case.get("final_output") or {}
    answer = _latex(_short(final.get("answer"), 120))
    ids = ", ".join(str(item) for item in final.get("supporting_block_ids") or [])
    first = 100 * float((case.get("round_1") or {}).get("evidence_f1") or 0.0)
    final_f1 = 100 * float(final.get("evidence_f1") or 0.0)
    answer_f1 = final.get("answer_f1")
    answer_metric = "" if answer_f1 is None else rf"；答案 F1={100 * float(answer_f1):.2f}\%"
    return rf"答案：{answer}；支持块：\texttt{{[{_latex(ids)}]}}{answer_metric}；证据集 F1：{first:.2f}\%$\rightarrow${final_f1:.2f}\%。"


def render_addition(
    summaries: Mapping[str, Mapping[str, Any]],
    cases_payload: Mapping[str, Any],
) -> str:
    q = summaries["qasper"]
    h = summaries["hotpotqa"]
    cases = cases_payload["cases"]
    q_case = cases["qasper"]
    h_case = cases["hotpotqa"]

    q_no_candidate = int(float(q["unaccepted_without_second_round_count"]))
    h_no_candidate = int(float(h["unaccepted_without_second_round_count"]))
    q_total = int(float(q["total_questions"]))
    h_total = int(float(h["total_questions"]))
    q_no_rate = 100 * q_no_candidate / q_total
    h_no_rate = 100 * h_no_candidate / h_total

    return rf"""\subsection{{闭环检索行为分析}}

为考察充分性闭环在完整评测集上的实际行为，本文从已完成的主实验日志中逐题重建检索轮次、验证器输出和生成上下文。表~\ref{{tab:closed_loop_behavior}} 汇总了结果。这里的“触发”指系统确实生成了新候选并执行第二轮；首轮判为不充分但没有新候选的问题按日志中的 \texttt{{no\_new\_candidates}} 提前终止，不计入第二轮子集。第一轮答案未单独保存，因此不推测 Answer F1 或 Joint F1 的轮间变化，仅对两轮已经固定的原子证据集进行离线评测。

\begin{{table*}}[t]
\centering
\small
\caption{{\method\ 在完整评测集上的闭环检索行为。比例同时给出计数；证据指标均为触发第二轮问题的逐题宏平均（\%）。}}
\label{{tab:closed_loop_behavior}}
\setlength{{\tabcolsep}}{{9pt}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{统计量}} & \textbf{{Qasper}} & \textbf{{HotpotQA}} \\
\midrule
首轮直接接受率 & {_rate(q, 'first_round_accept')} & {_rate(h, 'first_round_accept')} \\
第二轮实际触发率 & {_rate(q, 'second_round_trigger')} & {_rate(h, 'second_round_trigger')} \\
未接受且无新候选 & {q_no_rate:.1f}\% ({q_no_candidate}/{q_total}) & {h_no_rate:.1f}\% ({h_no_candidate}/{h_total}) \\
扩展后接受率 & {_expanded_rate(q)} & {_expanded_rate(h)} \\
最终仍不充分比例 & {_rate(q, 'final_insufficient')} & {_rate(h, 'final_insufficient')} \\
平均检索轮数 & {float(q['mean_round_count']):.3f} & {float(h['mean_round_count']):.3f} \\
平均新增原子证据块数 & {float(q['triggered_mean_added_atomic_blocks']):.2f} & {float(h['triggered_mean_added_atomic_blocks']):.2f} \\
首轮证据集 F1 & {100 * float(q['triggered_first_evidence_metric']):.2f} & {100 * float(h['triggered_first_evidence_metric']):.2f} \\
最终证据集 F1 & {100 * float(q['triggered_final_evidence_metric']):.2f} & {100 * float(h['triggered_final_evidence_metric']):.2f} \\
绝对增益 & +{100 * float(q['triggered_mean_evidence_gain']):.2f} & +{100 * float(h['triggered_mean_evidence_gain']):.2f} \\
\bottomrule
\end{{tabular}}
\vspace{{2pt}}

\parbox{{0.96\textwidth}}{{\footnotesize 注：扩展后接受率、平均新增块数及轮间证据指标只在实际执行第二轮的问题上计算；Qasper 与 HotpotQA 分别报告 Evidence F1 和 Supporting Fact F1。标准证据仅在推理完成后用于离线评测和案例筛选，不参与充分性判断、候选扩展或受控动作生成。}}
\end{{table*}}

Qasper 有 {int(float(q['first_round_accept_count']))} 个问题在首轮直接接受，另有 {int(float(q['second_round_trigger_count']))} 个问题实际执行第二轮；HotpotQA 的对应数量为 {int(float(h['first_round_accept_count']))} 和 {int(float(h['second_round_trigger_count']))}。Qasper 的第二轮触发率高于 HotpotQA，这一差异与科学论文中证据跨段落、跨章节分布的特点相符，但这里只是数据集层面的描述性现象，不能据此作因果解释。触发子集上，Qasper Evidence F1 从 {100 * float(q['triggered_first_evidence_metric']):.2f}\% 升至 {100 * float(q['triggered_final_evidence_metric']):.2f}\%，HotpotQA Supporting Fact F1 从 {100 * float(h['triggered_first_evidence_metric']):.2f}\% 升至 {100 * float(h['triggered_final_evidence_metric']):.2f}\%；平均绝对增益分别为 {100 * float(q['triggered_mean_evidence_gain']):.2f} 和 {100 * float(h['triggered_mean_evidence_gain']):.2f} 个百分点。

第二轮平均仅增加 {float(q['triggered_mean_added_atomic_blocks']):.2f}/{float(h['triggered_mean_added_atomic_blocks']):.2f} 个原子块和 {float(q['triggered_mean_added_context_tokens']):.1f}/{float(h['triggered_mean_added_context_tokens']):.1f} 个上下文 token，说明闭环的额外开销集中在少量未通过首轮验证的问题上，而不是对全部问题统一扩展。另一方面，两个数据集的扩展后接受率均为 0：实际进入第二轮的问题都在达到 $R_{{\max}}=2$ 时仍未跨过当前验证阈值；此外，Qasper 和 HotpotQA 分别有 {q_no_candidate} 和 {h_no_candidate} 个首轮不充分问题因无新候选提前停止。因此，证据集指标的平均改善并不等价于验证器最终接受，该现象也揭示了当前充分性规则偏保守以及候选扩展空间受限的边界。

\subsection{{检索轨迹案例分析}}

表~\ref{{tab:retrieval_trajectory_cases}} 展示两个可审计的真实轨迹。候选首先限定为实际执行第二轮、证据集 F1 提升且新增原子块命中标准证据的问题；Qasper 进一步要求参考证据至少包含两个块、最终 Answer F1 不低于 80\%，且新增且命中标准证据的原子块进入最终支持集，HotpotQA 则限定为 Bridge 问题。为避免选取增益最大的极端样本，案例均以原始严格候选的证据增益中位数为参照，并以问题 ID 处理并列。Qasper 和 HotpotQA 的严格候选数分别为 {int(q_case['selection']['candidate_count'])} 和 {int(h_case['selection']['candidate_count'])}；Qasper 其中有 {int(q_case['selection'].get('successful_candidate_count') or q_case['selection']['candidate_count'])} 个满足答案正确性与最终引用条件。所选样本的增益分别为 {100 * float(q_case['selection']['selected_evidence_gain']):.2f} 和 {100 * float(h_case['selection']['selected_evidence_gain']):.2f} 个百分点。

\begin{{table*}}[t]
\centering
\footnotesize
\caption{{Qasper 与 HotpotQA 的真实检索轨迹。证据编号均可通过来源映射回溯到原文；辅助节点只参与扩展，不作为最终支持证据。}}
\label{{tab:retrieval_trajectory_cases}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{p{{0.115\textwidth}}p{{0.415\textwidth}}p{{0.415\textwidth}}}}
\toprule
\textbf{{阶段}} & \textbf{{Qasper 案例}} & \textbf{{HotpotQA 案例}} \\
\midrule
Question / Gold
& 问题：{_latex(_short(q_case.get('question'), 180))}\newline 标准答案：{_latex(_gold_text(q_case))}
& 问题：{_latex(_short(h_case.get('question'), 180))}\newline 标准答案：{_latex(_gold_text(h_case))} \\
\midrule
BM25+Reranker & {_baseline_cell(q_case)} & {_baseline_cell(h_case)} \\
\midrule
Round 1 & {_round_one_cell(q_case)} & {_round_one_cell(h_case)} \\
\midrule
Diagnosis & {_diagnosis_cell(q_case)} & {_diagnosis_cell(h_case)} \\
\midrule
Round 2 & {_round_two_cell(q_case)} & {_round_two_cell(h_case)} \\
\midrule
Final Output & {_final_cell(q_case)} & {_final_cell(h_case)} \\
\bottomrule
\end{{tabular}}
\end{{table*}}

Qasper 案例询问“{_latex(_short(q_case.get('question'), 100))}”。首轮主要停留在一般实验设置、语言与模型背景，虽命中一个参考段落，但缺少直接列出数据集名称的原子证据；诊断结果将缺失需求记为相关证据不足和噪声过高，并建议沿语义桥扩展。第二轮新增块 \texttt{{[{_latex(', '.join(str(item.get('block_id')) for item in q_case.get('round_2_added_evidence') or []))}]}}，其中 \texttt{{[{_latex(', '.join(str(item) for item in q_case.get('round_2_added_gold_support_ids') or []))}]}} 的 Datasets 段明确给出 IITB 与 ILCI English--Hindi parallel corpora。该新增 gold 证据随后被最终支持集实际采用，使 Evidence F1 从 {100 * float(q_case['round_1']['evidence_f1']):.2f}\% 提高到 {100 * float(q_case['final_output']['evidence_f1']):.2f}\%，最终答案 F1 达到 {100 * float(q_case['final_output'].get('answer_f1') or 0.0):.2f}\%。相比之下，BM25+Reranker 检索到的是一般迁移学习背景，并将语言列表误作数据集，Evidence F1 为 {100 * float(q_case['bm25_reranker']['evidence_f1']):.2f}\%。这一轨迹同时体现了受控扩展带来的覆盖改善、答案纠正与可追溯引用，而所选增益仍等于原始严格候选中位数，并非极端最好样本。

HotpotQA 案例通过 Ross Pople 与 Pierre Boulez 建立跨条目联系。首轮已召回“post-war classical music”描述及 Ross Pople 的合作名单，第二轮又通过同章节上下文和共享术语关系加入 Ross Pople 与 London Festival Orchestra 的原子事实，Supporting Fact F1 从 {100 * float(h_case['round_1']['evidence_f1']):.2f}\% 提高到 {100 * float(h_case['final_output']['evidence_f1']):.2f}\%，最终答案为“{_latex(_short(h_case['final_output']['answer'], 80))}”。相比之下，BM25+Reranker 检索到相关条目却输出“{_latex(_short(h_case['bm25_reranker']['answer'], 50))}”。需要指出的是，该题的验证器把缺失类型标为表格上下文，而语料实际由句级事实构成，且第二轮还加入了一个无关块；这是规则诊断与预算选择仍可改进的真实失败模式。两个案例只用于解释统计中观察到的轨迹，不代表全部问题的普遍行为。

"""


def insert_addition(source: str, addition: str) -> str:
    if source.count(ANCHOR) != 1:
        raise ValueError("expected exactly one parameter-sensitivity subsection anchor")
    if r"\label{tab:closed_loop_behavior}" in source or r"\label{tab:retrieval_trajectory_cases}" in source:
        raise ValueError("closed-loop analysis already exists in source")
    return source.replace(ANCHOR, addition.rstrip() + "\n\n" + ANCHOR, 1)


def upsert_addition(source: str, addition: str) -> str:
    if ANALYSIS_START not in source:
        return insert_addition(source, addition)
    if source.count(ANCHOR) != 1:
        raise ValueError("expected exactly one parameter-sensitivity subsection anchor")
    if source.count(ANALYSIS_START) != 1 or source.count(TRAJECTORY_START) != 1:
        raise ValueError("expected exactly one existing closed-loop analysis block")
    start = source.index(ANALYSIS_START)
    end = source.index(ANCHOR)
    if start >= end:
        raise ValueError("existing closed-loop analysis must precede parameter sensitivity")
    return source[:start] + addition.rstrip() + "\n\n" + source[end:]


def _read_summaries(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["dataset"]): row for row in rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--backup", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tex_path = Path(args.tex)
    backup_path = Path(args.backup)
    source = tex_path.read_text(encoding="utf-8")
    summaries = _read_summaries(Path(args.summary))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    addition = render_addition(summaries, cases)
    rendered = upsert_addition(source, addition)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(tex_path, backup_path)
    tex_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"tex": str(tex_path.resolve()), "backup": str(backup_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
