import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from Core.configs.dataset_config import DatasetConfig
from Core.utils.json_safety import make_json_safe
from Eval.utils.utils import get_all_cost


EVIDENCE_K_VALUES = (1, 3, 5, 8, 10, 15)


def _normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def _as_keyword_groups(value: Any) -> List[List[str]]:
    if not value:
        return []
    if isinstance(value, str):
        return [[value]]
    groups = []
    for item in value:
        if isinstance(item, str):
            groups.append([item])
        elif isinstance(item, Iterable):
            group = [str(keyword) for keyword in item if str(keyword).strip()]
            if group:
                groups.append(group)
    return groups


def _coverage_ratio(groups: List[List[str]], text: str) -> float:
    if not groups:
        return 0.0
    normalized_text = _normalize_text(text)
    hits = 0
    for group in groups:
        if any(_normalize_text(keyword) in normalized_text for keyword in group):
            hits += 1
    return hits / len(groups)


def _optional_coverage_ratio(groups: List[List[str]], text: str) -> Optional[float]:
    if not groups:
        return None
    return _coverage_ratio(groups, text)


def _matches_keyword_group(text: str, group: List[str]) -> bool:
    normalized_text = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized_text for keyword in group)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _as_int_list(value: Any) -> List[int]:
    pages: List[int] = []
    for item in _as_list(value):
        try:
            pages.append(int(item))
        except (TypeError, ValueError):
            continue
    return pages


def _as_relation_types(value: Any) -> List[str]:
    relation_types: List[str] = []
    for item in _as_list(value):
        if isinstance(item, str):
            if item.strip():
                relation_types.append(item.strip())
        elif isinstance(item, dict):
            relation_type = item.get("relation_type") or item.get("type")
            if relation_type:
                relation_types.append(str(relation_type).strip())
    return relation_types


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_text_from_json_payload(payload: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(payload, str):
        if payload.strip():
            texts.append(payload)
    elif isinstance(payload, dict):
        for key in ["content", "text", "evidence", "page_content"]:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        anchor = payload.get("anchor")
        if isinstance(anchor, dict):
            value = anchor.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value)
    elif isinstance(payload, list):
        for item in payload:
            texts.extend(_extract_text_from_json_payload(item))
    return texts


def _normalize_role(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw.lower()
    role_map = [
        ("definition", ["definition", "define", "concept", "termdefinition"]),
        ("condition", ["condition"]),
        ("requirement", ["requirement", "requires"]),
        ("table", ["table"]),
        ("appendix", ["appendix"]),
        ("exception", ["exception"]),
        ("supplement", ["supplement"]),
        ("article", ["article"]),
        ("chapter", ["chapter"]),
    ]
    for role, keywords in role_map:
        if any(keyword in normalized for keyword in keywords):
            return role
    return normalized


def _section_text_from_payload(payload: Dict[str, Any], anchor: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in ["section", "section_id"]:
        value = payload.get(key)
        if value:
            values.append(str(value))
    for key in ["section", "section_id"]:
        value = anchor.get(key)
        if value:
            values.append(str(value))
    title_path = payload.get("title_path") or anchor.get("title_path")
    if isinstance(title_path, list):
        values.extend(str(item) for item in title_path if str(item).strip())
    elif title_path:
        values.append(str(title_path))
    return "\n".join(values)


def _relations_from_payload(payload: Any) -> List[Any]:
    relations: List[Any] = []
    if isinstance(payload, dict):
        if payload.get("relation_type") or payload.get("type"):
            relations.append(payload)
        payload_relations = payload.get("relations")
        if isinstance(payload_relations, list):
            relations.extend(payload_relations)
        elif payload_relations:
            relations.append(payload_relations)
        for key in ["edges", "links"]:
            value = payload.get(key)
            if value:
                relations.extend(_relations_from_payload(value))
    elif isinstance(payload, list):
        for item in payload:
            relations.extend(_relations_from_payload(item))
    return relations


def _item_text(item: Dict[str, Any]) -> str:
    text = item.get("text", "")
    if isinstance(text, list):
        return "\n".join(str(part) for part in text if str(part).strip())
    return "" if text is None else str(text)


def _payload_to_retrieval_item(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        texts = _extract_text_from_json_payload(payload)
        if not texts:
            return None
        return {"text": "\n".join(texts), "relations": _relations_from_payload(payload)}

    anchor = payload.get("anchor") if isinstance(payload.get("anchor"), dict) else {}
    texts = _extract_text_from_json_payload(payload)
    item = {
        "text": "\n".join(texts),
        "role": _normalize_role(
            payload.get("role")
            or payload.get("node_type")
            or anchor.get("node_type")
            or payload.get("anchor_kind")
            or anchor.get("anchor_kind")
        ),
        "node_type": payload.get("node_type") or anchor.get("node_type"),
        "page": payload.get("page") or anchor.get("page"),
        "section": _section_text_from_payload(payload, anchor),
        "relations": _relations_from_payload(payload),
    }
    if not item["text"] and not item["section"]:
        return None
    return item


def _json_sort_key(path: Path) -> Tuple[int, Any]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def collect_query_relations(query_dir: Path) -> List[Any]:
    relations: List[Any] = []
    for name in ["retrieval_res.json", "evidence_chain.json"]:
        path = query_dir / name
        if path.exists():
            relations.extend(_relations_from_payload(_load_json(path)))
    return relations


def collect_evidence_chain_items(query_dir: Path) -> List[Dict[str, Any]]:
    evidence_path = query_dir / "evidence_chain.json"
    if not evidence_path.exists():
        return []
    payload = _load_json(evidence_path)
    if isinstance(payload, dict):
        for key in ["evidence_chain", "chain", "items", "nodes"]:
            values = payload.get(key)
            if isinstance(values, list) and values:
                return [
                    item
                    for item in (_payload_to_retrieval_item(value) for value in values)
                    if item is not None
                ]
    values = payload if isinstance(payload, list) else [payload]
    return [
        item
        for item in (_payload_to_retrieval_item(value) for value in values)
        if item is not None
    ]


def collect_retrieval_items(
    query_dir: Path, retrieval_ids: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    retrieval_path = query_dir / "retrieval_res.json"
    if retrieval_path.exists():
        retrieval_payload = _load_json(retrieval_path)
        if isinstance(retrieval_payload, dict):
            for key in ["budgeted_results", "ranked_results", "hybrid_results", "coarse_results"]:
                values = retrieval_payload.get(key)
                if isinstance(values, list) and values:
                    items = [
                        item
                        for item in (_payload_to_retrieval_item(value) for value in values)
                        if item is not None
                    ]
                    if items:
                        return items

    evidence_path = query_dir / "evidence_chain.json"
    if evidence_path.exists():
        payload = _load_json(evidence_path)
        values = payload if isinstance(payload, list) else [payload]
        return [
            item
            for item in (_payload_to_retrieval_item(value) for value in values)
            if item is not None
        ]

    ignored_names = {"result.json", "retrieval_res.json", "evidence_chain.json"}
    if retrieval_ids:
        ordered_items: List[Dict[str, Any]] = []
        seen_paths = set()
        for retrieval_id in retrieval_ids:
            json_path = query_dir / f"{retrieval_id}.json"
            if not json_path.exists():
                continue
            item = _payload_to_retrieval_item(_load_json(json_path))
            if item is not None:
                ordered_items.append(item)
                seen_paths.add(json_path.name)
        for json_path in sorted(query_dir.glob("*.json"), key=_json_sort_key):
            if json_path.name in ignored_names or json_path.name in seen_paths:
                continue
            item = _payload_to_retrieval_item(_load_json(json_path))
            if item is not None:
                ordered_items.append(item)
        if ordered_items:
            return ordered_items

    retrieval_items: List[Dict[str, Any]] = []
    for json_path in sorted(query_dir.glob("*.json"), key=_json_sort_key):
        if json_path.name in ignored_names:
            continue
        payload = _load_json(json_path)
        item = _payload_to_retrieval_item(payload)
        if item is not None:
            retrieval_items.append(item)
    return retrieval_items


def collect_retrieval_texts(query_dir: Path) -> List[str]:
    return [_item_text(item) for item in collect_retrieval_items(query_dir)]


def _evidence_recall_at_k(
    groups: List[List[str]], retrieval_items: List[Dict[str, Any]], k: int
) -> Optional[float]:
    if not groups:
        return None
    evidence_text = "\n".join(_item_text(item) for item in retrieval_items[:k])
    return _coverage_ratio(groups, evidence_text)


def _retrieval_item_matches_any_group(
    item: Dict[str, Any], groups: List[List[str]]
) -> bool:
    text = _item_text(item)
    return any(_matches_keyword_group(text, group) for group in groups)


def _perfect_evidence_recall_at_k(
    groups: List[List[str]], retrieval_items: List[Dict[str, Any]], k: int
) -> Optional[float]:
    recall = _evidence_recall_at_k(groups, retrieval_items, k)
    if recall is None:
        return None
    return 1.0 if recall >= 1.0 else 0.0


def _irrelevant_evidence_ratio_at_k(
    groups: List[List[str]], retrieval_items: List[Dict[str, Any]], k: int
) -> Optional[float]:
    if not groups:
        return None
    top_items = retrieval_items[:k]
    if not top_items:
        return 0.0
    irrelevant = sum(
        1 for item in top_items if not _retrieval_item_matches_any_group(item, groups)
    )
    return irrelevant / len(top_items)


def _mrr(groups: List[List[str]], retrieval_items: List[Dict[str, Any]]) -> Optional[float]:
    if not groups:
        return None
    for rank, item in enumerate(retrieval_items, start=1):
        text = _item_text(item)
        if any(_matches_keyword_group(text, group) for group in groups):
            return 1.0 / rank
    return 0.0


def _role_keywords(value: Any) -> Dict[str, List[List[str]]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(role): _as_keyword_groups(keywords)
        for role, keywords in value.items()
        if _as_keyword_groups(keywords)
    }


def _role_hit_rate(
    gold_roles: List[str],
    retrieval_items: List[Dict[str, Any]],
    role_keywords: Dict[str, List[List[str]]],
) -> Optional[float]:
    if not gold_roles:
        return None
    predicted_roles = {
        role
        for role in (
            _normalize_role(item.get("role") or item.get("node_type"))
            for item in retrieval_items
        )
        if role
    }
    hits = 0
    all_text = "\n".join(_item_text(item) for item in retrieval_items)
    for role in gold_roles:
        normalized_role = _normalize_role(role) or str(role)
        if normalized_role in predicted_roles:
            hits += 1
            continue
        keywords = role_keywords.get(role) or role_keywords.get(normalized_role)
        if keywords and _coverage_ratio(keywords, all_text) > 0:
            hits += 1
    return hits / len(gold_roles)


def _page_hit_rate(
    gold_pages: List[int], retrieval_items: List[Dict[str, Any]]
) -> Optional[float]:
    if not gold_pages:
        return None
    predicted_pages = set()
    for item in retrieval_items:
        try:
            predicted_pages.add(int(item.get("page")))
        except (TypeError, ValueError):
            continue
    if not predicted_pages:
        return 0.0
    return len(set(gold_pages) & predicted_pages) / len(set(gold_pages))


def _section_hit_rate(
    gold_sections: List[str], retrieval_items: List[Dict[str, Any]]
) -> Optional[float]:
    gold_sections = [section for section in gold_sections if str(section).strip()]
    if not gold_sections:
        return None
    predicted_sections = [_normalize_text(item.get("section", "")) for item in retrieval_items]
    hits = 0
    for section in gold_sections:
        normalized_gold = _normalize_text(section)
        if any(
            normalized_gold in predicted or predicted in normalized_gold
            for predicted in predicted_sections
            if predicted
        ):
            hits += 1
    return hits / len(gold_sections)


def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return float(np.mean(numeric_values))


def _relation_hit_rate(
    gold_relations: List[str], predicted_relations: List[str]
) -> Optional[float]:
    if not gold_relations:
        return None
    predicted = {_normalize_text(relation) for relation in predicted_relations}
    hits = sum(1 for relation in gold_relations if _normalize_text(relation) in predicted)
    return hits / len(gold_relations)


def _truncate_for_judge(text: Any, limit: int = 700) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_judge_evidence_items(
    items: List[Dict[str, Any]], max_items: int = 8
) -> str:
    formatted_items: List[Dict[str, Any]] = []
    for rank, item in enumerate(items[:max_items], start=1):
        formatted_items.append(
            {
                "rank": rank,
                "role": item.get("role"),
                "node_type": item.get("node_type"),
                "page": item.get("page"),
                "section": _truncate_for_judge(item.get("section"), 220),
                "relations": _as_relation_types(item.get("relations")),
                "text": _truncate_for_judge(_item_text(item)),
            }
        )
    return json.dumps(formatted_items, ensure_ascii=False)


def build_hydro_judge_content(
    item: Dict[str, Any],
    retrieval_items: Optional[List[Dict[str, Any]]] = None,
    evidence_chain_items: Optional[List[Dict[str, Any]]] = None,
    query_relations: Optional[List[Any]] = None,
) -> str:
    retrieval_items = retrieval_items or []
    evidence_chain_items = evidence_chain_items or []
    keypoints = item.get("answer_keypoints") or []
    return (
        f"Question: {item.get('question', '')}\n"
        f"Reference Answer: {item.get('answer', '')}\n"
        f"Reference Key Points: {json.dumps(keypoints, ensure_ascii=False)}\n"
        f"Gold Evidence Keywords: {json.dumps(item.get('gold_evidence_keywords') or [], ensure_ascii=False)}\n"
        f"Gold Evidence Pages: {json.dumps(item.get('gold_evidence_pages') or [], ensure_ascii=False)}\n"
        f"Gold Evidence Sections: {json.dumps(item.get('gold_evidence_sections') or [], ensure_ascii=False)}\n"
        f"Gold Relations: {json.dumps(_as_relation_types(item.get('gold_relations')), ensure_ascii=False)}\n"
        f"Gold Path Relations: {json.dumps(_as_relation_types(item.get('gold_path_relations')), ensure_ascii=False)}\n"
        f"Gold Support Relations: {json.dumps(_as_relation_types(item.get('gold_relation_support')), ensure_ascii=False)}\n"
        f"Retrieved Evidence: {_format_judge_evidence_items(retrieval_items)}\n"
        f"Evidence Chain: {_format_judge_evidence_items(evidence_chain_items)}\n"
        f"Retrieved Relation Types: {json.dumps(_as_relation_types(query_relations or []), ensure_ascii=False)}\n"
        f"Model Response: {item.get('output', '')}\n"
    )


class HydroAnswerJudge:
    def __init__(self, api_config_path: str):
        config_path = Path(api_config_path)
        with config_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(
                f"Hydro judge API config must contain base_url, api_key, and model_name: {config_path}"
            )
        self.client = OpenAI(api_key=lines[1], base_url=lines[0])
        self.model_name = lines[2]
        prompt_path = Path(__file__).resolve().parent / "hydro_judge_prompt.md"
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_response(text: str) -> Tuple[str, str, float]:
        parsed: Dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip().lower()] = value.strip()
        score_raw = parsed.get("score", "0")
        try:
            score = float(score_raw)
        except ValueError:
            score = 0.0
        score = max(0.0, min(1.0, score))
        return (
            parsed.get("extracted results", ""),
            parsed.get("format", "KeyPoints"),
            score,
        )

    def judge(
        self,
        item: Dict[str, Any],
        retrieval_items: Optional[List[Dict[str, Any]]] = None,
        evidence_chain_items: Optional[List[Dict[str, Any]]] = None,
        query_relations: Optional[List[Any]] = None,
    ) -> Tuple[str, str, float, str]:
        content = build_hydro_judge_content(
            item,
            retrieval_items=retrieval_items,
            evidence_chain_items=evidence_chain_items,
            query_relations=query_relations,
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        extracted, answer_format, score = self._parse_response(raw)
        return extracted, answer_format, score, raw


def evaluate_hydro_result_item(
    item: Dict[str, Any],
    retrieval_texts: List[str],
    llm_score: Optional[float],
    extracted_res: Optional[str],
    pred_format: Optional[str] = None,
    judge_raw: Optional[str] = None,
    retrieval_items: Optional[List[Dict[str, Any]]] = None,
    query_relations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    output = item.get("output", "")
    keypoint_groups = _as_keyword_groups(item.get("answer_keypoints"))
    evidence_groups = _as_keyword_groups(item.get("gold_evidence_keywords"))
    if retrieval_items is None:
        retrieval_items = [{"text": text, "relations": []} for text in retrieval_texts]
    if not retrieval_texts:
        retrieval_texts = [_item_text(entry) for entry in retrieval_items]
    evidence_text = "\n".join(retrieval_texts)
    gold_chain_roles = [str(role) for role in _as_list(item.get("gold_chain_roles")) if str(role).strip()]
    gold_pages = _as_int_list(item.get("gold_evidence_pages"))
    gold_sections = [str(section) for section in _as_list(item.get("gold_evidence_sections")) if str(section).strip()]
    gold_relations = _as_relation_types(item.get("gold_relations"))
    gold_path_relations = _as_relation_types(item.get("gold_path_relations"))
    gold_support_relations = _as_relation_types(item.get("gold_relation_support")) or gold_path_relations or gold_relations
    predicted_relations = _as_relation_types(query_relations or [])
    for retrieval_item in retrieval_items:
        predicted_relations.extend(_as_relation_types(retrieval_item.get("relations")))

    evaluated = dict(item)
    evaluated["pred"] = extracted_res if extracted_res is not None else output
    evaluated["pred_format"] = pred_format or item.get("answer_format", "KeyPoints")
    evaluated["llm_score"] = llm_score
    evaluated["evidence_aware_llm_score"] = llm_score
    evaluated["extracted_res"] = judge_raw or extracted_res
    evaluated["keypoint_recall"] = round(_coverage_ratio(keypoint_groups, output), 6)
    evaluated["evidence_keyword_recall"] = round(
        _coverage_ratio(evidence_groups, evidence_text), 6
    )
    for k in EVIDENCE_K_VALUES:
        evaluated[f"evidence_recall@{k}"] = (
            round(value, 6)
            if (value := _evidence_recall_at_k(evidence_groups, retrieval_items, k))
            is not None
            else None
        )
        evaluated[f"perfect_evidence_recall@{k}"] = (
            round(value, 6)
            if (
                value := _perfect_evidence_recall_at_k(
                    evidence_groups, retrieval_items, k
                )
            )
            is not None
            else None
        )
        evaluated[f"irrelevant_evidence_ratio@{k}"] = (
            round(value, 6)
            if (
                value := _irrelevant_evidence_ratio_at_k(
                    evidence_groups, retrieval_items, k
                )
            )
            is not None
            else None
        )
    evaluated["mrr"] = (
        round(value, 6)
        if (value := _mrr(evidence_groups, retrieval_items)) is not None
        else None
    )
    evaluated["evidence_chain_coverage"] = (
        round(value, 6)
        if (
            value := _role_hit_rate(
                gold_chain_roles,
                retrieval_items,
                _role_keywords(item.get("gold_chain_role_keywords")),
            )
        )
        is not None
        else None
    )
    for k in EVIDENCE_K_VALUES:
        evaluated[f"page_hit_rate@{k}"] = (
            round(value, 6)
            if (value := _page_hit_rate(gold_pages, retrieval_items[:k])) is not None
            else None
        )
        evaluated[f"section_hit_rate@{k}"] = (
            round(value, 6)
            if (value := _section_hit_rate(gold_sections, retrieval_items[:k]))
            is not None
            else None
        )
        evaluated[f"page_section_hit_rate@{k}"] = (
            round(value, 6)
            if (
                value := _mean_optional(
                    [
                        evaluated[f"page_hit_rate@{k}"],
                        evaluated[f"section_hit_rate@{k}"],
                    ]
                )
            )
            is not None
            else None
        )
    evaluated["page_hit_rate"] = evaluated.get("page_hit_rate@5")
    evaluated["section_hit_rate"] = evaluated.get("section_hit_rate@5")
    evaluated["page_section_hit_rate"] = evaluated.get("page_section_hit_rate@5")
    evaluated["relation_hit_rate"] = (
        round(value, 6)
        if (value := _relation_hit_rate(gold_relations, predicted_relations)) is not None
        else None
    )
    evaluated["path_completeness"] = (
        round(value, 6)
        if (value := _relation_hit_rate(gold_path_relations, predicted_relations))
        is not None
        else None
    )
    evaluated["relation_support_rate"] = (
        round(value, 6)
        if (value := _relation_hit_rate(gold_support_relations, predicted_relations))
        is not None
        else None
    )
    evaluated["retrieved_count"] = len(retrieval_items)
    return evaluated


def eval_single_file(
    res_path: str,
    skip_llm_judge: bool = False,
    api_config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    res_dir = Path(res_path)
    res_data = _load_json(res_dir / "final_results.json")
    judge = None
    if not skip_llm_judge:
        judge = HydroAnswerJudge(api_config_path or "Eval/utils/api.txt")

    evaluated_items = []
    for idx, item in enumerate(res_data, start=1):
        query_dir = res_dir / f"query_{idx:03d}"
        retrieval_items = collect_retrieval_items(
            query_dir, retrieval_ids=item.get("retrieved_node_ids")
        )
        evidence_chain_items = collect_evidence_chain_items(query_dir)
        retrieval_texts = [_item_text(entry) for entry in retrieval_items]
        query_relations = collect_query_relations(query_dir)
        extracted = None
        pred_format = None
        llm_score = None
        judge_raw = None
        if judge is not None:
            extracted, pred_format, llm_score, judge_raw = judge.judge(
                item,
                retrieval_items=retrieval_items,
                evidence_chain_items=evidence_chain_items,
                query_relations=query_relations,
            )
        evaluated_items.append(
            evaluate_hydro_result_item(
                item=item,
                retrieval_texts=retrieval_texts,
                llm_score=llm_score,
                extracted_res=extracted,
                pred_format=pred_format,
                judge_raw=judge_raw,
                retrieval_items=retrieval_items,
                query_relations=query_relations,
            )
        )

    with (res_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(evaluated_items, f, ensure_ascii=False, indent=2)
    return evaluated_items


def _mean(items: List[Dict[str, Any]], key: str) -> float:
    values = [item[key] for item in items if item.get(key) is not None]
    if not values:
        return 0.0
    return round(float(np.mean(values)), 6)


def eval_hydro(
    data_df: pd.DataFrame,
    data_cfg: DatasetConfig,
    method: str,
    max_workers: int = 1,
    skip_llm_judge: bool = False,
    api_config_path: Optional[str] = None,
):
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])
    result: List[Dict[str, Any]] = []

    for (doc_uuid, _), _group in tqdm(document_groups, desc="Processing Documents"):
        dir_name = f"eval_{data_cfg.dataset_name}_{method}"
        doc_res_dir = os.path.join(data_cfg.working_dir, str(doc_uuid), dir_name)
        result.extend(
            eval_single_file(
                doc_res_dir,
                skip_llm_judge=skip_llm_judge,
                api_config_path=api_config_path,
            )
        )

    evidence_available = [
        item for item in result if item.get("gold_evidence_keywords")
    ]
    chain_available = [item for item in result if item.get("gold_chain_roles")]
    page_section_available = [
        item
        for item in result
        if item.get("gold_evidence_pages") or item.get("gold_evidence_sections")
    ]
    score_dict: Dict[str, Any] = {}
    for k in EVIDENCE_K_VALUES:
        score_dict[f"Avg Evidence Recall@{k}"] = _mean(
            evidence_available, f"evidence_recall@{k}"
        )
    score_dict.update(
        {
            "Avg MRR": _mean(evidence_available, "mrr"),
            "Avg Evidence Chain Coverage": _mean(
                chain_available, "evidence_chain_coverage"
            ),
            "Avg retrieved_count": _mean(result, "retrieved_count"),
            "Evidence available samples": len(evidence_available),
            "Chain available samples": len(chain_available),
            "Page/Section available samples": len(page_section_available),
            "Total samples": len(result),
        }
    )
    for k in EVIDENCE_K_VALUES:
        score_dict[f"Avg Perfect Evidence Recall@{k}"] = _mean(
            evidence_available, f"perfect_evidence_recall@{k}"
        )
        score_dict[f"Avg Irrelevant Evidence Ratio@{k}"] = _mean(
            evidence_available, f"irrelevant_evidence_ratio@{k}"
        )
        score_dict[f"Avg Page/Section Hit@{k}"] = _mean(
            page_section_available, f"page_section_hit_rate@{k}"
        )
    cost_dict = get_all_cost(data_df, data_cfg, method)
    score_dict.update({k: v for k, v in cost_dict.items() if k not in score_dict})

    save_dir = Path(data_cfg.working_dir) / "0_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    detail_path = save_dir / f"final_eval_{data_cfg.dataset_name}_{method}.json"
    score_path = save_dir / f"final_eval_{data_cfg.dataset_name}_{method}.score.json"

    priority_keys = [
        "question",
        "answer",
        "pred",
        "evidence_aware_llm_score",
        "llm_score",
        "keypoint_recall",
        "evidence_keyword_recall",
    ]
    for k in EVIDENCE_K_VALUES:
        priority_keys.extend(
            [
                f"evidence_recall@{k}",
                f"perfect_evidence_recall@{k}",
                f"irrelevant_evidence_ratio@{k}",
                f"page_section_hit_rate@{k}",
            ]
        )
    priority_keys.extend(
        [
            "mrr",
            "evidence_chain_coverage",
            "page_hit_rate",
            "section_hit_rate",
            "page_section_hit_rate",
            "relation_hit_rate",
            "path_completeness",
            "relation_support_rate",
            "retrieved_count",
            "output",
        ]
    )
    sorted_result = []
    for item in result:
        sorted_item = {k: item[k] for k in priority_keys if k in item}
        sorted_item.update({k: v for k, v in item.items() if k not in priority_keys})
        sorted_result.append(make_json_safe(sorted_item))

    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2, allow_nan=False)
    with score_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(score_dict), f, ensure_ascii=False, indent=2, allow_nan=False)

    print("--------------------------------------")
    print(f"total samples: {len(result)}")
    print(f"Avg Evidence Recall@5: {score_dict['Avg Evidence Recall@5']:.6f}")
    print(f"Avg MRR: {score_dict['Avg MRR']:.6f}")
    print(
        "Avg Perfect Evidence Recall@5: "
        f"{score_dict['Avg Perfect Evidence Recall@5']:.6f}"
    )
    print(
        "Avg Irrelevant Evidence Ratio@5: "
        f"{score_dict['Avg Irrelevant Evidence Ratio@5']:.6f}"
    )
    print(
        f"Avg Evidence Chain Coverage: {score_dict['Avg Evidence Chain Coverage']:.6f}"
    )
    print(f"Avg Page/Section Hit@5: {score_dict['Avg Page/Section Hit@5']:.6f}")
    print(f"Saved detailed results to {detail_path}")
    print(f"Saved score summary to {score_path}")
