import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.utils.bm25 import BM25


HydroNodeType = Literal[
    "Chapter",
    "Article",
    "TermDefinition",
    "Table",
    "Appendix",
    "Concept",
    "Requirement",
    "Condition",
    "Exception",
    "Supplement",
]
AnchorKind = Literal["tree", "semantic"]
HydroRelationType = Literal[
    "defines",
    "condition_of",
    "requires",
    "refers_to",
    "parameter_of",
    "supplements",
    "exception_to",
]


_CJK_BLOCK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_./-]+")
_CHAPTER_RE = re.compile(r"(第[一二三四五六七八九十百千万\d]+[章节篇])")
_ARTICLE_RE = re.compile(r"(第\s*[一二三四五六七八九十百千万\d.．]+条|\b\d+(?:[.．]\d+)+\b)")
_EXPLICIT_TABLE_RE = re.compile(
    r"(附表\s*[A-Za-z一二三四五六七八九十百千万\d]+(?:[-－—–.．]\d+)*|"
    r"表\s*[A-Za-z一二三四五六七八九十百千万\d]+(?:[-－—–.．]\d+)*)"
)
_FOLLOWING_TABLE_RE = re.compile(r"(?:如下表|下表)")
_TABLE_RE = re.compile(
    r"(附表\s*[A-Za-z一二三四五六七八九十百千万\d]+(?:[-－—–.．]\d+)*|"
    r"表\s*[A-Za-z一二三四五六七八九十百千万\d]+(?:[-－—–.．]\d+)*|"
    r"如下表|下表)"
)
_APPENDIX_RE = re.compile(r"(附录\s*[A-Za-zＡ-Ｚａ-ｚ一二三四五六七八九十百千万\d]+)")
_DEFINITION_RE = re.compile(r"(?:是指|系指|指的是|定义为|称为)")
_CONDITION_RE = re.compile(r"(?:当.+?时|在.+?情况下|若|如果)")
_ACTION_WORDS = [
    "及时",
    "加强",
    "建立",
    "完善",
    "明确",
    "优化",
    "制定",
    "开展",
    "实现",
    "支撑",
    "提高",
    "遵循",
    "统筹",
    "履行",
    "分析",
    "形成",
    "进行",
    "根据",
    "结合",
    "采取",
    "满足",
    "具备",
    "采用",
    "包括",
    "完成",
    "设置",
    "配置",
    "提供",
    "开启",
    "关闭",
    "启动",
    "发布",
    "报备",
    "优选",
    "编制",
    "记录",
    "报送",
]
_NOMINAL_REQUIREMENT_PHRASES = [
    "技术要求",
    "功能要求",
    "基本要求",
    "标准要求",
    "保障要求",
    "文件要求",
    "要求等标准制定",
]
_POLICY_BACKGROUND_RE = re.compile(
    r"(规划纲要|新型基础设施建设规划|决策部署|相继出台|指导意见|顶层设计|实施方案|明确提出)"
)


def hydro_tokenize(text: str) -> List[str]:
    """Tokenize Chinese regulation text without external dependencies."""
    tokens: List[str] = []
    for match in _TOKEN_RE.findall((text or "").lower()):
        if _CJK_BLOCK_RE.fullmatch(match):
            tokens.extend(match)
            tokens.extend(match[i : i + 2] for i in range(max(len(match) - 1, 0)))
        else:
            tokens.append(match)
    return [token for token in tokens if token.strip()]


class HydroBM25(BM25):
    def _tokenize(self, text: str) -> List[str]:
        return hydro_tokenize(text)


class EvidenceAnchor(BaseModel):
    node_id: int
    doc_name: Optional[str] = None
    section_id: str = ""
    page: Optional[int] = None
    node_type: HydroNodeType
    text: str = ""
    title_path: List[str] = Field(default_factory=list)
    parent_id: Optional[int] = None
    anchor_kind: AnchorKind = "tree"
    source_node_id: Optional[int] = None
    attributes: Dict[str, str] = Field(default_factory=dict)


class HydroRelation(BaseModel):
    source_id: int
    target_id: int
    relation_type: HydroRelationType
    evidence: str = ""
    weight: float = 1.0


class HRIIndex:
    _INDEX_FILE = "hri_index.json"
    _BM25_FILE = "hri_bm25.pkl"

    def __init__(
        self,
        save_dir: str,
        anchors: Optional[Dict[int, EvidenceAnchor]] = None,
        relations: Optional[List[HydroRelation]] = None,
        bm25_node_ids: Optional[List[int]] = None,
    ):
        self.save_dir = save_dir
        self.anchors: Dict[int, EvidenceAnchor] = anchors or {}
        self.relations: List[HydroRelation] = relations or []
        self.bm25_node_ids: List[int] = bm25_node_ids or []
        self._out_relations: Dict[int, List[HydroRelation]] = defaultdict(list)
        self._in_relations: Dict[int, List[HydroRelation]] = defaultdict(list)
        self._rebuild_relation_maps()

    @classmethod
    def from_tree(cls, tree: DocumentTree, save_dir: str) -> "HRIIndex":
        hri = cls(save_dir=save_dir)
        for node in tree.get_nodes(hasRoot=False):
            text = cls._node_text(node)
            if not text:
                continue
            node_type = cls.infer_node_type(node)
            page = cls._human_page(node)
            title_path = cls._title_path(tree, node)
            anchor = EvidenceAnchor(
                node_id=node.index_id,
                doc_name=tree.meta_info.file_name,
                section_id=cls._section_id(node, node_type),
                page=page,
                node_type=node_type,
                text=text,
                title_path=title_path,
                parent_id=node.parent.index_id if node.parent else None,
            )
            hri.anchors[node.index_id] = anchor

        hri.relations = hri._extract_relations()
        hri._rebuild_relation_maps()
        return hri

    @staticmethod
    def get_index_path(save_dir: str) -> str:
        return os.path.join(save_dir, HRIIndex._INDEX_FILE)

    @staticmethod
    def get_bm25_path(save_dir: str) -> str:
        return os.path.join(save_dir, HRIIndex._BM25_FILE)

    @classmethod
    def load_from_dir(cls, save_dir: str) -> "HRIIndex":
        with open(cls.get_index_path(save_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        anchors = {
            int(anchor["node_id"]): EvidenceAnchor(**anchor)
            for anchor in data.get("anchors", [])
        }
        relations = [HydroRelation(**rel) for rel in data.get("relations", [])]
        return cls(
            save_dir=save_dir,
            anchors=anchors,
            relations=relations,
            bm25_node_ids=[int(node_id) for node_id in data.get("bm25_node_ids", [])],
        )

    def save_to_dir(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.get_index_path(self.save_dir), "w", encoding="utf-8") as f:
            json.dump(self.to_json_data(), f, ensure_ascii=False, indent=2)

    def to_json_data(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "anchors": [
                anchor.model_dump()
                for _, anchor in sorted(self.anchors.items(), key=lambda item: item[0])
            ],
            "relations": [relation.model_dump() for relation in self.relations],
            "bm25_node_ids": self.bm25_node_ids,
        }

    def build_bm25(self) -> HydroBM25:
        node_ids = sorted(
            node_id
            for node_id, anchor in self.anchors.items()
            if anchor.anchor_kind == "tree"
        )
        corpus = [self._bm25_document(self.anchors[node_id]) for node_id in node_ids]
        bm25 = HydroBM25(corpus)
        bm25.initialize()
        self.bm25_node_ids = node_ids
        return bm25

    def save_bm25(self, bm25: HydroBM25) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.get_bm25_path(self.save_dir), "wb") as f:
            pickle.dump(bm25, f)

    @classmethod
    def load_bm25(cls, save_dir: str) -> HydroBM25:
        with open(cls.get_bm25_path(save_dir), "rb") as f:
            return pickle.load(f)

    def search_bm25(self, bm25: HydroBM25, query: str, top_k: int) -> List[Dict[str, Any]]:
        results = bm25.search(query_text=query, top_k=top_k)
        if not self.bm25_node_ids:
            self.bm25_node_ids = sorted(self.anchors.keys())

        mapped_results: List[Dict[str, Any]] = []
        for item in results:
            doc_idx = int(item["id"])
            if doc_idx >= len(self.bm25_node_ids):
                continue
            node_id = self.bm25_node_ids[doc_idx]
            anchor = self.anchors.get(node_id)
            if anchor is None:
                continue
            mapped_results.append(
                {
                    "node_id": node_id,
                    "score": item.get("score", 0.0),
                    "node_type": anchor.node_type,
                    "section_id": anchor.section_id,
                    "page": anchor.page,
                    "text": anchor.text,
                    "anchor": anchor.model_dump(),
                }
            )
        return mapped_results

    def get_related_relations(
        self,
        node_ids: Iterable[int],
        expand_depth: int = 1,
        relation_types: Optional[Iterable[str]] = None,
    ) -> List[HydroRelation]:
        allowed = set(relation_types) if relation_types else None
        frontier = set(node_ids)
        visited = set(frontier)
        selected: List[HydroRelation] = []
        seen = set()

        for _ in range(max(expand_depth, 0)):
            next_frontier = set()
            for node_id in frontier:
                for relation in self._out_relations.get(node_id, []) + self._in_relations.get(node_id, []):
                    if allowed and relation.relation_type not in allowed:
                        continue
                    key = (relation.source_id, relation.target_id, relation.relation_type)
                    if key not in seen:
                        selected.append(relation)
                        seen.add(key)
                    for related_id in (relation.source_id, relation.target_id):
                        if related_id not in visited:
                            next_frontier.add(related_id)
                            visited.add(related_id)
            frontier = next_frontier
            if not frontier:
                break
        return selected

    def get_anchor(self, node_id: int) -> Optional[EvidenceAnchor]:
        return self.anchors.get(node_id)

    def _rebuild_relation_maps(self) -> None:
        self._out_relations = defaultdict(list)
        self._in_relations = defaultdict(list)
        for relation in self.relations:
            self._out_relations[relation.source_id].append(relation)
            self._in_relations[relation.target_id].append(relation)

    def _extract_relations(self) -> List[HydroRelation]:
        relations: List[HydroRelation] = []
        seen = set()
        table_lookup = self._build_table_lookup()
        appendix_lookup = self._build_appendix_lookup()
        semantic_anchor_ids: Dict[tuple, int] = {}
        next_semantic_id = min([0, *self.anchors.keys()]) - 1

        def add(source_id: int, target_id: int, relation_type: HydroRelationType, evidence: str, weight: float = 1.0):
            if source_id not in self.anchors or target_id not in self.anchors:
                return
            if source_id == target_id:
                return
            key = (source_id, target_id, relation_type)
            if key in seen:
                return
            seen.add(key)
            relations.append(
                HydroRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    evidence=evidence,
                    weight=weight,
                )
            )

        def add_semantic_anchor(
            source: EvidenceAnchor,
            node_type: HydroNodeType,
            section_id: str,
            text: str,
            attributes: Optional[Dict[str, str]] = None,
        ) -> int:
            nonlocal next_semantic_id
            normalized = self._normalize_semantic_text(text)
            key = (source.node_id, node_type, normalized)
            if key in semantic_anchor_ids:
                return semantic_anchor_ids[key]

            node_id = next_semantic_id
            next_semantic_id -= 1
            self.anchors[node_id] = EvidenceAnchor(
                node_id=node_id,
                doc_name=source.doc_name,
                section_id=section_id,
                page=source.page,
                node_type=node_type,
                text=text.strip(),
                title_path=source.title_path,
                parent_id=source.node_id,
                anchor_kind="semantic",
                source_node_id=source.node_id,
                attributes=attributes or {},
            )
            semantic_anchor_ids[key] = node_id
            return node_id

        for node_id, anchor in list(self.anchors.items()):
            text = anchor.text
            if anchor.node_type == "TermDefinition":
                term = self._definition_term(text)
                if term:
                    definition = self._definition_text(text)
                    concept_id = add_semantic_anchor(
                        anchor,
                        "Concept",
                        f"定义对象：{term}",
                        term,
                        attributes={"term": term, "definition": definition},
                    )
                    add(node_id, concept_id, "defines", self._first_sentence(text), weight=0.95)

            for sentence in self._sentences(text):
                requirement_info = self._parse_requirement(sentence)
                requirement_id = None
                if requirement_info:
                    requirement_text = requirement_info["text"]
                    if requirement_text:
                        requirement_id = add_semantic_anchor(
                            anchor,
                            "Requirement",
                            "规范要求",
                            requirement_text,
                            attributes=requirement_info,
                        )
                        add(node_id, requirement_id, "requires", sentence[:160], weight=0.8)

                condition_text = self._condition_text(sentence)
                if condition_text:
                    condition_id = add_semantic_anchor(
                        anchor,
                        "Condition",
                        "适用条件",
                        condition_text,
                        attributes={"condition": condition_text},
                    )
                    target_id = requirement_id if requirement_id is not None else node_id
                    add(condition_id, target_id, "condition_of", sentence[:160], weight=0.8)

                exception_text = self._exception_text(sentence)
                if exception_text:
                    exception_id = add_semantic_anchor(
                        anchor,
                        "Exception",
                        "例外限制",
                        exception_text,
                        attributes={"exception": exception_text},
                    )
                    target_id = requirement_id if requirement_id is not None else node_id
                    add(exception_id, target_id, "exception_to", sentence[:160], weight=0.85)

                supplement_text = self._supplement_text(sentence)
                if supplement_text:
                    supplement_id = add_semantic_anchor(
                        anchor,
                        "Supplement",
                        "补充说明",
                        supplement_text,
                        attributes={"supplement": supplement_text},
                    )
                    add(node_id, supplement_id, "supplements", sentence[:160], weight=0.75)

            for table_ref in self._table_refs(text):
                table_id = table_lookup.get(self._normalize_marker(table_ref))
                if table_id is None and _FOLLOWING_TABLE_RE.fullmatch(table_ref):
                    table_id = self._find_following_table(node_id)
                if table_id is None or table_id == node_id:
                    continue
                evidence = f"{anchor.section_id or node_id} references {table_ref}"
                add(node_id, table_id, "refers_to", evidence, weight=0.9)
                add(table_id, node_id, "parameter_of", evidence, weight=0.9)

            for appendix_ref in self._appendix_refs(text):
                appendix_id = appendix_lookup.get(self._normalize_marker(appendix_ref))
                if appendix_id is None or appendix_id == node_id:
                    continue
                evidence = f"{anchor.section_id or node_id} references {appendix_ref}"
                add(node_id, appendix_id, "refers_to", evidence, weight=0.9)

        return relations

    def _build_table_lookup(self) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        for node_id, anchor in self.anchors.items():
            if anchor.node_type != "Table":
                continue
            for text in (anchor.section_id, anchor.text):
                for marker in _EXPLICIT_TABLE_RE.findall(text or ""):
                    lookup[self._normalize_marker(marker)] = node_id
        return lookup

    def _build_appendix_lookup(self) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        for node_id, anchor in self.anchors.items():
            if anchor.node_type != "Appendix":
                continue
            for text in (anchor.section_id, anchor.text):
                for marker in _APPENDIX_RE.findall(text or ""):
                    lookup[self._normalize_marker(marker)] = node_id
        return lookup

    def _find_following_table(self, node_id: int) -> Optional[int]:
        source = self.anchors.get(node_id)
        if source is None:
            return None
        candidates = [
            anchor.node_id
            for anchor in self.anchors.values()
            if anchor.anchor_kind == "tree"
            and anchor.node_type == "Table"
            and anchor.parent_id == source.parent_id
            and anchor.node_id > node_id
        ]
        return min(candidates) if candidates else None

    @staticmethod
    def _normalize_marker(text: str) -> str:
        return re.sub(r"\s+", "", text or "").replace("－", "-").replace("—", "-").replace("–", "-").replace("．", ".")

    @staticmethod
    def infer_node_type(node: TreeNode) -> HydroNodeType:
        text = HRIIndex._node_text(node)
        head = text[:120]
        if node.type == NodeType.TABLE or _EXPLICIT_TABLE_RE.match(head.strip()):
            return "Table"
        if _APPENDIX_RE.search(head[:40]):
            return "Appendix"
        if _DEFINITION_RE.search(text):
            return "TermDefinition"
        if _ARTICLE_RE.search(head):
            return "Article"
        if node.type == NodeType.TITLE or _CHAPTER_RE.search(head):
            return "Chapter"
        return "Article"

    @staticmethod
    def _node_text(node: TreeNode) -> str:
        parts = []
        meta = node.meta_info
        for value in (meta.content, meta.caption, meta.footnote, meta.table_body):
            if value:
                parts.append(str(value))
        return "\n".join(dict.fromkeys(parts)).strip()

    @staticmethod
    def _human_page(node: TreeNode) -> Optional[int]:
        page_idx = node.meta_info.page_idx
        if page_idx is None or page_idx < 0:
            return None
        return int(page_idx) + 1

    @staticmethod
    def _title_path(tree: DocumentTree, node: TreeNode) -> List[str]:
        path = tree.get_path_from_root(node.index_id)
        titles = []
        for item in path:
            content = item.meta_info.content
            if content:
                titles.append(content)
        return titles

    @staticmethod
    def _section_id(node: TreeNode, node_type: HydroNodeType) -> str:
        text = HRIIndex._node_text(node)
        if node_type == "Table":
            match = _EXPLICIT_TABLE_RE.search(text)
        elif node_type == "Appendix":
            match = _APPENDIX_RE.search(text)
        elif node_type == "Chapter":
            match = _CHAPTER_RE.search(text)
        elif node_type == "Article":
            match = _ARTICLE_RE.search(text)
        else:
            match = None
        if match:
            return match.group(1).strip()
        if node_type == "TermDefinition":
            return re.split(r"是指|系指|指的是|定义为|称为", text, maxsplit=1)[0].strip(" ：:，,。")
        return text.splitlines()[0][:80] if text else str(node.index_id)

    @staticmethod
    def _first_sentence(text: str) -> str:
        return re.split(r"[。；;\n]", text, maxsplit=1)[0][:160]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [
            sentence.strip(" \t\r\n，,。；;")
            for sentence in re.split(r"[。；;\n]", text or "")
            if sentence.strip(" \t\r\n，,。；;")
        ]

    @staticmethod
    def _definition_term(text: str) -> str:
        term = re.split(r"是指|系指|指的是|定义为|称为", text, maxsplit=1)[0]
        return HRIIndex._clean_leading_marker(term)

    @staticmethod
    def _definition_text(text: str) -> str:
        parts = re.split(r"是指|系指|指的是|定义为|称为", text, maxsplit=1)
        if len(parts) < 2:
            return ""
        return parts[1].strip(" ：:，,。；;\t\r\n")

    @staticmethod
    def _parse_requirement(sentence: str) -> Optional[Dict[str, str]]:
        if HRIIndex._is_policy_background_sentence(sentence):
            return None

        candidates: List[tuple[int, str]] = []
        for trigger in ("应当", "必须", "不得", "严禁", "宜", "须"):
            start = sentence.find(trigger)
            if start >= 0:
                if trigger == "须" and start > 0 and sentence[start - 1] == "必":
                    continue
                candidates.append((start, trigger))

        for match in re.finditer("应", sentence):
            start = match.start()
            prev_char = sentence[start - 1] if start > 0 else ""
            following = sentence[start : start + 4]
            if prev_char in {"响", "适"} or following.startswith(("应用", "应急", "应对", "应付")):
                continue
            candidates.append((start, "应"))

        for match in re.finditer("要", sentence):
            start = match.start()
            prev_char = sentence[start - 1] if start > 0 else ""
            following = sentence[start : start + 12]
            if prev_char in {"纲", "主", "重", "必", "概", "摘", "需"}:
                continue
            if following.startswith(("要求", "要素", "要点")):
                continue
            body = sentence[start + 1 : start + 12]
            if not any(word in body for word in _ACTION_WORDS):
                continue
            candidates.append((start, "要"))

        if not candidates:
            return None

        start, trigger = min(candidates, key=lambda item: item[0])
        requirement_text = HRIIndex._requirement_text(sentence, start)
        if not HRIIndex._is_valid_requirement_text(requirement_text, trigger):
            return None

        body = requirement_text[len(trigger) :].strip(" ：:，,。；;\t\r\n")
        action = HRIIndex._first_action_word(body)
        object_text = body[len(action) :].strip(" ：:，,。；;\t\r\n") if action else body
        subject = HRIIndex._subject_before_trigger(sentence[:start])
        return {
            "text": requirement_text,
            "subject": subject,
            "trigger": trigger,
            "action": action,
            "object": object_text,
        }

    @staticmethod
    def _is_policy_background_sentence(sentence: str) -> bool:
        return bool(_POLICY_BACKGROUND_RE.search(sentence))

    @staticmethod
    def _is_valid_requirement_text(text: str, trigger: str) -> bool:
        if not text:
            return False
        if any(phrase in text[:20] for phrase in _NOMINAL_REQUIREMENT_PHRASES):
            return False
        body = text[len(trigger) :].strip(" ：:，,。；;\t\r\n")
        if not body:
            return False
        if trigger == "要" and not any(word in body[:12] for word in _ACTION_WORDS):
            return False
        return True

    @staticmethod
    def _first_action_word(text: str) -> str:
        for word in sorted(_ACTION_WORDS, key=len, reverse=True):
            if text.startswith(word):
                return word
        for word in sorted(_ACTION_WORDS, key=len, reverse=True):
            idx = text.find(word)
            if 0 <= idx <= 4:
                return word
        match = re.match(r"[\u4e00-\u9fff]{1,4}", text or "")
        return match.group(0) if match else ""

    @staticmethod
    def _subject_before_trigger(text: str) -> str:
        chunks = re.split(r"[。；;，,：:]", text or "")
        return HRIIndex._clean_leading_marker(chunks[-1] if chunks else "")

    @staticmethod
    def _condition_text(sentence: str) -> str:
        patterns = [
            r"当(.+?)时",
            r"在(.+?)情况下",
            r"若(.+?)(?:，|,|则|应|要|必须|须|不得|宜)",
            r"如果(.+?)(?:，|,|则|应|要|必须|须|不得|宜)",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                return HRIIndex._clean_leading_marker(match.group(1))
        return ""

    @staticmethod
    def _exception_text(sentence: str) -> str:
        match = re.search(r"除(.+?)外", sentence)
        if match:
            return HRIIndex._clean_leading_marker(match.group(1))
        match = re.search(r"(特殊情况下|例外|但是.+|但.+)", sentence)
        if match:
            return HRIIndex._clean_leading_marker(match.group(1))
        return ""

    @staticmethod
    def _supplement_text(sentence: str) -> str:
        match = re.search(r"(补充说明.+|附录说明.+|同时.+|另外.+|此外.+|另.+|还应.+)", sentence)
        if match:
            return HRIIndex._clean_leading_marker(match.group(1))
        return ""

    @staticmethod
    def _requirement_text(sentence: str, start_idx: int) -> str:
        text = sentence[start_idx:].strip(" ，,。；;")
        return HRIIndex._clean_leading_marker(text)

    @staticmethod
    def _clean_leading_marker(text: str) -> str:
        text = re.sub(r"^\s*[（(]?[一二三四五六七八九十\d]+[）).、]\s*", "", text or "")
        text = re.sub(r"^\s*第\s*[一二三四五六七八九十百千万\d.．]+\s*[章节条]\s*", "", text)
        return text.strip(" ：:，,。；;\t\r\n")

    @staticmethod
    def _normalize_semantic_text(text: str) -> str:
        return re.sub(r"\s+", "", text or "").strip("：:，,。；;")

    @staticmethod
    def _table_refs(text: str) -> List[str]:
        refs = list(_EXPLICIT_TABLE_RE.findall(text or ""))
        if _FOLLOWING_TABLE_RE.search(text or ""):
            refs.append("下表")
        return refs

    @staticmethod
    def _appendix_refs(text: str) -> List[str]:
        return list(_APPENDIX_RE.findall(text or ""))

    @staticmethod
    def _bm25_document(anchor: EvidenceAnchor) -> str:
        return "\n".join(
            [
                anchor.node_type,
                anchor.section_id,
                " > ".join(anchor.title_path),
                anchor.text,
            ]
        )
