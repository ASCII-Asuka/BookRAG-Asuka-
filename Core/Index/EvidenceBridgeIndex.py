import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

import networkx as nx
from pydantic import BaseModel, Field

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.utils.bm25 import BM25

# 定义了证据块的枚举类型
EvidenceBlockType = Literal[
    "paragraph",
    "table",
    "figure",
    "caption",
    "title",
    "summary",
    "entity",
    "patch",
    "equation",
    "unknown",
]
BridgeType = Literal["context", "semantic", "hierarchy"]

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z][a-zA-Z0-9_-]*|\d+(?:\.\d+)?")
_TABLE_REF_RE = re.compile(
    r"\b(?:table|tab\.)\s*[A-Za-z0-9]+(?:[-.]\d+)*\b|"
    r"表\s*[A-Za-z0-9一二三四五六七八九十百千万]+(?:[-.－—]\d+)*",
    re.IGNORECASE,
)
_FIGURE_REF_RE = re.compile(
    r"\b(?:figure|fig\.)\s*[A-Za-z0-9]+(?:[-.]\d+)*\b|"
    r"图\s*[A-Za-z0-9一二三四五六七八九十百千万]+(?:[-.－—]\d+)*",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about",
    "above",
    "after",
    "before",
    "between",
    "checked",
    "during",
    "from",
    "into",
    "must",
    "should",
    "table",
    "that",
    "their",
    "there",
    "this",
    "uses",
    "with",
}


def evidence_tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_RE.findall((text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(token)
            tokens.extend(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
        elif len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    return [token for token in tokens if token.strip()]


class EvidenceBM25(BM25):
    def _tokenize(self, text: str) -> List[str]:
        return evidence_tokenize(text)


class EvidenceBlock(BaseModel):
    block_id: int
    block_type: EvidenceBlockType
    text: str = ""
    source_node_id: Optional[int] = None
    doc_name: Optional[str] = None
    doc_path: Optional[str] = None
    section_id: str = ""
    page: Optional[int] = None
    title_path: List[str] = Field(default_factory=list)
    parent_id: Optional[int] = None
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBridge(BaseModel):
    source_id: int
    target_id: int
    bridge_type: BridgeType
    relation_type: str
    evidence: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBridgeIndex:
    _INDEX_FILE = "evibridge_index.json"
    _BM25_FILE = "evibridge_bm25.pkl"

    def __init__(
        self,
        save_dir: str,
        blocks: Optional[Dict[int, EvidenceBlock]] = None,
        bridges: Optional[List[EvidenceBridge]] = None,
        bm25_block_ids: Optional[List[int]] = None,
    ):
        self.save_dir = save_dir
        self.blocks: Dict[int, EvidenceBlock] = blocks or {}
        self.bridges: List[EvidenceBridge] = bridges or []
        self.bm25_block_ids: List[int] = bm25_block_ids or []
        self._out_bridges: Dict[int, List[EvidenceBridge]] = defaultdict(list)
        self._in_bridges: Dict[int, List[EvidenceBridge]] = defaultdict(list)
        self._rebuild_bridge_maps()

    @classmethod
    def from_tree(cls, tree: DocumentTree, save_dir: str) -> "EvidenceBridgeIndex":
        index = cls(save_dir=save_dir)
        for node in tree.get_nodes(hasRoot=False):
            text = cls._node_text(node)
            if not text:
                continue
            block = EvidenceBlock(
                block_id=node.index_id,
                source_node_id=node.index_id,
                doc_name=tree.meta_info.file_name,
                doc_path=tree.meta_info.file_path,
                block_type=cls._block_type(node),
                text=text,
                section_id=cls._section_id(node),
                page=cls._human_page(node),
                title_path=cls._title_path(node),
                parent_id=node.parent.index_id if node.parent and node.parent.index_id != 0 else None,
                metadata=cls._metadata(node),
            )
            index.blocks[block.block_id] = block

        index._add_generated_blocks(tree)
        index.bridges = index._build_bridges(tree)
        index._rebuild_bridge_maps()
        return index

    @staticmethod
    def get_index_path(save_dir: str) -> str:
        return os.path.join(save_dir, EvidenceBridgeIndex._INDEX_FILE)

    @staticmethod
    def get_bm25_path(save_dir: str) -> str:
        return os.path.join(save_dir, EvidenceBridgeIndex._BM25_FILE)

    @classmethod
    def load_from_dir(cls, save_dir: str) -> "EvidenceBridgeIndex":
        with open(cls.get_index_path(save_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        blocks = {
            int(block["block_id"]): EvidenceBlock(**block)
            for block in data.get("blocks", [])
        }
        bridges = [EvidenceBridge(**bridge) for bridge in data.get("bridges", [])]
        return cls(
            save_dir=save_dir,
            blocks=blocks,
            bridges=bridges,
            bm25_block_ids=[int(block_id) for block_id in data.get("bm25_block_ids", [])],
        )

    def save_to_dir(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.get_index_path(self.save_dir), "w", encoding="utf-8") as f:
            json.dump(self.to_json_data(), f, ensure_ascii=False, indent=2)

    def to_json_data(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "blocks": [
                block.model_dump()
                for _, block in sorted(self.blocks.items(), key=lambda item: item[0])
            ],
            "bridges": [bridge.model_dump() for bridge in self.bridges],
            "bm25_block_ids": self.bm25_block_ids,
        }

    def build_bm25(self) -> EvidenceBM25:
        block_ids = sorted(self.blocks)
        corpus = [self._retrieval_document(self.blocks[block_id]) for block_id in block_ids]
        bm25 = EvidenceBM25(corpus)
        bm25.initialize()
        self.bm25_block_ids = block_ids
        return bm25

    def save_bm25(self, bm25: EvidenceBM25) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.get_bm25_path(self.save_dir), "wb") as f:
            pickle.dump(bm25, f)

    @classmethod
    def load_bm25(cls, save_dir: str) -> EvidenceBM25:
        with open(cls.get_bm25_path(save_dir), "rb") as f:
            return pickle.load(f)

    def search_bm25(self, bm25: EvidenceBM25, query: str, top_k: int) -> List[Dict[str, Any]]:
        results = bm25.search(query_text=query, top_k=top_k)
        if not self.bm25_block_ids:
            self.bm25_block_ids = sorted(self.blocks)
        mapped_results: List[Dict[str, Any]] = []
        for item in results:
            doc_idx = int(item["id"])
            if doc_idx >= len(self.bm25_block_ids):
                continue
            block_id = self.bm25_block_ids[doc_idx]
            block = self.blocks.get(block_id)
            if block is None:
                continue
            bridge_types = sorted(
                {
                    bridge.bridge_type
                    for bridge in self._out_bridges.get(block_id, [])
                    + self._in_bridges.get(block_id, [])
                }
            )
            mapped_results.append(
                {
                    "block_id": block_id,
                    "score": item.get("score", 0.0),
                    "block_type": block.block_type,
                    "section_id": block.section_id,
                    "page": block.page,
                    "text": block.text,
                    "bridge_types": bridge_types,
                    "block": block.model_dump(),
                }
            )
        return mapped_results

    def iter_vector_documents(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for block_id, block in sorted(self.blocks.items(), key=lambda item: item[0]):
            documents.append(
                {
                    "block_id": block_id,
                    "text": self._retrieval_document(block),
                    "metadata": {
                        "block_id": block_id,
                        "source_node_id": block.source_node_id,
                        "block_type": block.block_type,
                        "section_id": block.section_id,
                        "page": block.page if block.page is not None else -1,
                    },
                }
            )
        return documents

    def to_typed_graph(self, bridge_types: Optional[Iterable[str]] = None) -> nx.MultiDiGraph:
        allowed = set(bridge_types) if bridge_types else None
        graph = nx.MultiDiGraph()
        for block_id, block in self.blocks.items():
            graph.add_node(block_id, **block.model_dump())
        for bridge in self.bridges:
            if allowed and bridge.bridge_type not in allowed:
                continue
            graph.add_edge(
                bridge.source_id,
                bridge.target_id,
                bridge_type=bridge.bridge_type,
                relation_type=bridge.relation_type,
                weight=bridge.weight,
                evidence=bridge.evidence,
                metadata=bridge.metadata,
            )
        return graph

    def get_related_bridges(
        self,
        block_ids: Iterable[int],
        expand_depth: int = 1,
        bridge_types: Optional[Iterable[str]] = None,
    ) -> List[EvidenceBridge]:
        allowed = set(bridge_types) if bridge_types else None
        frontier = set(block_ids)
        visited = set(frontier)
        selected: List[EvidenceBridge] = []
        seen = set()
        for _ in range(max(expand_depth, 0)):
            next_frontier = set()
            for block_id in frontier:
                for bridge in self._out_bridges.get(block_id, []) + self._in_bridges.get(block_id, []):
                    if allowed and bridge.bridge_type not in allowed:
                        continue
                    key = (bridge.source_id, bridge.target_id, bridge.bridge_type, bridge.relation_type)
                    if key not in seen:
                        selected.append(bridge)
                        seen.add(key)
                    for related_id in (bridge.source_id, bridge.target_id):
                        if related_id not in visited:
                            next_frontier.add(related_id)
                            visited.add(related_id)
            frontier = next_frontier
            if not frontier:
                break
        return selected

    def get_block(self, block_id: int) -> Optional[EvidenceBlock]:
        return self.blocks.get(block_id)

    def _build_bridges(self, tree: DocumentTree) -> List[EvidenceBridge]:
        bridges: List[EvidenceBridge] = []
        seen = set()

        def add(
            source_id: int,
            target_id: int,
            bridge_type: BridgeType,
            relation_type: str,
            evidence: str = "",
            weight: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            if source_id not in self.blocks or target_id not in self.blocks or source_id == target_id:
                return
            key = (source_id, target_id, bridge_type, relation_type)
            if key in seen:
                return
            seen.add(key)
            bridges.append(
                EvidenceBridge(
                    source_id=source_id,
                    target_id=target_id,
                    bridge_type=bridge_type,
                    relation_type=relation_type,
                    evidence=evidence,
                    weight=weight,
                    metadata=metadata or {},
                )
            )

        for block in list(self.blocks.values()):
            if block.parent_id in self.blocks:
                add(block.parent_id, block.block_id, "context", "parent_child", weight=0.8)
                add(block.block_id, block.parent_id, "context", "child_parent", weight=0.8)
                add(block.block_id, block.parent_id, "hierarchy", "belongs_to_section", weight=0.9)
                add(block.parent_id, block.block_id, "hierarchy", "section_contains", weight=0.9)

        self._add_generated_bridges(add)
        self._add_same_section_bridges(add)

        for node in tree.get_nodes(hasRoot=True):
            child_ids = [child.index_id for child in node.children if child.index_id in self.blocks]
            for left_id, right_id in zip(child_ids, child_ids[1:]):
                add(left_id, right_id, "context", "next", weight=0.75)
                add(right_id, left_id, "context", "previous", weight=0.75)

        table_lookup = self._build_marker_lookup("table")
        figure_lookup = self._build_marker_lookup("figure")
        for block in list(self.blocks.values()):
            for marker in self._reference_markers(block.text, _TABLE_REF_RE):
                target_id = table_lookup.get(marker)
                if target_id is not None:
                    add(block.block_id, target_id, "context", "reference_to_table", marker, weight=0.95)
                    add(target_id, block.block_id, "context", "referenced_by", marker, weight=0.7)
            for marker in self._reference_markers(block.text, _FIGURE_REF_RE):
                target_id = figure_lookup.get(marker)
                if target_id is not None:
                    add(block.block_id, target_id, "context", "reference_to_figure", marker, weight=0.95)
                    add(target_id, block.block_id, "context", "referenced_by", marker, weight=0.7)

        self._add_semantic_bridges(add)
        return bridges

    def _add_generated_blocks(self, tree: DocumentTree) -> None:
        next_id = max(self.blocks.keys(), default=0) + 1

        def allocate_id() -> int:
            nonlocal next_id
            while next_id in self.blocks:
                next_id += 1
            block_id = next_id
            next_id += 1
            return block_id

        source_to_caption: Dict[int, int] = {}
        for node in tree.get_nodes(hasRoot=False):
            source_block = self.blocks.get(node.index_id)
            caption = (getattr(node.meta_info, "caption", None) or "").strip()
            if not source_block or not caption:
                continue
            caption_id = allocate_id()
            self.blocks[caption_id] = EvidenceBlock(
                block_id=caption_id,
                block_type="caption",
                text=caption,
                source_node_id=node.index_id,
                doc_name=source_block.doc_name,
                doc_path=source_block.doc_path,
                section_id=source_block.section_id,
                page=source_block.page,
                title_path=source_block.title_path.copy(),
                parent_id=source_block.block_id,
                metadata={
                    "generated": True,
                    "generated_type": "caption",
                    "caption_of": source_block.block_id,
                },
            )
            source_to_caption[node.index_id] = caption_id

        document_summary = self._document_summary_text(tree)
        if document_summary:
            summary_id = allocate_id()
            self.blocks[summary_id] = EvidenceBlock(
                block_id=summary_id,
                block_type="summary",
                text=document_summary,
                source_node_id=None,
                doc_name=tree.meta_info.file_name,
                doc_path=tree.meta_info.file_path,
                section_id="document",
                page=None,
                title_path=["Document Summary"],
                parent_id=None,
                metadata={
                    "generated": True,
                    "generated_type": "document_summary",
                    "summary_scope": "document",
                },
            )

        for node in tree.get_nodes(hasRoot=False):
            if node.type != NodeType.TITLE or node.index_id not in self.blocks:
                continue
            section_block = self.blocks[node.index_id]
            descendants = self._descendant_block_ids(node)
            child_texts = [
                self.blocks[block_id].text
                for block_id in descendants
                if block_id in self.blocks and block_id != node.index_id
            ]
            summary_text = self._section_summary_text(section_block, child_texts)
            if summary_text:
                summary_id = allocate_id()
                self.blocks[summary_id] = EvidenceBlock(
                    block_id=summary_id,
                    block_type="summary",
                    text=summary_text,
                    source_node_id=node.index_id,
                    doc_name=section_block.doc_name,
                    doc_path=section_block.doc_path,
                    section_id=section_block.section_id,
                    page=section_block.page,
                    title_path=section_block.title_path + [section_block.text.splitlines()[0][:80]],
                    parent_id=section_block.block_id,
                    metadata={
                        "generated": True,
                        "generated_type": "section_summary",
                        "summary_of": section_block.block_id,
                    },
                )
            patch_text = self._section_patch_text(section_block, child_texts)
            if patch_text:
                patch_id = allocate_id()
                self.blocks[patch_id] = EvidenceBlock(
                    block_id=patch_id,
                    block_type="patch",
                    text=patch_text,
                    source_node_id=node.index_id,
                    doc_name=section_block.doc_name,
                    doc_path=section_block.doc_path,
                    section_id=section_block.section_id,
                    page=section_block.page,
                    title_path=section_block.title_path + [section_block.text.splitlines()[0][:80]],
                    parent_id=section_block.block_id,
                    metadata={
                        "generated": True,
                        "generated_type": "section_patch",
                        "patch_of": section_block.block_id,
                        "contains_block_ids": descendants[:80],
                    },
                )

        self._add_entity_blocks(allocate_id)

    def _add_entity_blocks(self, allocate_id) -> None:
        entity_to_block_id: Dict[str, int] = {}
        source_mentions: Dict[int, List[str]] = {}
        source_blocks = [
            block
            for block in self.blocks.values()
            if block.block_type not in {"entity", "summary", "patch"}
        ]
        for block in source_blocks:
            entities = self._extract_entities(block.text)
            if not entities:
                continue
            source_mentions[block.block_id] = entities
            for entity in entities:
                if entity in entity_to_block_id:
                    continue
                entity_id = allocate_id()
                entity_to_block_id[entity] = entity_id
                self.blocks[entity_id] = EvidenceBlock(
                    block_id=entity_id,
                    block_type="entity",
                    text=entity,
                    source_node_id=None,
                    doc_name=block.doc_name,
                    doc_path=block.doc_path,
                    section_id="entity",
                    page=None,
                    title_path=["Entities"],
                    parent_id=None,
                    metadata={
                        "generated": True,
                        "generated_type": "entity",
                        "entity": entity,
                        "mentioned_by": [],
                    },
                )
                if entity_id in self.blocks:
                    self.blocks[entity_id].metadata["mentioned_by"].append(block.block_id)
            for entity in entities:
                entity_id = entity_to_block_id.get(entity)
                if entity_id in self.blocks:
                    mentions = self.blocks[entity_id].metadata.setdefault("mentioned_by", [])
                    if block.block_id not in mentions:
                        mentions.append(block.block_id)

        for block_id, entities in source_mentions.items():
            self.blocks[block_id].metadata["mentioned_entities"] = [
                {"entity": entity, "block_id": entity_to_block_id[entity]}
                for entity in entities
                if entity in entity_to_block_id
            ]

    def _add_generated_bridges(self, add) -> None:
        entity_mentions: Dict[int, List[int]] = defaultdict(list)
        for block in list(self.blocks.values()):
            generated_type = block.metadata.get("generated_type")
            if generated_type == "caption":
                source_id = block.metadata.get("caption_of")
                if isinstance(source_id, int):
                    add(source_id, block.block_id, "context", "has_caption", weight=0.9)
                    add(block.block_id, source_id, "context", "caption_of", weight=0.95)
            elif generated_type in {"document_summary", "section_summary"}:
                source_id = block.metadata.get("summary_of") or block.parent_id
                if isinstance(source_id, int) and source_id in self.blocks:
                    add(source_id, block.block_id, "hierarchy", "has_summary", weight=0.85)
                    add(block.block_id, source_id, "hierarchy", "summary_of", weight=0.95)
            elif generated_type == "section_patch":
                source_id = block.metadata.get("patch_of") or block.parent_id
                if isinstance(source_id, int) and source_id in self.blocks:
                    add(source_id, block.block_id, "hierarchy", "has_patch", weight=0.8)
                    add(block.block_id, source_id, "hierarchy", "patch_of", weight=0.9)
                for child_id in block.metadata.get("contains_block_ids", []):
                    if child_id in self.blocks and child_id != block.block_id:
                        add(block.block_id, child_id, "context", "patch_contains", weight=0.72)
                        add(child_id, block.block_id, "context", "contained_in_patch", weight=0.65)

            for mention in block.metadata.get("mentioned_entities", []):
                entity_id = mention.get("block_id")
                if isinstance(entity_id, int) and entity_id in self.blocks:
                    add(block.block_id, entity_id, "semantic", "mentions_entity", mention.get("entity", ""), weight=0.78)
                    add(entity_id, block.block_id, "semantic", "mentioned_by", mention.get("entity", ""), weight=0.6)
                    entity_mentions[block.block_id].append(entity_id)

        for entity_ids in entity_mentions.values():
            unique_ids = sorted(set(entity_ids))
            for i, source_id in enumerate(unique_ids):
                for target_id in unique_ids[i + 1 :]:
                    source = self.blocks[source_id].metadata.get("entity", self.blocks[source_id].text)
                    target = self.blocks[target_id].metadata.get("entity", self.blocks[target_id].text)
                    evidence = f"{source}, {target}"
                    add(source_id, target_id, "semantic", "entity_cooccur", evidence, weight=0.55)
                    add(target_id, source_id, "semantic", "entity_cooccur", evidence, weight=0.55)

    def _add_same_section_bridges(self, add) -> None:
        section_groups: Dict[str, List[int]] = defaultdict(list)
        for block in self.blocks.values():
            if block.block_type in {"entity", "summary", "patch"}:
                continue
            section_groups[block.section_id].append(block.block_id)
        for section_id, block_ids in section_groups.items():
            if not section_id or len(block_ids) < 2 or len(block_ids) > 40:
                continue
            ordered_ids = sorted(block_ids)
            for i, source_id in enumerate(ordered_ids):
                for target_id in ordered_ids[i + 1 :]:
                    add(source_id, target_id, "context", "same_section", section_id, weight=0.35)
                    add(target_id, source_id, "context", "same_section", section_id, weight=0.35)

    def _add_semantic_bridges(self, add) -> None:
        token_to_blocks: Dict[str, Set[int]] = defaultdict(set)
        block_terms: Dict[int, Set[str]] = {}
        for block_id, block in self.blocks.items():
            text = " ".join([block.section_id, " ".join(block.title_path), block.text])
            terms = {token for token in evidence_tokenize(text) if len(token) >= 4}
            block_terms[block_id] = terms
            for term in terms:
                token_to_blocks[term].add(block_id)

        pair_terms: Dict[tuple[int, int], Set[str]] = defaultdict(set)
        for term, block_ids in token_to_blocks.items():
            if len(block_ids) < 2 or len(block_ids) > 20:
                continue
            sorted_ids = sorted(block_ids)
            for i, source_id in enumerate(sorted_ids):
                for target_id in sorted_ids[i + 1 :]:
                    pair_terms[(source_id, target_id)].add(term)

        for (source_id, target_id), terms in pair_terms.items():
            source_terms = block_terms.get(source_id, set())
            target_terms = block_terms.get(target_id, set())
            overlap = len(terms) / max(len(source_terms | target_terms), 1)
            if len(terms) < 2 and overlap < 0.12:
                continue
            evidence = ", ".join(sorted(terms)[:8])
            weight = min(0.85, 0.35 + overlap)
            add(source_id, target_id, "semantic", "shared_terms", evidence, weight=weight)
            add(target_id, source_id, "semantic", "shared_terms", evidence, weight=weight)

    def _descendant_block_ids(self, node: TreeNode) -> List[int]:
        block_ids: List[int] = []
        stack = list(node.children)
        while stack:
            child = stack.pop(0)
            if child.index_id in self.blocks:
                block_ids.append(child.index_id)
            stack.extend(child.children)
        return block_ids

    @staticmethod
    def _document_summary_text(tree: DocumentTree) -> str:
        title_texts = []
        for node in tree.get_nodes(hasRoot=False):
            if node.type == NodeType.TITLE and node.meta_info.content:
                title_texts.append(EvidenceBridgeIndex._clean_title(node.meta_info.content))
        title_texts = title_texts[:12]
        if not title_texts:
            return ""
        doc_name = tree.meta_info.file_name or "document"
        return f"Document summary for {doc_name}. Main sections: " + "; ".join(title_texts)

    @staticmethod
    def _section_summary_text(section_block: EvidenceBlock, child_texts: List[str]) -> str:
        section_title = section_block.text.splitlines()[0].strip()
        snippets = []
        for text in child_texts:
            clean = re.sub(r"\s+", " ", text).strip()
            if clean:
                snippets.append(clean[:220])
            if len(snippets) >= 4:
                break
        if not snippets:
            return ""
        return f"Section summary for {section_title}: " + " ".join(snippets)[:900]

    @staticmethod
    def _section_patch_text(section_block: EvidenceBlock, child_texts: List[str]) -> str:
        section_title = section_block.text.splitlines()[0].strip()
        merged = []
        for text in child_texts:
            clean = re.sub(r"\s+", " ", text).strip()
            if clean:
                merged.append(clean)
            if sum(len(item) for item in merged) > 1600:
                break
        if not merged:
            return ""
        return f"Evidence patch for {section_title}: " + " ".join(merged)[:1800]

    @staticmethod
    def _extract_entities(text: str, max_entities: int = 8) -> List[str]:
        raw_text = text or ""
        candidates: List[str] = []
        for match in re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3}\b", raw_text):
            normalized = match.strip().lower()
            if normalized and normalized not in _STOPWORDS:
                candidates.append(normalized)
        for token in evidence_tokenize(raw_text):
            if len(token) >= 5 and token not in _STOPWORDS:
                candidates.append(token)
        selected: List[str] = []
        seen = set()
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;()[]{}").lower()
            if not candidate or candidate in seen or len(candidate) < 3:
                continue
            seen.add(candidate)
            selected.append(candidate)
            if len(selected) >= max_entities:
                break
        return selected

    def _build_marker_lookup(self, block_type: EvidenceBlockType) -> Dict[str, int]:
        pattern = _TABLE_REF_RE if block_type == "table" else _FIGURE_REF_RE
        lookup: Dict[str, int] = {}
        for block in self.blocks.values():
            if block.block_type != block_type:
                continue
            for marker in self._reference_markers(block.text, pattern):
                lookup[marker] = block.block_id
        return lookup

    def _rebuild_bridge_maps(self) -> None:
        self._out_bridges = defaultdict(list)
        self._in_bridges = defaultdict(list)
        for bridge in self.bridges:
            self._out_bridges[bridge.source_id].append(bridge)
            self._in_bridges[bridge.target_id].append(bridge)

    @staticmethod
    def _reference_markers(text: str, pattern: re.Pattern) -> List[str]:
        markers = []
        for match in pattern.findall(text or ""):
            normalized = re.sub(r"[\s.。:：-]+", "", str(match).lower())
            normalized = normalized.replace("tab", "table").replace("fig", "figure")
            if normalized:
                markers.append(normalized)
        return markers

    @staticmethod
    def _block_type(node: TreeNode) -> EvidenceBlockType:
        node_type = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        if node_type == "text":
            return "paragraph"
        if node_type == "image":
            return "figure"
        if node_type in {"title", "table", "equation"}:
            return node_type
        return "unknown"

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
    def _title_path(node: TreeNode) -> List[str]:
        titles: List[str] = []
        cursor = node
        while cursor is not None and cursor.parent is not None:
            if cursor.type == NodeType.TITLE and cursor.meta_info.content:
                titles.append(EvidenceBridgeIndex._clean_title(cursor.meta_info.content))
            cursor = cursor.parent
        return list(reversed(titles))

    @staticmethod
    def _section_id(node: TreeNode) -> str:
        title_path = EvidenceBridgeIndex._title_path(node)
        if title_path:
            return title_path[-1]
        text = EvidenceBridgeIndex._node_text(node)
        return text.splitlines()[0][:120] if text else str(node.index_id)

    @staticmethod
    def _metadata(node: TreeNode) -> Dict[str, Any]:
        meta = node.meta_info
        metadata: Dict[str, Any] = {}
        if meta.pdf_id is not None:
            metadata["pdf_id"] = meta.pdf_id
        if meta.img_path:
            metadata["img_path"] = meta.img_path
        if meta.caption:
            metadata["caption"] = meta.caption
        if meta.footnote:
            metadata["footnote"] = meta.footnote
        if meta.table_body:
            metadata["has_table_body"] = True
        return metadata

    @staticmethod
    def _retrieval_document(block: EvidenceBlock) -> str:
        return "\n".join(
            part
            for part in [
                block.block_type,
                block.section_id,
                " > ".join(block.title_path),
                block.text,
            ]
            if part
        )

    @staticmethod
    def _clean_title(text: str) -> str:
        return re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", text or "").strip() or text
