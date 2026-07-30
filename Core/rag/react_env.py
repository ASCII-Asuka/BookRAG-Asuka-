from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReactEvidenceUnit:
    block_id: Any
    page_key: str
    content: str
    metadata: Dict[str, Any]
    corpus_index: int


@dataclass
class ReactObservation:
    text: str
    evidence: List[ReactEvidenceUnit] = field(default_factory=list)
    valid_action: bool = True
    done: bool = False
    answer: str = ""
    action_type: str = ""


class ReactLocalEnvironment:
    def __init__(self, docs, bm25, page_observation_units=5, search_topk=1):
        self.docs = list(docs)
        self.bm25 = bm25
        self.page_observation_units = max(1, int(page_observation_units))
        self.search_topk = max(1, int(search_topk))
        self.pages: Dict[str, List[ReactEvidenceUnit]] = {}
        self.page_keys: List[str] = []
        self.active_page: Optional[str] = None
        self.lookup_keyword: Optional[str] = None
        self.lookup_results: List[ReactEvidenceUnit] = []
        self.lookup_index = 0
        self.steps = 0
        self.answer: Optional[str] = None
        self._build_pages()

    @staticmethod
    def _block_id(doc, fallback):
        meta = doc.get("metadata") or {}
        return meta.get(
            "source_node_id",
            meta.get("node_id", doc.get("id", fallback)),
        )

    @staticmethod
    def _page_key(doc, fallback):
        meta = doc.get("metadata") or {}
        for key in ("hotpot_title", "section_id", "section", "title_path"):
            value = str(meta.get(key) or "").strip()
            if value:
                return value
        return f"evidence-{fallback}"

    def _build_pages(self):
        for index, doc in enumerate(self.docs):
            key = self._page_key(doc, index)
            if key not in self.pages:
                self.pages[key] = []
                self.page_keys.append(key)
            self.pages[key].append(
                ReactEvidenceUnit(
                    block_id=self._block_id(doc, index),
                    page_key=key,
                    content=str(doc.get("content") or ""),
                    metadata=dict(doc.get("metadata") or {}),
                    corpus_index=index,
                )
            )

    @staticmethod
    def _normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

    def _find_page(self, query):
        normalized = self._normalize(query)
        exact = {self._normalize(key): key for key in self.page_keys}
        if normalized in exact:
            return exact[normalized]
        containing = [
            key
            for key in self.page_keys
            if normalized
            and (
                normalized in self._normalize(key)
                or self._normalize(key) in normalized
            )
        ]
        if containing:
            return min(
                containing,
                key=lambda key: (len(key), self.page_keys.index(key)),
            )
        if self.bm25 is not None:
            hits = self.bm25.search(
                query_text=query,
                top_k=self.search_topk,
            )
            if hits:
                return self._page_key(hits[0], hits[0].get("id", 0))
        return None

    def _search(self, query):
        page_key = self._find_page(query)
        if page_key not in self.pages:
            return ReactObservation(
                text=f"Could not find {query}.",
                valid_action=False,
                action_type="search",
            )
        self.active_page = page_key
        self.lookup_keyword = None
        self.lookup_results = []
        self.lookup_index = 0
        evidence = self.pages[page_key][: self.page_observation_units]
        text = " ".join(
            item.content.strip()
            for item in evidence
            if item.content.strip()
        )
        return ReactObservation(
            text=text,
            evidence=evidence,
            action_type="search",
        )

    @staticmethod
    def _sentences(unit):
        parts = re.split(r"(?<=[.!?])\s+", unit.content.strip())
        return [
            ReactEvidenceUnit(
                block_id=unit.block_id,
                page_key=unit.page_key,
                content=part.strip(),
                metadata=unit.metadata,
                corpus_index=unit.corpus_index,
            )
            for part in parts
            if part.strip()
        ]

    def _lookup(self, keyword):
        if self.active_page is None:
            return ReactObservation(
                text="No active page.",
                valid_action=False,
                action_type="lookup",
            )
        if self.lookup_keyword != keyword:
            self.lookup_keyword = keyword
            self.lookup_results = [
                sentence
                for unit in self.pages[self.active_page]
                for sentence in self._sentences(unit)
                if keyword.lower() in sentence.content.lower()
            ]
            self.lookup_index = 0
        if self.lookup_index >= len(self.lookup_results):
            return ReactObservation(
                text="No more results.",
                action_type="lookup",
            )
        item = self.lookup_results[self.lookup_index]
        self.lookup_index += 1
        return ReactObservation(
            text=(
                f"(Result {self.lookup_index} / {len(self.lookup_results)}) "
                f"{item.content}"
            ),
            evidence=[item],
            action_type="lookup",
        )

    def step(self, action):
        action = str(action or "").strip()
        self.steps += 1
        match = re.fullmatch(r"(?i:search)\[(.*)\]", action, re.DOTALL)
        if match:
            return self._search(match.group(1).strip())
        match = re.fullmatch(r"(?i:lookup)\[(.*)\]", action, re.DOTALL)
        if match:
            return self._lookup(match.group(1).strip())
        match = re.fullmatch(r"(?i:finish)\[(.*)\]", action, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            self.answer = answer
            return ReactObservation(
                text="Episode finished.",
                done=True,
                answer=answer,
                action_type="finish",
            )
        return ReactObservation(
            text=f"Invalid action: {action}",
            valid_action=False,
            action_type="invalid",
        )
