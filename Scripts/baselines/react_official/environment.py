from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: str) -> list[str]:
    return normalize_text(value).split()


@dataclass(frozen=True)
class ObservationResult:
    action: str
    query: str
    observation: str
    evidence: list[dict[str, Any]]


@dataclass
class _Page:
    title: str
    aliases: set[str]
    atoms: list[dict[str, Any]]

    @property
    def search_text(self) -> str:
        return " ".join(
            [self.title] + [str(atom.get("content") or "") for atom in self.atoms]
        )


class _ClosedCorpusEnvironment:
    def __init__(self, pages: Sequence[_Page], max_observation_characters: int = 4000):
        if not pages:
            raise ValueError("closed-corpus environment requires at least one page")
        self.pages = list(pages)
        self.max_observation_characters = max(1, int(max_observation_characters))
        self.current_page: _Page | None = None
        self._lookup_key = ""
        self._lookup_position = 0

    def _matching_page(self, query: str) -> _Page:
        aliases = {_normalize_title(query), normalize_text(query)}
        exact = [page for page in self.pages if page.aliases.intersection(aliases)]
        if exact:
            return exact[0]
        return self._bm25_page(query)

    def _bm25_page(self, query: str) -> _Page:
        query_terms = _tokens(query)
        documents = [_tokens(page.search_text) for page in self.pages]
        average_length = sum(len(document) for document in documents) / max(
            1, len(documents)
        )
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        total = len(documents)

        def score(index: int) -> float:
            document = documents[index]
            counts = Counter(document)
            value = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency <= 0:
                    continue
                df = document_frequency.get(term, 0)
                inverse = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
                )
                value += inverse * frequency * 2.5 / denominator
            return value

        best_index = max(range(len(self.pages)), key=lambda index: (score(index), -index))
        return self.pages[best_index]

    def _render_search(self, page: _Page) -> tuple[str, list[dict[str, Any]]]:
        lines = [page.title]
        evidence: list[dict[str, Any]] = []
        used = len(page.title)
        for atom in page.atoms:
            content = str(atom.get("content") or "").strip()
            if not content:
                continue
            remaining = self.max_observation_characters - used - 1
            if remaining <= 0:
                break
            displayed = content[:remaining]
            lines.append(displayed)
            evidence.append({**atom, "action": "Search"})
            used += len(displayed) + 1
            if len(displayed) < len(content):
                break
        return "\n".join(lines), evidence

    def search(self, query: str) -> ObservationResult:
        page = self._matching_page(query)
        self.current_page = page
        self._lookup_key = ""
        self._lookup_position = 0
        observation, evidence = self._render_search(page)
        return ObservationResult("Search", str(query), observation, evidence)

    def lookup(self, keyword: str) -> ObservationResult:
        if self.current_page is None:
            return ObservationResult(
                "Lookup", str(keyword), "No page is currently selected.", []
            )
        normalized = normalize_text(keyword)
        if normalized != self._lookup_key:
            self._lookup_key = normalized
            self._lookup_position = 0
        matches = [
            atom
            for atom in self.current_page.atoms
            if normalized in normalize_text(str(atom.get("content") or ""))
        ]
        if self._lookup_position >= len(matches):
            return ObservationResult(
                "Lookup",
                str(keyword),
                f"No more results for {keyword}.",
                [],
            )
        atom = matches[self._lookup_position]
        self._lookup_position += 1
        observation = str(atom.get("content") or "")[: self.max_observation_characters]
        return ObservationResult(
            "Lookup", str(keyword), observation, [{**atom, "action": "Lookup"}]
        )

    def execute(self, action: str, argument: str) -> ObservationResult:
        if action == "Search":
            return self.search(argument)
        if action == "Lookup":
            return self.lookup(argument)
        raise ValueError(f"unsupported environment action: {action}")


class HotpotEnvironment(_ClosedCorpusEnvironment):
    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Mapping[str, Any]],
        max_observation_characters: int = 4000,
    ) -> "HotpotEnvironment":
        pages: list[_Page] = []
        seen_titles: set[str] = set()
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            title = str(metadata.get("hotpot_title") or metadata.get("title") or "")
            normalized_title = _normalize_title(title)
            if not title or normalized_title in seen_titles:
                raise ValueError(f"duplicate or empty HotpotQA title: {title!r}")
            seen_titles.add(normalized_title)
            atoms: list[dict[str, Any]] = []
            for sentence in metadata.get("hotpot_sentences") or []:
                if not isinstance(sentence, Mapping):
                    continue
                atoms.append(
                    {
                        "source_id": str(chunk.get("source_id") or title),
                        "hotpot_title": title,
                        "hotpot_sent_id": int(sentence.get("sent_id")),
                        "content": str(sentence.get("text") or ""),
                    }
                )
            if not atoms:
                raise ValueError(f"HotpotQA page has no legal sentences: {title!r}")
            pages.append(_Page(title, {normalized_title}, atoms))
        return cls(pages, max_observation_characters=max_observation_characters)


class QasperEnvironment(_ClosedCorpusEnvironment):
    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Mapping[str, Any]],
        max_observation_characters: int = 4000,
    ) -> "QasperEnvironment":
        grouped: dict[str, _Page] = {}
        order: list[str] = []
        seen_source_ids: set[str] = set()
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            source_id = str(chunk.get("source_id") or "")
            if not source_id or source_id in seen_source_ids:
                raise ValueError(f"duplicate or empty Qasper source ID: {source_id!r}")
            seen_source_ids.add(source_id)
            section = str(metadata.get("section") or "Document")
            title_path = str(metadata.get("title_path") or section)
            group_key = title_path or section
            if group_key not in grouped:
                grouped[group_key] = _Page(
                    title=title_path,
                    aliases={normalize_text(title_path), normalize_text(section)},
                    atoms=[],
                )
                order.append(group_key)
            paragraph_id = metadata.get("paragraph_id")
            if paragraph_id is None:
                paragraph_id = metadata.get("qasper_paragraph_id", source_id)
            grouped[group_key].atoms.append(
                {
                    "source_id": source_id,
                    "paragraph_id": paragraph_id,
                    "section": section,
                    "title_path": title_path,
                    "content": str(
                        metadata.get("qasper_evidence_text")
                        or metadata.get("original_text")
                        or chunk.get("content")
                        or ""
                    ),
                }
            )
        return cls(
            [grouped[key] for key in order],
            max_observation_characters=max_observation_characters,
        )
