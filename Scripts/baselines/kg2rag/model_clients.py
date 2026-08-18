from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import requests


def _endpoint(base_url: str, suffix: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError(f"missing base URL for {suffix}")
    normalized_suffix = "/" + suffix.strip("/")
    if base.endswith(normalized_suffix):
        return base
    return base + normalized_suffix


def _usage(payload: Mapping[str, Any] | None) -> dict[str, int]:
    data = dict(payload or {})
    prompt = int(data.get("prompt_tokens") or 0)
    completion = int(data.get("completion_tokens") or 0)
    total = int(data.get("total_tokens") or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, Mapping):
        return dict(content)
    text = str(content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("chat response JSON must be an object")
    return dict(value)


@dataclass
class OpenAICompatibleClient:
    api_key: str
    llm_base_url: str
    embedding_base_url: str
    reranker_base_url: str
    timeout: float = 120.0
    max_attempts: int = 3
    json_max_attempts: int = 6
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._session = requests.Session()

    @classmethod
    def from_environment(cls, **overrides: Any) -> "OpenAICompatibleClient":
        llm_base = os.getenv("KG2RAG_LLM_BASE_URL", "")
        embedding_base = os.getenv("KG2RAG_EMBEDDING_BASE_URL", "")
        reranker_base = os.getenv("KG2RAG_RERANKER_BASE_URL", "")
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            llm_base_url=llm_base,
            embedding_base_url=embedding_base,
            reranker_base_url=reranker_base or embedding_base,
            **overrides,
        )

    def _post(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.timeout,
                    verify=True,
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, Mapping):
                    raise ValueError("model service response must be a JSON object")
                return dict(result)
            except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt + 1 >= self.max_attempts:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("model service request failed after bounded retries") from last_error

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        request = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error: Exception | None = None
        for attempt in range(self.json_max_attempts):
            try:
                response = self._post(
                    _endpoint(self.llm_base_url, "chat/completions"), request
                )
                response_usage = _usage(response.get("usage"))
                for key in total_usage:
                    total_usage[key] += response_usage[key]
                choices = response.get("choices") or []
                if not choices:
                    raise ValueError("chat response contains no choices")
                content = dict(choices[0].get("message") or {}).get("content")
                return _json_content(content), total_usage
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt + 1 >= self.json_max_attempts:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError(
            "chat response content was invalid after bounded retries"
        ) from last_error

    def embed(
        self, model: str, texts: list[str]
    ) -> tuple[np.ndarray, dict[str, int]]:
        response = self._post(
            _endpoint(self.embedding_base_url, "embeddings"),
            {"model": model, "input": list(texts)},
        )
        rows = list(response.get("data") or [])
        if len(rows) != len(texts):
            raise ValueError(
                f"embedding response count mismatch: {len(rows)} != {len(texts)}"
            )
        rows.sort(key=lambda item: int(item.get("index") or 0))
        vectors = np.asarray([item.get("embedding") for item in rows], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embedding response is not a two-dimensional matrix")
        return vectors, _usage(response.get("usage"))

    def rerank(
        self, model: str, query: str, documents: list[str]
    ) -> tuple[list[float], dict[str, int]]:
        if not documents:
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        response = self._post(
            _endpoint(self.reranker_base_url, "rerank"),
            {
                "model": model,
                "query": query,
                "documents": list(documents),
                "return_documents": False,
                "top_n": len(documents),
            },
        )
        results = list(response.get("results") or response.get("data") or [])
        if len(results) != len(documents):
            raise ValueError(
                f"reranker response count mismatch: {len(results)} != {len(documents)}"
            )
        scores = [0.0] * len(documents)
        for position, item in enumerate(results):
            index = int(item.get("index", position))
            if index < 0 or index >= len(documents):
                raise ValueError(f"reranker returned invalid document index: {index}")
            scores[index] = float(
                item.get("relevance_score", item.get("score", item.get("similarity", 0.0)))
            )
        return scores, _usage(response.get("usage"))
