from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import requests


def _completion_endpoint(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError("model service base URL is required")
    if base.endswith("/chat/completions") or base.endswith("/completions"):
        return base
    return base + "/completions"


def _usage(payload: Mapping[str, Any] | None) -> dict[str, int]:
    data = dict(payload or {})
    prompt = int(data.get("prompt_tokens") or 0)
    completion = int(data.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(data.get("total_tokens") or prompt + completion),
    }


def _choice_text(payload: Mapping[str, Any]) -> str:
    choices = list(payload.get("choices") or [])
    if not choices:
        raise ValueError("model service returned no completion choices")
    choice = dict(choices[0] or {})
    if choice.get("text") is not None:
        return str(choice.get("text") or "")
    message = dict(choice.get("message") or {})
    if message.get("content") is not None:
        return str(message.get("content") or "")
    raise ValueError("model service completion contains no text")


def _truncate_at_stop(text: str, stop: str) -> str:
    marker = str(stop or "")
    if not marker:
        return text
    position = text.find(marker)
    return text if position < 0 else text[:position]


@dataclass
class CompletionClient:
    api_key: str
    base_url: str
    timeout: float = 180.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    session: Any | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("model service API key is required")
        self.session = self.session or requests.Session()

    def complete(
        self,
        model: str,
        prompt: str,
        stop: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> tuple[str, dict[str, int]]:
        endpoint = _completion_endpoint(self.base_url)
        request: dict[str, Any] = {
            "model": str(model),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stop": [str(stop)],
        }
        if endpoint.endswith("/chat/completions"):
            request["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "Act as a text-completion engine. Continue only from the final "
                        "characters of the supplied transcript. Do not repeat or rewrite "
                        "existing transcript text; return only the missing continuation."
                    ),
                },
                {"role": "user", "content": str(prompt)},
            ]
        else:
            request["prompt"] = str(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(max(1, int(self.max_attempts))):
            try:
                response = self.session.post(
                    endpoint,
                    headers=headers,
                    json=request,
                    timeout=float(self.timeout),
                    verify=True,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise ValueError("model service response must be a JSON object")
                text = _truncate_at_stop(_choice_text(payload), str(stop))
                return text.rstrip(), _usage(payload.get("usage"))
            except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt + 1 >= max(1, int(self.max_attempts)):
                    break
                time.sleep(float(self.retry_backoff_seconds) * (2**attempt))
        raise RuntimeError("model service request failed after bounded retries") from last_error
