from dataclasses import dataclass
from typing import Optional


@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    device: str = "cuda:2"
    backend: str = "local"  # Options: 'local', 'vllm'
    api_base: str = "http://localhost:8011/v1"
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_backoff: float = 0.5
    request_timeout: float = 60.0
