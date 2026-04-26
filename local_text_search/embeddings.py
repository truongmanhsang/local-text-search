from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from local_text_search.config import AppConfig


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingClient(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, host: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self.host}:{self.model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc
        payload = response.json()
        if "embeddings" in payload:
            embeddings = payload["embeddings"]
        elif "embedding" in payload:
            embeddings = [payload["embedding"]]
        else:
            raise EmbeddingError("Ollama response did not include embeddings.")
        return [[float(value) for value in vector] for vector in embeddings]


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def fingerprint(self) -> str:
        return f"openai:{self.base_url}:{self.model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"OpenAI-compatible embedding request failed: {exc}") from exc
        data = response.json().get("data")
        if not isinstance(data, list):
            raise EmbeddingError("Embedding response did not include `data`.")
        return [[float(value) for value in item["embedding"]] for item in data]


def build_embedding_client(config: AppConfig) -> EmbeddingClient:
    provider = config.embeddings.default_provider
    if provider == "ollama":
        settings = config.embeddings.ollama
        return OllamaEmbeddingClient(
            host=settings.host,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
        )
    settings = config.embeddings.openai
    api_key = settings.resolved_api_key()
    if not api_key:
        raise EmbeddingError(
            f"OpenAI embedding API key is missing. Set `{settings.api_key_env}` or configure `api_key`."
        )
    return OpenAICompatibleEmbeddingClient(
        base_url=settings.base_url,
        api_key=api_key,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
    )
