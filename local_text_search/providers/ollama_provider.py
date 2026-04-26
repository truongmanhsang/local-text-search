from __future__ import annotations

from typing import Sequence

import httpx

from local_text_search.models import ChatTurn, SearchHit
from local_text_search.providers.base import BaseProvider, ProviderError


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, *, host: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def _post(self, prompt: str) -> str:
        body = {
            "model": self.model_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},
        }
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json=body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("Ollama response did not include message content.")
        return content

    def generate_answer(
        self,
        question: str,
        context_chunks: Sequence[SearchHit],
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> str:
        return self._post(self.build_context_prompt(question, context_chunks, conversation_history))

    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[str]:
        candidate_ids = [candidate.chunk_id for candidate in candidates]
        response = self._post(self.build_rerank_prompt(query, candidates))
        return self.parse_rerank_response(response, candidate_ids)
