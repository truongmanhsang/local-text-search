from __future__ import annotations

from typing import Any, Sequence

import httpx

from local_text_search.models import ChatTurn, SearchHit
from local_text_search.providers.base import BaseProvider, ProviderError


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def _post(self, prompt: str, *, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise ProviderError("OpenAI-compatible response did not include choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)

    def generate_answer(
        self,
        question: str,
        context_chunks: Sequence[SearchHit],
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> str:
        return self._post(
            self.build_context_prompt(question, context_chunks, conversation_history),
            max_tokens=700,
        )

    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[str]:
        candidate_ids = [candidate.chunk_id for candidate in candidates]
        response = self._post(self.build_rerank_prompt(query, candidates), max_tokens=300)
        return self.parse_rerank_response(response, candidate_ids)
