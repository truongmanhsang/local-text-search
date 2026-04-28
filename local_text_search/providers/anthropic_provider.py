from __future__ import annotations

from typing import Sequence

import httpx

from local_text_search.models import ChatTurn, SearchHit
from local_text_search.providers.base import BaseProvider, ProviderError


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        master_prompt: str | None = None,
    ) -> None:
        super().__init__(master_prompt=master_prompt)
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def _post(self, prompt: str, *, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        payload = response.json()
        parts = payload.get("content", [])
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("type") == "text"]
        if not text_parts:
            raise ProviderError("Anthropic response did not include text content.")
        return "".join(text_parts)

    def complete(self, prompt: str, *, max_tokens: int = 700) -> str:
        return self._post(prompt, max_tokens=max_tokens)

    def generate_answer(
        self,
        question: str,
        context_chunks: Sequence[SearchHit],
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> str:
        return self.complete(
            self.build_context_prompt(
                question,
                context_chunks,
                conversation_history,
                master_prompt=self.master_prompt,
            ),
            max_tokens=700,
        )

    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[str]:
        candidate_ids = [candidate.chunk_id for candidate in candidates]
        response = self.complete(self.build_rerank_prompt(query, candidates), max_tokens=300)
        return self.parse_rerank_response(response, candidate_ids)
