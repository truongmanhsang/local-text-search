from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Sequence

from local_text_search.config import AppConfig
from local_text_search.models import SearchHit

RERANK_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class ProviderError(RuntimeError):
    """Raised when provider configuration or requests fail."""


class BaseProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_answer(self, question: str, context_chunks: Sequence[SearchHit]) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def build_context_prompt(question: str, context_chunks: Sequence[SearchHit]) -> str:
        sections = []
        for index, chunk in enumerate(context_chunks, start=1):
            heading = chunk.heading or "No heading"
            sections.append(
                "\n".join(
                    [
                        f"[{index}] chunk_id={chunk.chunk_id}",
                        f"source={chunk.file_path}",
                        f"heading={heading}",
                        "content:",
                        chunk.text,
                    ]
                )
            )
        context = "\n\n".join(sections)
        return (
            "Answer the question using only the provided context. "
            "Cite supporting chunks inline using square brackets like [1] or [2]. "
            "If the answer is not in the context, say so.\n\n"
            f"Question:\n{question}\n\nContext:\n{context}"
        )

    @staticmethod
    def build_rerank_prompt(query: str, candidates: Sequence[SearchHit]) -> str:
        lines = [
            "Rank the candidate chunks from most relevant to least relevant for the query.",
            "Return JSON only in one of these forms:",
            '{"ordered_chunk_ids": ["chunk-a", "chunk-b"]}',
            '["chunk-a", "chunk-b"]',
            "",
            f"Query: {query}",
            "",
        ]
        for candidate in candidates:
            lines.extend(
                [
                    f"chunk_id={candidate.chunk_id}",
                    f"source={candidate.source_label}",
                    candidate.excerpt(400),
                    "",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def parse_rerank_response(response_text: str, candidate_ids: Sequence[str]) -> list[str]:
        match = RERANK_JSON_RE.search(response_text)
        if not match:
            return list(candidate_ids)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return list(candidate_ids)
        if isinstance(payload, dict):
            ordered = payload.get("ordered_chunk_ids", [])
        elif isinstance(payload, list):
            ordered = payload
        else:
            ordered = []
        filtered = [item for item in ordered if item in candidate_ids]
        for candidate_id in candidate_ids:
            if candidate_id not in filtered:
                filtered.append(candidate_id)
        return filtered


def build_provider(config: AppConfig, provider_name: str | None = None) -> BaseProvider:
    selected = provider_name or config.providers.default_provider
    if selected == "openai":
        from local_text_search.providers.openai_provider import OpenAIProvider

        settings = config.providers.openai
        api_key = settings.resolved_api_key()
        if not api_key:
            raise ProviderError(
                f"OpenAI API key is missing. Set `{settings.api_key_env}` or configure `api_key`."
            )
        return OpenAIProvider(
            base_url=settings.base_url,
            api_key=api_key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
        )
    if selected == "anthropic":
        from local_text_search.providers.anthropic_provider import AnthropicProvider

        settings = config.providers.anthropic
        api_key = settings.resolved_api_key()
        if not api_key:
            raise ProviderError(
                f"Anthropic API key is missing. Set `{settings.api_key_env}` or configure `api_key`."
            )
        return AnthropicProvider(
            api_key=api_key,
            model=settings.model,
            timeout_seconds=settings.timeout_seconds,
        )
    from local_text_search.providers.ollama_provider import OllamaProvider

    settings = config.providers.ollama
    return OllamaProvider(host=settings.host, model=settings.model, timeout_seconds=settings.timeout_seconds)
