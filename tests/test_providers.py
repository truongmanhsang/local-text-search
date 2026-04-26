from __future__ import annotations

import pytest

from local_text_search.config import AppConfig
from local_text_search.models import ChatTurn, SearchHit, SearchMode
from local_text_search.providers.anthropic_provider import AnthropicProvider
from local_text_search.providers.base import BaseProvider, ProviderError, build_provider
from local_text_search.providers.ollama_provider import OllamaProvider
from local_text_search.providers.openai_provider import OpenAIProvider


def test_default_provider_is_ollama() -> None:
    provider = build_provider(AppConfig())
    assert isinstance(provider, OllamaProvider)


def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AppConfig.model_validate({"providers": {"default_provider": "openai"}})
    with pytest.raises(ProviderError):
        build_provider(config)


def test_openai_provider_builds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = AppConfig.model_validate({"providers": {"default_provider": "openai"}})
    provider = build_provider(config)
    assert isinstance(provider, OpenAIProvider)


def test_anthropic_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = AppConfig.model_validate({"providers": {"default_provider": "anthropic"}})
    with pytest.raises(ProviderError):
        build_provider(config)


def test_anthropic_provider_builds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = AppConfig.model_validate({"providers": {"default_provider": "anthropic"}})
    provider = build_provider(config)
    assert isinstance(provider, AnthropicProvider)


def test_context_prompt_prioritizes_current_question() -> None:
    prompt = BaseProvider.build_context_prompt(
        "What notes mention beta?",
        [
            SearchHit(
                chunk_id="chunk-1",
                score=1.0,
                file_path="beta.md",
                vault_name="notes",
                heading="Beta",
                text="Beta note content",
                modified_time=0.0,
                chunk_index=0,
                search_mode=SearchMode.HYBRID,
            )
        ],
        [
            ChatTurn(role="user", content="Tell me about alpha"),
            ChatTurn(role="assistant", content="This assistant answer should not appear"),
        ],
    )
    assert "Prioritize the current question" in prompt
    assert "User: Tell me about alpha" in prompt
    assert "assistant answer should not appear" not in prompt


def test_context_prompt_includes_master_prompt() -> None:
    prompt = BaseProvider.build_context_prompt(
        "What notes mention beta?",
        [
            SearchHit(
                chunk_id="chunk-1",
                score=1.0,
                file_path="beta.md",
                vault_name="notes",
                heading="Beta",
                text="Beta note content",
                modified_time=0.0,
                chunk_index=0,
                search_mode=SearchMode.HYBRID,
            )
        ],
        master_prompt="Answer in JSON only.",
    )
    assert "Answer the current question using only the provided context." in prompt
    assert "Additional answering instructions" in prompt
    assert "Answer in JSON only." in prompt


def test_build_provider_propagates_master_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = AppConfig.model_validate(
        {
            "providers": {
                "default_provider": "openai",
                "master_prompt": "Use terse responses.",
            }
        }
    )
    provider = build_provider(config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.master_prompt == "Use terse responses."
