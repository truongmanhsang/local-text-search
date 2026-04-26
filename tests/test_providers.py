from __future__ import annotations

import pytest

from local_text_search.config import AppConfig
from local_text_search.providers.anthropic_provider import AnthropicProvider
from local_text_search.providers.base import ProviderError, build_provider
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
