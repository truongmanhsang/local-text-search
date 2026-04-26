from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from local_text_search.config import (
    AppConfig,
    load_config,
    provider_readiness,
    register_vault,
    save_config,
)


def test_load_config_creates_default_file(app_home: Path) -> None:
    config = load_config(create_if_missing=True)
    assert config.active_vault is None
    assert (app_home / "config.toml").exists()


def test_register_vault_persists_active_vault(tmp_path: Path, app_home: Path) -> None:
    vault_dir = tmp_path / "notes"
    vault_dir.mkdir()
    config = load_config(create_if_missing=True)
    register_vault(config, "notes", vault_dir)
    save_config(config)

    reloaded = load_config(create_if_missing=False)
    assert reloaded.active_vault == "notes"
    assert reloaded.require_vault("notes").path == vault_dir.resolve()


def test_provider_readiness_uses_environment(monkeypatch: pytest.MonkeyPatch, app_home: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    config = load_config(create_if_missing=True)
    readiness = provider_readiness(config)
    assert readiness["qdrant.remote"] is True
    assert readiness["embeddings.openai"] is True
    assert readiness["providers.openai"] is True
    assert readiness["providers.anthropic"] is False


def test_invalid_provider_value_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"providers": {"default_provider": "invalid"}})


def test_qdrant_remote_url_normalizes_missing_scheme() -> None:
    config = AppConfig.model_validate({"qdrant": {"location": "remote", "url": "192.168.1.8:6333/"}})
    assert config.qdrant.url == "http://192.168.1.8:6333/"


def test_provider_master_prompt_is_configurable() -> None:
    config = AppConfig.model_validate({"providers": {"master_prompt": "Answer in JSON only."}})
    assert config.providers.master_prompt == "Answer in JSON only."
