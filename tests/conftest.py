from __future__ import annotations

from pathlib import Path

import pytest

from local_text_search.config import AppConfig, load_config, register_vault, save_config
from local_text_search.embeddings import EmbeddingClient
from local_text_search.storage import VaultStorage


class FakeEmbeddingClient(EmbeddingClient):
    @property
    def fingerprint(self) -> str:
        return "fake:test"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            checksum = sum(ord(character) for character in text)
            vectors.append(
                [
                    float(len(text.split()) or 1),
                    float(checksum % 997) / 997.0,
                    float((checksum // 3) % 991) / 991.0,
                ]
            )
        return vectors


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture()
def app_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "app-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_TEXT_SEARCH_HOME", str(home))
    return home


@pytest.fixture()
def registered_vault(tmp_path: Path, app_home: Path) -> tuple[object, Path]:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = load_config(create_if_missing=True)
    register_vault(config, "notes", vault_dir)
    save_config(config)
    return config.require_vault("notes"), vault_dir


@pytest.fixture()
def config_with_vault(tmp_path: Path, app_home: Path) -> tuple[AppConfig, object, Path]:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    config = load_config(create_if_missing=True)
    register_vault(config, "notes", vault_dir)
    save_config(config)
    return config, config.require_vault("notes"), vault_dir
