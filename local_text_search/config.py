from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import tomli_w
from pydantic import BaseModel, Field, field_validator

from local_text_search.models import VaultRecord

APP_HOME_ENV = "LOCAL_TEXT_SEARCH_HOME"
DEFAULT_APP_DIRNAME = ".local-text-search"
DEFAULT_CONFIG_NAME = "config.toml"

EXAMPLE_CONFIG = """active_vault = "notes"

[indexing]
include_patterns = ["**/*.md", "**/*.markdown", "**/*.txt"]
exclude_patterns = [".git/**", ".obsidian/**", ".local-text-search/**"]
chunk_size_words = 220
chunk_overlap_words = 40
batch_size = 24

[search]
semantic_weight = 0.65
bm25_weight = 0.35
default_top_k = 8
rerank_top_n = 6
rerank_default_search = false
rerank_default_ask = true

[qdrant]
location = "local"
url = "http://192.168.1.8:6333/"
api_key_env = "QDRANT_API_KEY"
timeout_seconds = 10

[embeddings]
default_provider = "ollama"

[embeddings.ollama]
host = "http://localhost:11434"
model = "nomic-embed-text"
timeout_seconds = 60

[embeddings.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "text-embedding-3-small"
timeout_seconds = 60

[providers]
default_provider = "ollama"

[providers.ollama]
host = "http://localhost:11434"
model = "llama3.1"
timeout_seconds = 120

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o-mini"
timeout_seconds = 120

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-3-5-haiku-latest"
timeout_seconds = 120

[[vaults]]
name = "notes"
path = "/absolute/path/to/your/vault"
"""


def get_app_home() -> Path:
    override = os.environ.get(APP_HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / DEFAULT_APP_DIRNAME).resolve()


def get_config_path() -> Path:
    return get_app_home() / DEFAULT_CONFIG_NAME


def get_data_root() -> Path:
    return get_app_home() / "data"


class OllamaEmbeddingConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout_seconds: float = 60.0


class OpenAIEmbeddingConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-small"
    timeout_seconds: float = 60.0

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)


class EmbeddingsConfig(BaseModel):
    default_provider: Literal["ollama", "openai"] = "ollama"
    ollama: OllamaEmbeddingConfig = Field(default_factory=OllamaEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)


class OllamaProviderConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout_seconds: float = 120.0


class OpenAIProviderConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 120.0

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)


class AnthropicProviderConfig(BaseModel):
    api_key: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-3-5-haiku-latest"
    timeout_seconds: float = 120.0

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)


class ProvidersConfig(BaseModel):
    default_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)


class IndexingConfig(BaseModel):
    include_patterns: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.markdown", "**/*.txt"]
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [".git/**", ".obsidian/**", ".local-text-search/**"]
    )
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    batch_size: int = 24


class SearchConfig(BaseModel):
    semantic_weight: float = 0.65
    bm25_weight: float = 0.35
    default_top_k: int = 8
    rerank_top_n: int = 6
    rerank_default_search: bool = False
    rerank_default_ask: bool = True
    wikilink_weight: float = 0.12
    wikilink_anchor_count: int = 3
    wikilink_chunks_per_target: int = 2


class QdrantConfig(BaseModel):
    location: Literal["local", "remote"] = "local"
    url: str | None = None
    api_key: str | None = None
    api_key_env: str = "QDRANT_API_KEY"
    timeout_seconds: float = 10.0

    @field_validator("url", mode="before")
    @classmethod
    def _normalize_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Qdrant remote URL must include a host, e.g. `http://192.168.1.8:6333/`.")
        return normalized

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.environ.get(self.api_key_env)

    def require_remote_url(self) -> str:
        if not self.url:
            raise ValueError("Qdrant remote URL is required when `qdrant.location = \"remote\"`.")
        return self.url


class VaultConfig(BaseModel):
    name: str
    path: Path

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    def to_record(self) -> VaultRecord:
        return VaultRecord(name=self.name, path=str(self.path))


class AppConfig(BaseModel):
    active_vault: str | None = None
    vaults: list[VaultConfig] = Field(default_factory=list)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)

    def find_vault(self, name: str) -> VaultConfig | None:
        for vault in self.vaults:
            if vault.name == name:
                return vault
        return None

    def require_vault(self, name: str | None) -> VaultConfig:
        resolved_name = name or self.active_vault
        if not resolved_name:
            raise ValueError("No vault selected. Use --vault or run `local-text-search init <folder>`.")
        vault = self.find_vault(resolved_name)
        if vault is None:
            raise ValueError(f"Vault `{resolved_name}` is not registered.")
        return vault


def ensure_app_dirs() -> None:
    get_app_home().mkdir(parents=True, exist_ok=True)
    get_data_root().mkdir(parents=True, exist_ok=True)


def default_config() -> AppConfig:
    return AppConfig()


def load_config(create_if_missing: bool = True) -> AppConfig:
    ensure_app_dirs()
    path = get_config_path()
    if not path.exists():
        config = default_config()
        if create_if_missing:
            save_config(config)
        return config
    with path.open("rb") as handle:
        raw = handle.read()
    if not raw.strip():
        return default_config()
    import tomllib

    return AppConfig.model_validate(tomllib.loads(raw.decode("utf-8")))


def save_config(config: AppConfig) -> None:
    ensure_app_dirs()
    path = get_config_path()
    payload = config.model_dump(mode="json", exclude_none=True)
    with path.open("wb") as handle:
        handle.write(tomli_w.dumps(payload).encode("utf-8"))


def register_vault(config: AppConfig, name: str, folder: Path) -> AppConfig:
    normalized = folder.expanduser().resolve()
    existing = config.find_vault(name)
    if existing:
        existing.path = normalized
    else:
        config.vaults.append(VaultConfig(name=name, path=normalized))
    config.active_vault = name
    return config


def vault_data_dir(vault_name: str) -> Path:
    return get_data_root() / vault_name


def provider_readiness(config: AppConfig) -> dict[str, bool]:
    return {
        "qdrant.remote": config.qdrant.location == "local" or bool(config.qdrant.url),
        "embeddings.ollama": True,
        "embeddings.openai": bool(config.embeddings.openai.resolved_api_key()),
        "providers.ollama": True,
        "providers.openai": bool(config.providers.openai.resolved_api_key()),
        "providers.anthropic": bool(config.providers.anthropic.resolved_api_key()),
    }
