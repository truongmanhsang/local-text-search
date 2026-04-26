from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class VaultRecord(BaseModel):
    name: str
    path: str


class FileRecord(BaseModel):
    path: str
    file_hash: str
    size: int
    modified_time: float
    indexed_at: float
    duplicate_of: str | None = None


class ChunkDraft(BaseModel):
    heading: str | None = None
    tags: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    text: str
    chunk_index: int
    token_count: int


class ChunkRecord(BaseModel):
    chunk_id: str
    file_path: str
    vault_name: str
    heading: str | None = None
    tags: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    modified_time: float
    chunk_index: int
    text: str
    content_hash: str
    token_count: int
    term_frequencies: dict[int, int] = Field(default_factory=dict)

    def qdrant_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "vault_name": self.vault_name,
            "heading": self.heading,
            "tags": self.tags,
            "backlinks": self.backlinks,
            "modified_time": self.modified_time,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "source": self.source_label,
        }

    @property
    def source_label(self) -> str:
        if self.heading:
            return f"{self.file_path}#{self.heading}"
        return self.file_path


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    file_path: str
    vault_name: str
    heading: str | None = None
    text: str
    tags: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    modified_time: float
    chunk_index: int
    search_mode: SearchMode
    semantic_score: float | None = None
    bm25_score: float | None = None

    @property
    def source_label(self) -> str:
        if self.heading:
            return f"{self.file_path}#{self.heading}"
        return self.file_path

    def excerpt(self, limit: int = 240) -> str:
        collapsed = " ".join(self.text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3].rstrip()}..."


class AnswerResult(BaseModel):
    answer: str
    provider: str
    model: str
    sources: list[SearchHit] = Field(default_factory=list)
    context_chunks: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: str
    content: str


class IndexStats(BaseModel):
    scanned_files: int = 0
    indexed_files: int = 0
    reused_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    chunks_upserted: int = 0
