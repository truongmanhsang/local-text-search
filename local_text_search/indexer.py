from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Literal

from local_text_search.chunker import chunk_document
from local_text_search.config import AppConfig, VaultConfig
from local_text_search.embeddings import EmbeddingClient, build_embedding_client
from local_text_search.models import ChunkRecord, FileRecord, IndexStats
from local_text_search.storage import VaultStorage


@dataclass(slots=True)
class IndexProgress:
    phase: Literal["scan", "delete", "refresh", "complete"]
    current: int
    total: int
    path: str | None
    action: str
    stats: IndexStats


class Indexer:
    def __init__(
        self,
        *,
        config: AppConfig,
        vault: VaultConfig,
        embedding_client: EmbeddingClient | None = None,
        storage: VaultStorage | None = None,
    ) -> None:
        self.config = config
        self.vault = vault
        self.embedding_client = embedding_client or build_embedding_client(config)
        self.storage = storage or VaultStorage(vault, config.qdrant)

    def close(self) -> None:
        self.storage.close()

    @staticmethod
    def slug_chunk_id(file_path: str, chunk_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}:{chunk_index}"))

    @staticmethod
    def _matches_pattern(relative_path: str, pattern: str) -> bool:
        path = Path(relative_path)
        normalized = pattern[3:] if pattern.startswith("**/") else pattern
        return path.match(pattern) or path.match(normalized) or fnmatch(relative_path, pattern) or fnmatch(
            relative_path, normalized
        )

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_supported(self, relative_path: str) -> bool:
        include = any(
            self._matches_pattern(relative_path, pattern)
            for pattern in self.config.indexing.include_patterns
        )
        exclude = any(
            self._matches_pattern(relative_path, pattern)
            for pattern in self.config.indexing.exclude_patterns
        )
        return include and not exclude

    def _scan_files(self) -> list[Path]:
        results: list[Path] = []
        for candidate in self.vault.path.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(self.vault.path).as_posix()
            if self._is_supported(relative):
                results.append(candidate)
        return sorted(results)

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _build_chunks(self, relative_path: str, modified_time: float, text: str) -> list[ChunkRecord]:
        drafts = chunk_document(
            text,
            is_markdown=relative_path.endswith((".md", ".markdown")),
            chunk_size_words=self.config.indexing.chunk_size_words,
            chunk_overlap_words=self.config.indexing.chunk_overlap_words,
        )
        chunks: list[ChunkRecord] = []
        for draft in drafts:
            chunks.append(
                ChunkRecord(
                    chunk_id=self.slug_chunk_id(relative_path, draft.chunk_index),
                    file_path=relative_path,
                    vault_name=self.vault.name,
                    heading=draft.heading,
                    tags=draft.tags,
                    backlinks=draft.backlinks,
                    modified_time=modified_time,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    content_hash=self.hash_text(draft.text),
                    token_count=draft.token_count,
                    term_frequencies=self.storage.encoder.term_frequencies(draft.text),
                )
            )
        return chunks

    def run(
        self,
        full_rebuild: bool = False,
        progress_callback: Callable[[IndexProgress], None] | None = None,
    ) -> IndexStats:
        stats = IndexStats()
        if full_rebuild:
            self.storage.reset()
        manifest = self.storage.load_file_manifest()
        file_paths = self._scan_files()
        current_paths = {file_path.relative_to(self.vault.path).as_posix() for file_path in file_paths}
        removed = sorted(set(manifest) - current_paths)
        total_steps = len(file_paths) + len(removed) + 1
        completed_steps = 0
        now = time.time()
        for file_path in file_paths:
            stats.scanned_files += 1
            relative_path = file_path.relative_to(self.vault.path).as_posix()
            file_stat = file_path.stat()
            existing = manifest.get(relative_path)
            action = "indexing"
            if (
                existing
                and not full_rebuild
                and existing.size == file_stat.st_size
                and abs(existing.modified_time - file_stat.st_mtime) < 1e-6
            ):
                stats.skipped_files += 1
                action = "skipped"
            else:
                text = self._read_text(file_path)
                file_hash = self.hash_text(text)
                file_record = FileRecord(
                    path=relative_path,
                    file_hash=file_hash,
                    size=file_stat.st_size,
                    modified_time=file_stat.st_mtime,
                    indexed_at=now,
                )
                if existing and existing.file_hash == file_hash:
                    self.storage.update_file_metadata(file_record)
                    stats.skipped_files += 1
                    action = "metadata-updated"
                else:
                    duplicate_path = self.storage.find_duplicate_file(file_hash, relative_path)
                    if duplicate_path:
                        source_chunks = self.storage.get_chunks_for_file(duplicate_path)
                        new_chunk_ids = [
                            self.slug_chunk_id(relative_path, chunk.chunk_index) for chunk in source_chunks
                        ]
                        file_record.duplicate_of = duplicate_path
                        cloned = self.storage.clone_file_from_existing(
                            source_path=duplicate_path,
                            file_record=file_record,
                            new_chunk_ids=new_chunk_ids,
                            modified_time=file_stat.st_mtime,
                        )
                        stats.reused_files += 1
                        stats.chunks_upserted += len(cloned)
                        action = "reused-duplicate"
                    else:
                        chunks = self._build_chunks(relative_path, file_stat.st_mtime, text)
                        if not chunks:
                            self.storage.remove_file(relative_path)
                            self.storage.update_file_metadata(file_record)
                            stats.indexed_files += 1
                            action = "indexed-empty"
                        else:
                            texts = [chunk.text for chunk in chunks]
                            dense_vectors = self.embedding_client.embed_texts(texts)
                            self.storage.save_file_and_chunks(
                                file_record=file_record,
                                chunks=chunks,
                                dense_vectors=dense_vectors,
                                embedding_fingerprint=self.embedding_client.fingerprint,
                            )
                            stats.indexed_files += 1
                            stats.chunks_upserted += len(chunks)
                            action = "indexed"
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(
                    IndexProgress(
                        phase="scan",
                        current=completed_steps,
                        total=total_steps,
                        path=relative_path,
                        action=action,
                        stats=stats.model_copy(),
                    )
                )
        for path in removed:
            self.storage.remove_file(path)
            stats.deleted_files += 1
            completed_steps += 1
            if progress_callback is not None:
                progress_callback(
                    IndexProgress(
                        phase="delete",
                        current=completed_steps,
                        total=total_steps,
                        path=path,
                        action="deleted",
                        stats=stats.model_copy(),
                    )
                )
        if progress_callback is not None:
            progress_callback(
                IndexProgress(
                    phase="refresh",
                    current=completed_steps,
                    total=total_steps,
                    path=None,
                    action="refreshing sparse vectors",
                    stats=stats.model_copy(),
                )
            )
        if stats.indexed_files or stats.reused_files or stats.deleted_files or full_rebuild:
            self.storage.refresh_sparse_vectors()
        completed_steps += 1
        if progress_callback is not None:
            progress_callback(
                IndexProgress(
                    phase="complete",
                    current=completed_steps,
                    total=total_steps,
                    path=None,
                    action="complete",
                    stats=stats.model_copy(),
                )
            )
        return stats
