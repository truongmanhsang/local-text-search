from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from local_text_search.config import QdrantConfig, VaultConfig, vault_data_dir
from local_text_search.models import ChunkRecord, FileRecord, SearchHit, SearchMode


class StorageError(RuntimeError):
    """Raised when storage operations fail."""


class BM25SparseEncoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def tokenize(text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*", text.lower())

    @staticmethod
    def term_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def term_frequencies(self, text: str) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for token in self.tokenize(text):
            counts[self.term_id(token)] += 1
        return dict(counts)

    def encode_document(
        self,
        *,
        term_frequencies: dict[int, int],
        document_length: int,
        average_document_length: float,
    ) -> tuple[list[int], list[float]]:
        if not term_frequencies:
            return [], []
        avgdl = max(average_document_length, 1.0)
        norm = self.k1 * (1 - self.b + self.b * (document_length / avgdl))
        pairs = []
        for term_id, frequency in term_frequencies.items():
            weight = ((self.k1 + 1) * frequency) / (frequency + norm)
            pairs.append((term_id, float(weight)))
        pairs.sort(key=lambda item: item[0])
        return [term_id for term_id, _ in pairs], [weight for _, weight in pairs]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        term_ids = sorted({self.term_id(token) for token in self.tokenize(text)})
        return term_ids, [1.0] * len(term_ids)


class VaultStorage:
    dense_vector_name = "dense"
    sparse_vector_name = "bm25"

    def __init__(self, vault: VaultConfig, qdrant_config: QdrantConfig) -> None:
        self.vault = vault
        self.qdrant_config = qdrant_config
        self.root = vault_data_dir(vault.name)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "state.sqlite"
        self.qdrant_path = self.root / "qdrant"
        self.encoder = BM25SparseEncoder()
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.collection_name = self._build_collection_name(vault.name)
        self.client = self._build_client()
        self._initialize_schema()

    @staticmethod
    def _build_collection_name(vault_name: str) -> str:
        sanitized = "".join(character.lower() if character.isalnum() else "_" for character in vault_name)
        sanitized = sanitized.strip("_") or "vault"
        return f"local_text_search__{sanitized}"

    def _build_client(self) -> QdrantClient:
        if self.qdrant_config.location == "remote":
            return QdrantClient(
                url=self.qdrant_config.require_remote_url(),
                api_key=self.qdrant_config.resolved_api_key(),
                timeout=int(self.qdrant_config.timeout_seconds),
            )
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(self.qdrant_path))

    def close(self) -> None:
        self.connection.close()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                indexed_at REAL NOT NULL,
                duplicate_of TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT,
                modified_time REAL NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                backlinks_json TEXT NOT NULL,
                term_freqs_json TEXT NOT NULL,
                FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def reset(self) -> None:
        self.connection.executescript(
            """
            DELETE FROM chunks;
            DELETE FROM files;
            DELETE FROM metadata;
            """
        )
        self.connection.commit()
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def ensure_collection(self, vector_size: int, embedding_fingerprint: str) -> None:
        current_fingerprint = self.get_metadata("embedding_fingerprint")
        if self.client.collection_exists(self.collection_name):
            if current_fingerprint and current_fingerprint != embedding_fingerprint:
                raise StorageError(
                    "Embedding configuration changed for this vault. Run `local-text-search reindex --yes`."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_vector_name: qdrant_models.SparseVectorParams(
                    modifier=qdrant_models.Modifier.IDF
                )
            },
        )
        self.set_metadata("embedding_fingerprint", embedding_fingerprint)

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def load_file_manifest(self) -> dict[str, FileRecord]:
        rows = self.connection.execute(
            "SELECT path, file_hash, size, modified_time, indexed_at, duplicate_of FROM files"
        ).fetchall()
        return {
            str(row["path"]): FileRecord(
                path=str(row["path"]),
                file_hash=str(row["file_hash"]),
                size=int(row["size"]),
                modified_time=float(row["modified_time"]),
                indexed_at=float(row["indexed_at"]),
                duplicate_of=row["duplicate_of"],
            )
            for row in rows
        }

    def save_file_and_chunks(
        self,
        file_record: FileRecord,
        chunks: list[ChunkRecord],
        dense_vectors: list[list[float]],
        embedding_fingerprint: str,
    ) -> None:
        if chunks:
            self.ensure_collection(len(dense_vectors[0]), embedding_fingerprint)
        old_ids = [
            row["chunk_id"]
            for row in self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_record.path,)
            ).fetchall()
        ]
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO files(path, file_hash, size, modified_time, indexed_at, duplicate_of)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    size = excluded.size,
                    modified_time = excluded.modified_time,
                    indexed_at = excluded.indexed_at,
                    duplicate_of = excluded.duplicate_of
                """,
                (
                    file_record.path,
                    file_record.file_hash,
                    file_record.size,
                    file_record.modified_time,
                    file_record.indexed_at,
                    file_record.duplicate_of,
                ),
            )
            self.connection.execute("DELETE FROM chunks WHERE file_path = ?", (file_record.path,))
            self.connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, file_path, chunk_index, heading, modified_time, text,
                    content_hash, token_count, tags_json, backlinks_json, term_freqs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.chunk_index,
                        chunk.heading,
                        chunk.modified_time,
                        chunk.text,
                        chunk.content_hash,
                        chunk.token_count,
                        json.dumps(chunk.tags),
                        json.dumps(chunk.backlinks),
                        json.dumps(chunk.term_frequencies),
                    )
                    for chunk in chunks
                ],
            )
        if old_ids and self.client.collection_exists(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=old_ids),
            )
        if chunks:
            points = [
                qdrant_models.PointStruct(
                    id=chunk.chunk_id,
                    vector={self.dense_vector_name: dense_vector},
                    payload=chunk.qdrant_payload(),
                )
                for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def clone_file_from_existing(
        self,
        *,
        source_path: str,
        file_record: FileRecord,
        new_chunk_ids: list[str],
        modified_time: float,
    ) -> list[ChunkRecord]:
        source_rows = self.connection.execute(
            """
            SELECT chunk_id, chunk_index, heading, text, content_hash, token_count,
                   tags_json, backlinks_json, term_freqs_json
            FROM chunks
            WHERE file_path = ?
            ORDER BY chunk_index ASC
            """,
            (source_path,),
        ).fetchall()
        source_chunk_ids = [str(row["chunk_id"]) for row in source_rows]
        dense_vectors = self._fetch_dense_vectors(source_chunk_ids)
        cloned_chunks: list[ChunkRecord] = []
        for index, row in enumerate(source_rows):
            cloned_chunks.append(
                ChunkRecord(
                    chunk_id=new_chunk_ids[index],
                    file_path=file_record.path,
                    vault_name=self.vault.name,
                    heading=row["heading"],
                    tags=json.loads(str(row["tags_json"])),
                    backlinks=json.loads(str(row["backlinks_json"])),
                    modified_time=modified_time,
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    content_hash=str(row["content_hash"]),
                    token_count=int(row["token_count"]),
                    term_frequencies={int(k): int(v) for k, v in json.loads(str(row["term_freqs_json"])).items()},
                )
            )
        self.save_file_and_chunks(
            file_record=file_record,
            chunks=cloned_chunks,
            dense_vectors=dense_vectors,
            embedding_fingerprint=self.get_metadata("embedding_fingerprint") or "reused",
        )
        return cloned_chunks

    def _fetch_dense_vectors(self, chunk_ids: list[str]) -> list[list[float]]:
        if not chunk_ids:
            return []
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_vectors=[self.dense_vector_name],
            with_payload=False,
        )
        points_by_id = {str(point.id): point for point in points}
        output: list[list[float]] = []
        for chunk_id in chunk_ids:
            point = points_by_id.get(chunk_id)
            if point is None:
                raise StorageError(f"Missing dense vector for chunk `{chunk_id}`.")
            vector_data = point.vector
            if isinstance(vector_data, dict):
                vector = vector_data.get(self.dense_vector_name)
            else:
                vector = vector_data
            if not isinstance(vector, list):
                raise StorageError(f"Invalid dense vector for chunk `{chunk_id}`.")
            output.append([float(value) for value in vector])
        return output

    def find_duplicate_file(self, file_hash: str, exclude_path: str) -> str | None:
        row = self.connection.execute(
            "SELECT path FROM files WHERE file_hash = ? AND path != ? LIMIT 1",
            (file_hash, exclude_path),
        ).fetchone()
        return None if row is None else str(row["path"])

    def get_chunks_for_file(self, path: str) -> list[ChunkRecord]:
        rows = self.connection.execute(
            """
            SELECT chunk_id, file_path, chunk_index, heading, modified_time, text,
                   content_hash, token_count, tags_json, backlinks_json, term_freqs_json
            FROM chunks
            WHERE file_path = ?
            ORDER BY chunk_index ASC
            """,
            (path,),
        ).fetchall()
        return [
            ChunkRecord(
                chunk_id=str(row["chunk_id"]),
                file_path=str(row["file_path"]),
                vault_name=self.vault.name,
                heading=row["heading"],
                tags=json.loads(str(row["tags_json"])),
                backlinks=json.loads(str(row["backlinks_json"])),
                modified_time=float(row["modified_time"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                content_hash=str(row["content_hash"]),
                token_count=int(row["token_count"]),
                term_frequencies={int(k): int(v) for k, v in json.loads(str(row["term_freqs_json"])).items()},
            )
            for row in rows
        ]

    def remove_file(self, path: str) -> int:
        chunk_ids = [
            row["chunk_id"]
            for row in self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE file_path = ?", (path,)
            ).fetchall()
        ]
        with self.connection:
            self.connection.execute("DELETE FROM files WHERE path = ?", (path,))
        if chunk_ids and self.client.collection_exists(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=chunk_ids),
            )
        return len(chunk_ids)

    def update_file_metadata(self, file_record: FileRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO files(path, file_hash, size, modified_time, indexed_at, duplicate_of)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    size = excluded.size,
                    modified_time = excluded.modified_time,
                    indexed_at = excluded.indexed_at,
                    duplicate_of = excluded.duplicate_of
                """,
                (
                    file_record.path,
                    file_record.file_hash,
                    file_record.size,
                    file_record.modified_time,
                    file_record.indexed_at,
                    file_record.duplicate_of,
                ),
            )

    def refresh_sparse_vectors(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        rows = self.connection.execute(
            """
            SELECT chunk_id, token_count, term_freqs_json
            FROM chunks
            ORDER BY file_path ASC, chunk_index ASC
            """
        ).fetchall()
        if not rows:
            return
        average_document_length = sum(int(row["token_count"]) for row in rows) / len(rows)
        updates: list[qdrant_models.PointVectors] = []
        for row in rows:
            term_frequencies = {int(k): int(v) for k, v in json.loads(str(row["term_freqs_json"])).items()}
            indices, values = self.encoder.encode_document(
                term_frequencies=term_frequencies,
                document_length=int(row["token_count"]),
                average_document_length=average_document_length,
            )
            updates.append(
                qdrant_models.PointVectors(
                    id=str(row["chunk_id"]),
                    vector={
                        self.sparse_vector_name: qdrant_models.SparseVector(indices=indices, values=values)
                    },
                )
            )
        batch_size = 256
        for start in range(0, len(updates), batch_size):
            self.client.update_vectors(
                collection_name=self.collection_name,
                points=updates[start : start + batch_size],
            )

    def query(
        self,
        *,
        query_vector: list[float] | qdrant_models.SparseVector,
        using: str,
        vault_name: str,
        limit: int,
    ) -> list[SearchHit]:
        if not self.client.collection_exists(self.collection_name):
            return []
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="vault_name",
                    match=qdrant_models.MatchValue(value=vault_name),
                )
            ]
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=using,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = response.points if hasattr(response, "points") else response
        hits: list[SearchHit] = []
        mode = SearchMode.SEMANTIC if using == self.dense_vector_name else SearchMode.BM25
        for point in points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id") or point.id),
                    score=float(point.score),
                    file_path=str(payload.get("file_path", "")),
                    vault_name=str(payload.get("vault_name", self.vault.name)),
                    heading=payload.get("heading"),
                    text=str(payload.get("text", "")),
                    tags=[str(item) for item in payload.get("tags", [])],
                    backlinks=[str(item) for item in payload.get("backlinks", [])],
                    modified_time=float(payload.get("modified_time", 0.0)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    search_mode=mode,
                )
            )
        return hits

    def has_indexed_chunks(self) -> bool:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return bool(row and int(row["count"]) > 0)

    def count_chunks(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return 0 if row is None else int(row["count"])

    def iter_chunks(self) -> Iterator[ChunkRecord]:
        rows = self.connection.execute("SELECT path FROM files ORDER BY path ASC").fetchall()
        for row in rows:
            for chunk in self.get_chunks_for_file(str(row["path"])):
                yield chunk
