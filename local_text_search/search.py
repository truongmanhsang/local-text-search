from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qdrant_client.http import models as qdrant_models

from local_text_search.config import AppConfig, VaultConfig
from local_text_search.embeddings import EmbeddingClient, build_embedding_client
from local_text_search.models import AnswerResult, SearchHit, SearchMode
from local_text_search.providers.base import BaseProvider, build_provider
from local_text_search.storage import VaultStorage


class SearchError(RuntimeError):
    """Raised when search or ask operations fail."""


@dataclass(slots=True)
class MergedHit:
    hit: SearchHit
    semantic_score: float | None = None
    bm25_score: float | None = None
    merged_score: float = 0.0


class SearchService:
    def __init__(
        self,
        *,
        config: AppConfig,
        vault: VaultConfig,
        storage: VaultStorage | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        self.vault = vault
        self.storage = storage or VaultStorage(vault, config.qdrant)
        self.embedding_client = embedding_client or build_embedding_client(config)

    def close(self) -> None:
        self.storage.close()

    @staticmethod
    def _normalize_scores(hits: Sequence[SearchHit]) -> dict[str, float]:
        if not hits:
            return {}
        raw_scores = [hit.score for hit in hits]
        minimum = min(raw_scores)
        maximum = max(raw_scores)
        if maximum - minimum < 1e-9:
            return {hit.chunk_id: 1.0 for hit in hits}
        return {hit.chunk_id: (hit.score - minimum) / (maximum - minimum) for hit in hits}

    def _semantic_hits(self, query: str, limit: int) -> list[SearchHit]:
        vector = self.embedding_client.embed_query(query)
        return self.storage.query(
            query_vector=vector,
            using=self.storage.dense_vector_name,
            vault_name=self.vault.name,
            limit=limit,
        )

    def _bm25_hits(self, query: str, limit: int) -> list[SearchHit]:
        indices, values = self.storage.encoder.encode_query(query)
        sparse_vector = qdrant_models.SparseVector(indices=indices, values=values)
        return self.storage.query(
            query_vector=sparse_vector,
            using=self.storage.sparse_vector_name,
            vault_name=self.vault.name,
            limit=limit,
        )

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int | None = None,
        rerank: bool = False,
        provider: BaseProvider | None = None,
    ) -> list[SearchHit]:
        if not self.storage.has_indexed_chunks():
            raise SearchError("Vault has not been indexed yet. Run `local-text-search index` first.")
        limit = top_k or self.config.search.default_top_k
        if mode == SearchMode.SEMANTIC:
            hits = self._semantic_hits(query, limit)
        elif mode == SearchMode.BM25:
            hits = self._bm25_hits(query, limit)
        else:
            semantic_hits = self._semantic_hits(query, limit * 2)
            bm25_hits = self._bm25_hits(query, limit * 2)
            hits = self.merge_hits(semantic_hits=semantic_hits, bm25_hits=bm25_hits, top_k=limit)
        if rerank and hits:
            provider = provider or build_provider(self.config)
            rerank_candidates = hits[: self.config.search.rerank_top_n]
            ordered_ids = provider.rerank(
                query=query,
                candidates=rerank_candidates,
            )
            hits_by_id = {hit.chunk_id: hit for hit in rerank_candidates}
            ranked = [hits_by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in hits_by_id]
            remainder = [hit for hit in hits if hit.chunk_id not in {item.chunk_id for item in ranked}]
            hits = ranked + remainder
        return hits[:limit]

    def merge_hits(
        self,
        *,
        semantic_hits: Sequence[SearchHit],
        bm25_hits: Sequence[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        semantic_scores = self._normalize_scores(semantic_hits)
        bm25_scores = self._normalize_scores(bm25_hits)
        merged: dict[str, MergedHit] = {}
        for hit in semantic_hits:
            entry = merged.setdefault(hit.chunk_id, MergedHit(hit=hit))
            entry.semantic_score = semantic_scores[hit.chunk_id]
            entry.merged_score += entry.semantic_score * self.config.search.semantic_weight
        for hit in bm25_hits:
            entry = merged.setdefault(hit.chunk_id, MergedHit(hit=hit))
            entry.bm25_score = bm25_scores[hit.chunk_id]
            entry.merged_score += entry.bm25_score * self.config.search.bm25_weight
        ranked = sorted(
            merged.values(),
            key=lambda item: (
                item.merged_score,
                item.semantic_score or 0.0,
                item.bm25_score or 0.0,
                -item.hit.chunk_index,
            ),
            reverse=True,
        )
        output: list[SearchHit] = []
        for item in ranked[:top_k]:
            hit = item.hit.model_copy()
            hit.search_mode = SearchMode.HYBRID
            hit.score = item.merged_score
            hit.semantic_score = item.semantic_score
            hit.bm25_score = item.bm25_score
            output.append(hit)
        return output

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        provider_name: str | None = None,
        rerank: bool = True,
    ) -> AnswerResult:
        provider = build_provider(self.config, provider_name=provider_name)
        hits = self.search(
            question,
            mode=SearchMode.HYBRID,
            top_k=top_k,
            rerank=rerank,
            provider=provider,
        )
        if not hits:
            raise SearchError("No search hits were found for the question.")
        answer = provider.generate_answer(question, hits)
        return AnswerResult(
            answer=answer,
            provider=provider.provider_name,
            model=provider.model_name,
            sources=hits,
            context_chunks=[hit.chunk_id for hit in hits],
        )
