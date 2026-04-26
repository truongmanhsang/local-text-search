from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qdrant_client.http import models as qdrant_models

from local_text_search.config import AppConfig, VaultConfig
from local_text_search.embeddings import EmbeddingClient, build_embedding_client
from local_text_search.models import AnswerResult, ChatTurn, SearchHit, SearchMode
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
        hits = self.apply_wikilink_expansion(hits, top_k=limit)
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

    def apply_wikilink_expansion(self, hits: Sequence[SearchHit], *, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        alias_index = self.storage.load_note_alias_index()
        if not alias_index:
            return list(hits[:top_k])
        anchor_count = max(0, self.config.search.wikilink_anchor_count)
        if anchor_count == 0:
            return list(hits[:top_k])
        candidate_scores: dict[str, float] = {hit.chunk_id: hit.score for hit in hits}
        candidate_hits: dict[str, SearchHit] = {hit.chunk_id: hit.model_copy() for hit in hits}
        target_path_scores: dict[str, float] = {}
        for anchor in hits[:anchor_count]:
            for backlink in anchor.backlinks:
                backlink_key = self.storage.normalize_note_reference(backlink)
                if not backlink_key:
                    continue
                linked_paths = alias_index.get(backlink_key, set())
                if not linked_paths:
                    continue
                for linked_path in linked_paths:
                    if linked_path == anchor.file_path:
                        continue
                    target_path_scores[linked_path] = max(
                        target_path_scores.get(linked_path, 0.0),
                        anchor.score * self.config.search.wikilink_weight,
                    )
        if not target_path_scores:
            return list(hits[:top_k])
        linked_paths = sorted(target_path_scores, key=target_path_scores.get, reverse=True)
        linked_hits = self.storage.get_search_hits_for_file_paths(
            linked_paths,
            per_file_limit=self.config.search.wikilink_chunks_per_target,
            search_mode=SearchMode.HYBRID,
        )
        for linked_hit in linked_hits:
            path_bonus = target_path_scores.get(linked_hit.file_path, 0.0)
            chunk_decay = max(0.6, 1.0 - (linked_hit.chunk_index * 0.1))
            boosted_score = path_bonus * chunk_decay
            if linked_hit.chunk_id in candidate_hits:
                candidate_scores[linked_hit.chunk_id] += boosted_score
                existing = candidate_hits[linked_hit.chunk_id]
                existing.score = candidate_scores[linked_hit.chunk_id]
                if existing.semantic_score is None:
                    existing.semantic_score = None
                if existing.bm25_score is None:
                    existing.bm25_score = None
            else:
                linked_copy = linked_hit.model_copy()
                linked_copy.score = boosted_score
                candidate_hits[linked_copy.chunk_id] = linked_copy
                candidate_scores[linked_copy.chunk_id] = boosted_score
        ranked = sorted(
            candidate_hits.values(),
            key=lambda hit: (
                candidate_scores[hit.chunk_id],
                hit.semantic_score or 0.0,
                hit.bm25_score or 0.0,
                -hit.chunk_index,
            ),
            reverse=True,
        )
        for ranked_hit in ranked:
            ranked_hit.score = candidate_scores[ranked_hit.chunk_id]
        return ranked[:top_k]

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
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> AnswerResult:
        provider = build_provider(self.config, provider_name=provider_name)
        retrieval_query = self._build_chat_query(question, conversation_history)
        hits = self.search(
            retrieval_query,
            mode=SearchMode.HYBRID,
            top_k=top_k,
            rerank=rerank,
            provider=provider,
        )
        if not hits:
            raise SearchError("No search hits were found for the question.")
        answer = provider.generate_answer(question, hits, conversation_history)
        return AnswerResult(
            answer=answer,
            provider=provider.provider_name,
            model=provider.model_name,
            sources=hits,
            context_chunks=[hit.chunk_id for hit in hits],
        )

    @staticmethod
    def _build_chat_query(
        question: str,
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> str:
        if not conversation_history:
            return question
        recent_user_turns = [turn.content for turn in conversation_history if turn.role == "user"][-3:]
        if not recent_user_turns:
            return question
        return "\n".join([*recent_user_turns, question])
