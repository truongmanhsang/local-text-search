from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from qdrant_client.http import models as qdrant_models

from local_text_search.config import AppConfig, VaultConfig
from local_text_search.embeddings import EmbeddingClient, build_embedding_client
from local_text_search.models import AnswerResult, ChatTurn, ChunkRecord, SearchHit, SearchMode, SummaryResult
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


@dataclass(slots=True)
class SummarySection:
    label: str
    content: str


class SearchService:
    FOLLOW_UP_PREFIXES = (
        "what about",
        "how about",
        "what else",
        "tell me more",
        "expand on",
        "compare that",
        "and ",
        "also ",
        "còn",
        "thế còn",
        "vậy còn",
        "nói thêm",
        "chi tiết hơn",
        "mở rộng",
        "so sánh",
    )
    REFERENTIAL_TERMS = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "he",
        "she",
        "him",
        "her",
        "again",
        "more",
        "same",
        "above",
        "former",
        "latter",
        "đó",
        "nó",
        "này",
        "kia",
        "ấy",
        "vậy",
        "thế",
        "thêm",
        "nữa",
        "tiếp",
    }

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
    def _normalize_text(value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value.strip().lower())

    @classmethod
    def _query_terms(cls, query: str) -> list[str]:
        terms = re.findall(r"[a-z0-9][a-z0-9_\-/.]*", cls._normalize_text(query))
        return [term for term in terms if len(term) >= 2]

    @classmethod
    def _field_match_ratio(cls, field_text: str, query_terms: Sequence[str]) -> float:
        if not field_text or not query_terms:
            return 0.0
        normalized = cls._normalize_text(field_text)
        matched = sum(1 for term in query_terms if term in normalized)
        return matched / len(query_terms)

    def apply_metadata_boosts(self, hits: Sequence[SearchHit], *, query: str, top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        query_terms = self._query_terms(query)
        normalized_query = self._normalize_text(query)
        boosted_hits: list[SearchHit] = []
        for hit in hits:
            boosted = hit.model_copy()
            file_name = Path(hit.file_path).stem
            parent_path = Path(hit.file_path).parent.as_posix()
            filename_ratio = self._field_match_ratio(file_name, query_terms)
            heading_ratio = self._field_match_ratio(hit.heading or "", query_terms)
            tag_ratio = self._field_match_ratio(" ".join(hit.tags), query_terms)
            path_ratio = self._field_match_ratio(parent_path if parent_path != "." else "", query_terms)
            boost = (
                filename_ratio * self.config.search.filename_weight
                + heading_ratio * self.config.search.heading_weight
                + tag_ratio * self.config.search.tag_weight
                + path_ratio * self.config.search.path_weight
            )
            if normalized_query:
                if normalized_query and normalized_query in self._normalize_text(file_name):
                    boost += self.config.search.exact_match_weight
                elif normalized_query in self._normalize_text(hit.heading or ""):
                    boost += self.config.search.exact_match_weight * 0.8
            boosted.score += boost
            boosted_hits.append(boosted)
        boosted_hits.sort(
            key=lambda hit: (
                hit.score,
                hit.semantic_score or 0.0,
                hit.bm25_score or 0.0,
                -hit.chunk_index,
            ),
            reverse=True,
        )
        return boosted_hits[: max(top_k, len(boosted_hits))]

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
        hits = self.apply_metadata_boosts(hits, query=query, top_k=limit)
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
        scoped_history = self._conversation_context(question, conversation_history)
        retrieval_query = self._build_chat_query(question, scoped_history)
        hits = self.search(
            retrieval_query,
            mode=SearchMode.HYBRID,
            top_k=top_k,
            rerank=rerank,
            provider=provider,
        )
        if not hits:
            raise SearchError("No search hits were found for the question.")
        answer = provider.generate_answer(question, hits, scoped_history or None)
        return AnswerResult(
            answer=answer,
            provider=provider.provider_name,
            model=provider.model_name,
            sources=hits,
            context_chunks=[hit.chunk_id for hit in hits],
        )

    def summarize(
        self,
        *,
        focus: str | None = None,
        retrieval_query: str | None = None,
        provider_name: str | None = None,
    ) -> SummaryResult:
        if not self.storage.has_indexed_chunks():
            raise SearchError("Vault has not been indexed yet. Run `local-text-search index` first.")
        provider = build_provider(self.config, provider_name=provider_name)
        summary_chunks = self._summary_chunks(
            provider=provider,
            retrieval_query=retrieval_query,
        )
        file_sections: list[SummarySection] = []
        file_count = 0
        chunk_count = len(summary_chunks)
        llm_calls = 0
        reduction_rounds = 0
        current_file_path: str | None = None
        current_chunks: list[ChunkRecord] = []
        for chunk in summary_chunks:
            if current_file_path is None:
                current_file_path = chunk.file_path
            if chunk.file_path != current_file_path:
                file_summary, file_calls, file_rounds = self._summarize_file_chunks(
                    current_file_path,
                    current_chunks,
                    provider,
                    focus=focus,
                )
                file_sections.append(SummarySection(label=current_file_path, content=file_summary))
                file_count += 1
                llm_calls += file_calls
                reduction_rounds += file_rounds
                current_file_path = chunk.file_path
                current_chunks = [chunk]
                continue
            current_chunks.append(chunk)
        if current_chunks and current_file_path is not None:
            file_summary, file_calls, file_rounds = self._summarize_file_chunks(
                current_file_path,
                current_chunks,
                provider,
                focus=focus,
            )
            file_sections.append(SummarySection(label=current_file_path, content=file_summary))
            file_count += 1
            llm_calls += file_calls
            reduction_rounds += file_rounds
        if not file_sections:
            raise SearchError("No indexed chunks were available to summarize.")
        final_summary, final_calls, final_rounds = self._reduce_summary_sections(
            file_sections,
            provider,
            focus=focus,
            target_words=self.config.summarization.final_summary_words,
            stage="vault-reduce",
            scope_label=f"vault={self.vault.name}",
            allow_passthrough=False,
        )
        return SummaryResult(
            summary=final_summary,
            provider=provider.provider_name,
            model=provider.model_name,
            files_summarized=file_count,
            chunks_summarized=chunk_count,
            llm_calls=llm_calls + final_calls,
            reduction_rounds=reduction_rounds + final_rounds,
            focus=focus,
            retrieval_query=retrieval_query,
        )

    @staticmethod
    def _build_chat_query(
        question: str,
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> str:
        if not conversation_history:
            return question
        recent_user_turns = [turn.content for turn in conversation_history if turn.role == "user"][-2:]
        if not recent_user_turns:
            return question
        return "\n".join([*recent_user_turns, question])

    @classmethod
    def _conversation_context(
        cls,
        question: str,
        conversation_history: Sequence[ChatTurn] | None = None,
    ) -> list[ChatTurn]:
        if not conversation_history or not cls._is_context_dependent_question(question):
            return []
        recent_user_turns = [
            ChatTurn(role="user", content=turn.content)
            for turn in conversation_history
            if turn.role == "user" and cls._normalize_text(turn.content)
        ]
        return recent_user_turns[-2:]

    @classmethod
    def _is_context_dependent_question(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        if not normalized:
            return False
        if any(normalized.startswith(prefix) for prefix in cls.FOLLOW_UP_PREFIXES):
            return True
        tokens = set(re.findall(r"\w+", normalized, flags=re.UNICODE))
        if len(tokens) <= 12 and tokens.intersection(cls.REFERENTIAL_TERMS):
            return True
        if len(normalized.split()) <= 6 and normalized.endswith(("more", "details", "again", "thêm", "nữa", "tiếp")):
            return True
        return False

    @staticmethod
    def _summary_max_tokens(target_words: int) -> int:
        return max(220, min(1400, target_words * 2))

    def _summary_chunks(
        self,
        *,
        provider: BaseProvider,
        retrieval_query: str | None,
    ) -> list[ChunkRecord]:
        if not retrieval_query:
            return list(self.storage.iter_chunks())
        hits = self.search(
            retrieval_query,
            mode=SearchMode.HYBRID,
            top_k=self.config.summarization.retrieval_top_k,
            rerank=self.config.search.rerank_default_ask,
            provider=provider,
        )
        if not hits:
            raise SearchError(f"No search hits were found for retrieval query `{retrieval_query}`.")
        unique_hits: dict[str, ChunkRecord] = {}
        for hit in hits:
            unique_hits[hit.chunk_id] = self._chunk_record_from_hit(hit)
        return sorted(
            unique_hits.values(),
            key=lambda chunk: (chunk.file_path, chunk.chunk_index),
        )

    @staticmethod
    def _chunk_record_from_hit(hit: SearchHit) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=hit.chunk_id,
            file_path=hit.file_path,
            vault_name=hit.vault_name,
            heading=hit.heading,
            tags=list(hit.tags),
            backlinks=list(hit.backlinks),
            modified_time=hit.modified_time,
            chunk_index=hit.chunk_index,
            text=hit.text,
            content_hash=f"search-hit:{hit.chunk_id}",
            token_count=len(hit.text.split()),
            term_frequencies={},
        )

    @staticmethod
    def _chunk_word_count(chunk: ChunkRecord) -> int:
        return max(1, chunk.token_count or len(chunk.text.split()) or 1)

    def _group_chunks_by_word_budget(
        self,
        chunks: Sequence[ChunkRecord],
        *,
        max_words: int,
    ) -> list[list[ChunkRecord]]:
        limit = max(1, max_words)
        groups: list[list[ChunkRecord]] = []
        current_group: list[ChunkRecord] = []
        current_words = 0
        for chunk in chunks:
            chunk_words = self._chunk_word_count(chunk)
            if current_group and current_words + chunk_words > limit:
                groups.append(current_group)
                current_group = [chunk]
                current_words = chunk_words
                continue
            current_group.append(chunk)
            current_words += chunk_words
        if current_group:
            groups.append(current_group)
        return groups

    def _summarize_file_chunks(
        self,
        file_path: str,
        chunks: Sequence[ChunkRecord],
        provider: BaseProvider,
        *,
        focus: str | None,
    ) -> tuple[str, int, int]:
        partial_sections: list[SummarySection] = []
        llm_calls = 0
        for group_index, group in enumerate(
            self._group_chunks_by_word_budget(
                chunks,
                max_words=self.config.summarization.chunk_group_words,
            ),
            start=1,
        ):
            prompt = self._build_chunk_group_summary_prompt(
                file_path=file_path,
                chunks=group,
                group_index=group_index,
                target_words=self.config.summarization.partial_summary_words,
                focus=focus,
            )
            partial_summary = provider.complete(
                prompt,
                max_tokens=self._summary_max_tokens(self.config.summarization.partial_summary_words),
            ).strip()
            partial_sections.append(SummarySection(label=f"{file_path}::part-{group_index}", content=partial_summary))
            llm_calls += 1
        file_summary, reduce_calls, reduce_rounds = self._reduce_summary_sections(
            partial_sections,
            provider,
            focus=focus,
            target_words=self.config.summarization.file_summary_words,
            stage="file-reduce",
            scope_label=f"file={file_path}",
            allow_passthrough=True,
        )
        return file_summary, llm_calls + reduce_calls, reduce_rounds

    def _reduce_summary_sections(
        self,
        sections: Sequence[SummarySection],
        provider: BaseProvider,
        *,
        focus: str | None,
        target_words: int,
        stage: str,
        scope_label: str,
        allow_passthrough: bool,
    ) -> tuple[str, int, int]:
        if not sections:
            raise SearchError("Cannot summarize an empty section set.")
        current_sections = [SummarySection(label=section.label, content=section.content) for section in sections]
        if len(current_sections) == 1 and allow_passthrough:
            return current_sections[0].content, 0, 0
        llm_calls = 0
        reduction_rounds = 0
        batch_size = max(2, self.config.summarization.reduce_group_size)
        while len(current_sections) > 1 or reduction_rounds == 0:
            reduction_rounds += 1
            next_sections: list[SummarySection] = []
            for batch_index in range(0, len(current_sections), batch_size):
                batch = current_sections[batch_index : batch_index + batch_size]
                prompt = self._build_reduce_summary_prompt(
                    sections=batch,
                    target_words=target_words,
                    focus=focus,
                    stage=stage,
                    scope_label=scope_label,
                    round_number=reduction_rounds,
                )
                reduced = provider.complete(
                    prompt,
                    max_tokens=self._summary_max_tokens(target_words),
                ).strip()
                next_sections.append(
                    SummarySection(
                        label=f"{scope_label}::round-{reduction_rounds}-batch-{(batch_index // batch_size) + 1}",
                        content=reduced,
                    )
                )
                llm_calls += 1
            current_sections = next_sections
        return current_sections[0].content, llm_calls, reduction_rounds

    def _build_chunk_group_summary_prompt(
        self,
        *,
        file_path: str,
        chunks: Sequence[ChunkRecord],
        group_index: int,
        target_words: int,
        focus: str | None,
    ) -> str:
        lines = [
            "Summary stage: file-chunks",
            f"Scope: file={file_path}",
            f"Group: {group_index}",
            f"Target length: about {target_words} words.",
            "Task: Summarize the source excerpts into the most important ideas, decisions, claims, and open questions.",
            "Requirements:",
            "- Use concise bullets.",
            "- Preserve nuance, disagreements, and unresolved issues when present.",
            "- Omit repetition and low-signal detail.",
        ]
        if focus:
            lines.append(f"Focus: {focus}")
        lines.append("")
        for index, chunk in enumerate(chunks, start=1):
            heading = chunk.heading or "No heading"
            lines.extend(
                [
                    f"## Chunk {index}",
                    f"source={chunk.source_label}",
                    f"heading={heading}",
                    "content:",
                    chunk.text,
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _build_reduce_summary_prompt(
        self,
        *,
        sections: Sequence[SummarySection],
        target_words: int,
        focus: str | None,
        stage: str,
        scope_label: str,
        round_number: int,
    ) -> str:
        lines = [
            f"Summary stage: {stage}",
            f"Scope: {scope_label}",
            f"Round: {round_number}",
            f"Target length: about {target_words} words.",
        ]
        if stage == "file-reduce":
            lines.append("Task: Combine the partial summaries from one file into a single coherent file summary.")
        else:
            lines.append("Task: Combine the source summaries into a coherent overview of the whole vault.")
        lines.extend(
            [
                "Requirements:",
                "- Synthesize repeated themes instead of concatenating bullets.",
                "- Keep the highest-signal details, distinctions, and tensions.",
                "- Use concise bullets followed by a short closing synthesis sentence.",
            ]
        )
        if focus:
            lines.append(f"Focus: {focus}")
        lines.append("")
        for index, section in enumerate(sections, start=1):
            lines.extend(
                [
                    f"## Summary {index}",
                    f"label={section.label}",
                    section.content,
                    "",
                ]
            )
        return "\n".join(lines).strip()
