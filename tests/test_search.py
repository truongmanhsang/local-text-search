from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_text_search.config import AppConfig, VaultConfig
from local_text_search.models import ChatTurn, SearchHit, SearchMode
from local_text_search.search import SearchService


@dataclass
class StubStorage:
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"

    class Encoder:
        @staticmethod
        def encode_query(text: str) -> tuple[list[int], list[float]]:
            return [1], [1.0]

    encoder = Encoder()

    def has_indexed_chunks(self) -> bool:
        return True

    def query(self, *, query_vector, using: str, vault_name: str, limit: int):
        if using == self.dense_vector_name:
            return [
                SearchHit(
                    chunk_id="a",
                    score=0.9,
                    file_path="a.md",
                    vault_name=vault_name,
                    heading="Alpha",
                    text="alpha semantic result",
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=SearchMode.SEMANTIC,
                ),
                SearchHit(
                    chunk_id="b",
                    score=0.4,
                    file_path="b.md",
                    vault_name=vault_name,
                    heading="Beta",
                    text="beta semantic result",
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=SearchMode.SEMANTIC,
                ),
            ]
        return [
            SearchHit(
                chunk_id="b",
                score=1.2,
                file_path="b.md",
                vault_name=vault_name,
                heading="Beta",
                text="beta lexical result",
                modified_time=0.0,
                chunk_index=0,
                search_mode=SearchMode.BM25,
            ),
            SearchHit(
                chunk_id="c",
                score=0.8,
                file_path="c.md",
                vault_name=vault_name,
                heading="Gamma",
                text="gamma lexical result",
                modified_time=0.0,
                chunk_index=0,
                search_mode=SearchMode.BM25,
            ),
        ]

    def close(self) -> None:
        return None


class StubProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_history = None

    def rerank(self, query: str, candidates):
        return ["b", "a"]

    def generate_answer(self, question: str, context_chunks, conversation_history=None):
        self.last_history = conversation_history
        return "Answer with [1]"


def test_hybrid_merge_normalizes_and_reranks(fake_embeddings) -> None:
    config = AppConfig()
    vault = VaultConfig(name="notes", path=Path("/tmp/notes"))
    service = SearchService(config=config, vault=vault, storage=StubStorage(), embedding_client=fake_embeddings)
    try:
        hits = service.search("query", mode=SearchMode.HYBRID, top_k=3, rerank=True, provider=StubProvider())
    finally:
        service.close()
    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]
    assert hits[0].search_mode == SearchMode.HYBRID


def test_ask_returns_sources_and_provider_metadata(fake_embeddings) -> None:
    config = AppConfig()
    vault = VaultConfig(name="notes", path=Path("/tmp/notes"))
    service = SearchService(config=config, vault=vault, storage=StubStorage(), embedding_client=fake_embeddings)
    original = __import__("local_text_search.search", fromlist=["build_provider"])
    previous = original.build_provider
    original.build_provider = lambda config, provider_name=None: StubProvider()
    try:
        result = service.ask("question", top_k=2, rerank=False)
    finally:
        original.build_provider = previous
        service.close()
    assert result.provider == "stub"
    assert result.model == "stub-model"
    assert len(result.sources) == 2


def test_ask_uses_history_for_prompt_and_query(fake_embeddings) -> None:
    config = AppConfig()
    vault = VaultConfig(name="notes", path=Path("/tmp/notes"))
    service = SearchService(config=config, vault=vault, storage=StubStorage(), embedding_client=fake_embeddings)
    stub_provider = StubProvider()
    original = __import__("local_text_search.search", fromlist=["build_provider"])
    previous = original.build_provider
    original.build_provider = lambda config, provider_name=None: stub_provider
    try:
        result = service.ask(
            "What about follow ups?",
            top_k=2,
            rerank=False,
            conversation_history=[ChatTurn(role="user", content="Tell me about alpha")],
        )
    finally:
        original.build_provider = previous
        service.close()
    assert result.answer == "Answer with [1]"
    assert stub_provider.last_history is not None
    assert stub_provider.last_history[0].content == "Tell me about alpha"
