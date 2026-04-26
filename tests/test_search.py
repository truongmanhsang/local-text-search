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
    last_query_texts: list[str] = None

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
                    backlinks=["Beta"],
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
                    backlinks=[],
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
                backlinks=[],
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
                backlinks=[],
                modified_time=0.0,
                chunk_index=0,
                search_mode=SearchMode.BM25,
            ),
        ]

    def load_note_alias_index(self):
        return {"beta": {"beta.md"}, "a": {"a.md"}, "alpha": {"a.md"}}

    def normalize_note_reference(self, value: str) -> str:
        return value.lower().strip()

    def get_search_hits_for_file_paths(self, file_paths, *, per_file_limit=2, search_mode=SearchMode.HYBRID):
        hits = []
        if "beta.md" in file_paths:
            hits.append(
                SearchHit(
                    chunk_id="b-linked",
                    score=0.0,
                    file_path="beta.md",
                    vault_name="notes",
                    heading="Beta Linked",
                    text="beta linked note result",
                    backlinks=[],
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=search_mode,
                )
            )
        return hits

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
    assert [hit.chunk_id for hit in hits] == ["b", "a", "b-linked"]
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


def test_wikilink_expansion_adds_linked_note_hits(fake_embeddings) -> None:
    config = AppConfig()
    config.search.wikilink_weight = 0.2
    vault = VaultConfig(name="notes", path=Path("/tmp/notes"))
    service = SearchService(config=config, vault=vault, storage=StubStorage(), embedding_client=fake_embeddings)
    try:
        hits = service.search("query", mode=SearchMode.HYBRID, top_k=4, rerank=False)
    finally:
        service.close()
    assert any(hit.chunk_id == "b-linked" for hit in hits)
    linked_hit = next(hit for hit in hits if hit.chunk_id == "b-linked")
    assert linked_hit.score > 0.0


def test_metadata_boosts_improve_heading_and_filename_matches(fake_embeddings) -> None:
    config = AppConfig()
    config.search.wikilink_anchor_count = 0
    vault = VaultConfig(name="notes", path=Path("/tmp/notes"))

    @dataclass
    class MetadataBoostStorage(StubStorage):
        def query(self, *, query_vector, using: str, vault_name: str, limit: int):
            if using == self.dense_vector_name:
                return [
                    SearchHit(
                        chunk_id="low",
                        score=0.6,
                        file_path="low.md",
                        vault_name=vault_name,
                        heading="Low",
                        text="low semantic result",
                        backlinks=[],
                        modified_time=0.0,
                        chunk_index=0,
                        search_mode=SearchMode.SEMANTIC,
                    ),
                    SearchHit(
                        chunk_id="generic",
                        score=0.9,
                        file_path="random.md",
                        vault_name=vault_name,
                        heading="Misc",
                        text="generic semantic result",
                        backlinks=[],
                        modified_time=0.0,
                        chunk_index=0,
                        search_mode=SearchMode.SEMANTIC,
                    ),
                    SearchHit(
                        chunk_id="target",
                        score=0.84,
                        file_path="how-to-build-gpt-model.md",
                        vault_name=vault_name,
                        heading="How to build GPT model",
                        text="target semantic result",
                        backlinks=[],
                        modified_time=0.0,
                        chunk_index=0,
                        search_mode=SearchMode.SEMANTIC,
                    ),
                ]
            return [
                SearchHit(
                    chunk_id="low",
                    score=0.5,
                    file_path="low.md",
                    vault_name=vault_name,
                    heading="Low",
                    text="low lexical result",
                    backlinks=[],
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=SearchMode.BM25,
                ),
                SearchHit(
                    chunk_id="generic",
                    score=1.0,
                    file_path="random.md",
                    vault_name=vault_name,
                    heading="Misc",
                    text="generic lexical result",
                    backlinks=[],
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=SearchMode.BM25,
                ),
                SearchHit(
                    chunk_id="target",
                    score=0.94,
                    file_path="how-to-build-gpt-model.md",
                    vault_name=vault_name,
                    heading="How to build GPT model",
                    text="target lexical result",
                    backlinks=[],
                    modified_time=0.0,
                    chunk_index=0,
                    search_mode=SearchMode.BM25,
                ),
            ]

    service = SearchService(
        config=config,
        vault=vault,
        storage=MetadataBoostStorage(),
        embedding_client=fake_embeddings,
    )
    try:
        hits = service.search("how to build gpt model", mode=SearchMode.HYBRID, top_k=2, rerank=False)
    finally:
        service.close()
    assert [hit.chunk_id for hit in hits] == ["target", "generic"]
