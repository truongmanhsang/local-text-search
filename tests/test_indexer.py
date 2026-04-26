from __future__ import annotations

from pathlib import Path

from local_text_search.indexer import IndexProgress, Indexer
from local_text_search.storage import VaultStorage


def test_indexer_incremental_and_duplicate_reuse(
    config_with_vault: tuple[object, object, Path],
    fake_embeddings,
) -> None:
    config, vault, vault_dir = config_with_vault
    (vault_dir / "alpha.md").write_text("# Alpha\nsemantic search and local notes\n", encoding="utf-8")
    (vault_dir / "beta.md").write_text("# Alpha\nsemantic search and local notes\n", encoding="utf-8")
    (vault_dir / "plain.txt").write_text("plain text file for indexing", encoding="utf-8")

    storage = VaultStorage(vault, config.qdrant)
    indexer = Indexer(config=config, vault=vault, embedding_client=fake_embeddings, storage=storage)
    try:
        first = indexer.run()
        manifest = storage.load_file_manifest()
        duplicate = manifest["beta.md"]
        assert first.indexed_files == 2
        assert first.reused_files == 1
        assert duplicate.duplicate_of == "alpha.md"
        assert storage.count_chunks() >= 2

        second = indexer.run()
        assert second.skipped_files == 3

        (vault_dir / "alpha.md").write_text("# Alpha\nupdated semantic search note\n", encoding="utf-8")
        third = indexer.run()
        assert third.indexed_files == 1

        (vault_dir / "plain.txt").unlink()
        fourth = indexer.run()
        assert fourth.deleted_files == 1
    finally:
        indexer.close()


def test_chunk_ids_are_stable_for_unchanged_files(
    config_with_vault: tuple[object, object, Path],
    fake_embeddings,
) -> None:
    config, vault, vault_dir = config_with_vault
    file_path = vault_dir / "stable.md"
    file_path.write_text("# Stable\none two three four five six\n", encoding="utf-8")

    storage = VaultStorage(vault, config.qdrant)
    indexer = Indexer(config=config, vault=vault, embedding_client=fake_embeddings, storage=storage)
    try:
        indexer.run()
        first_ids = [chunk.chunk_id for chunk in storage.get_chunks_for_file("stable.md")]
        indexer.run()
        second_ids = [chunk.chunk_id for chunk in storage.get_chunks_for_file("stable.md")]
        assert first_ids == second_ids
    finally:
        indexer.close()


def test_remote_qdrant_uses_vault_scoped_collection_name(config_with_vault) -> None:
    config, vault, _ = config_with_vault
    config.qdrant.location = "remote"
    config.qdrant.url = "http://192.168.1.8:6333/"
    storage = VaultStorage(vault, config.qdrant)
    try:
        assert storage.collection_name == "local_text_search__notes"
    finally:
        storage.close()


def test_index_progress_callback_reports_scan_and_complete(
    config_with_vault: tuple[object, object, Path],
    fake_embeddings,
) -> None:
    config, vault, vault_dir = config_with_vault
    (vault_dir / "alpha.md").write_text("# Alpha\nsemantic search and local notes\n", encoding="utf-8")

    storage = VaultStorage(vault, config.qdrant)
    indexer = Indexer(config=config, vault=vault, embedding_client=fake_embeddings, storage=storage)
    events: list[IndexProgress] = []
    try:
        indexer.run(progress_callback=events.append)
    finally:
        indexer.close()

    assert events
    assert events[0].phase == "scan"
    assert events[-1].phase == "complete"
    assert events[-1].current == events[-1].total
