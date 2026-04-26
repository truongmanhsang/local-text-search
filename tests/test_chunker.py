from __future__ import annotations

from local_text_search.chunker import chunk_document, split_markdown_sections


def test_markdown_chunking_preserves_headings_tags_and_backlinks() -> None:
    text = """---
tags:
  - project
  - research
---
# Overview
Alpha beta gamma delta epsilon zeta eta theta.
Reference [[Second Note]] and #inline-tag.

## Details
More content for the second section.
"""
    chunks = chunk_document(text, is_markdown=True, chunk_size_words=5, chunk_overlap_words=2)

    assert len(chunks) >= 3
    assert chunks[0].heading == "Overview"
    assert chunks[-1].heading == "Overview / Details"
    assert "Second Note" in chunks[0].backlinks
    assert "project" in chunks[0].tags
    assert "inline-tag" in chunks[0].tags


def test_chunk_overlap_repeats_tail_words() -> None:
    text = "# Heading\none two three four five six seven eight"
    chunks = chunk_document(text, is_markdown=True, chunk_size_words=4, chunk_overlap_words=2)
    assert chunks[0].text.endswith("one two three four")
    assert chunks[1].text.endswith("three four five six")


def test_split_markdown_sections_handles_nested_headings() -> None:
    sections = split_markdown_sections(
        "# Parent\nBody one\n\n## Child\nBody two\n\n### Grandchild\nBody three\n"
    )
    headings = [section.heading for section in sections]
    assert headings == ["Parent", "Parent / Child", "Parent / Child / Grandchild"]
