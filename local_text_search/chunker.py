from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import yaml

from local_text_search.models import ChunkDraft

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
BACKLINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
INLINE_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9][A-Za-z0-9_\-/]*)")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(slots=True)
class Section:
    heading_path: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def heading(self) -> str | None:
        if not self.heading_path:
            return None
        return " / ".join(self.heading_path)


@dataclass(slots=True)
class ParsedDocument:
    body: str
    tags: list[str]
    backlinks: list[str]


def _normalize_tag(value: str) -> str:
    return value.strip().lstrip("#").strip()


def parse_frontmatter(text: str) -> ParsedDocument:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return ParsedDocument(body=text, tags=[], backlinks=extract_backlinks(text))
    frontmatter = match.group(1)
    body = text[match.end() :]
    tags: list[str] = []
    try:
        parsed = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        parsed = {}
    raw_tags = parsed.get("tags", []) if isinstance(parsed, dict) else []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if isinstance(raw_tags, Iterable):
        for tag in raw_tags:
            if isinstance(tag, str):
                normalized = _normalize_tag(tag)
                if normalized:
                    tags.append(normalized)
    return ParsedDocument(body=body, tags=sorted(set(tags)), backlinks=extract_backlinks(body))


def extract_backlinks(text: str) -> list[str]:
    return sorted({match.strip() for match in BACKLINK_RE.findall(text) if match.strip()})


def extract_inline_tags(text: str) -> list[str]:
    return sorted({_normalize_tag(match) for match in INLINE_TAG_RE.findall(text) if _normalize_tag(match)})


def split_markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading_path: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(Section(heading_path=current_heading_path.copy(), body=content))
        current_lines.clear()

    for raw_line in text.splitlines():
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            current_heading_path = heading_stack.copy()
            continue
        current_lines.append(raw_line)

    flush()
    return sections or [Section(body=text.strip())]


def split_plain_text(text: str) -> list[Section]:
    stripped = text.strip()
    if not stripped:
        return []
    return [Section(body=stripped)]


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _slice_words(text: str, start: int, end: int) -> str:
    words = text.split()
    return " ".join(words[start:end])


def _chunk_section_text(section: Section, chunk_size_words: int, overlap_words: int) -> list[ChunkDraft]:
    body = section.body.strip()
    if not body:
        return []
    words = body.split()
    if not words:
        return []
    drafts: list[ChunkDraft] = []
    cursor = 0
    chunk_index = 0
    overlap = max(0, min(overlap_words, max(chunk_size_words - 1, 0)))
    while cursor < len(words):
        end = min(cursor + chunk_size_words, len(words))
        content = _slice_words(body, cursor, end).strip()
        if section.heading:
            text = f"{section.heading}\n\n{content}"
        else:
            text = content
        drafts.append(
            ChunkDraft(
                heading=section.heading,
                text=text,
                chunk_index=chunk_index,
                token_count=_count_words(content),
            )
        )
        chunk_index += 1
        if end >= len(words):
            break
        cursor = end - overlap
    return drafts


def chunk_document(
    text: str,
    *,
    is_markdown: bool,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[ChunkDraft]:
    parsed = parse_frontmatter(text)
    base_tags = set(parsed.tags)
    inline_tags = set(extract_inline_tags(parsed.body))
    backlinks = parsed.backlinks
    sections = split_markdown_sections(parsed.body) if is_markdown else split_plain_text(parsed.body)
    output: list[ChunkDraft] = []
    next_index = 0
    for section in sections:
        for draft in _chunk_section_text(section, chunk_size_words, chunk_overlap_words):
            draft.chunk_index = next_index
            draft.tags = sorted(base_tags | inline_tags)
            draft.backlinks = backlinks
            output.append(draft)
            next_index += 1
    return output
