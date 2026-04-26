# local-text-search

`local-text-search` is a production-oriented Python CLI for local semantic search across markdown and text note folders such as Obsidian vaults. It indexes files recursively, preserves markdown structure and metadata, stores dense and sparse vectors in Qdrant local mode or a remote Qdrant server, and supports answer generation through Ollama, OpenAI-compatible APIs, or Anthropic.

## Features

- Typer-based CLI with Rich output
- Recursive indexing for `.md`, `.markdown`, and `.txt`
- Heading-aware chunking with overlap
- Incremental indexing using `size + mtime` quick checks and content-hash dedupe
- Duplicate-file reuse to avoid redundant embedding calls
- Hybrid retrieval with lexical sparse vectors and dense embeddings
- Answer mode with source citations
- Global config at `~/.local-text-search/config.toml`
- Qdrant local mode by default, with optional remote Qdrant connection

## Installation

1. Create a Python 3.12+ environment.
2. Install the package:

```bash
pip install -e .
```

To install the CLI globally with `pipx`:

```bash
pipx install /path/to/local-text-search
```

From this repo checkout, for example:

```bash
pipx install .
```

Or install runtime dependencies directly:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

## Configuration

Print the example configuration:

```bash
local-text-search config --example
```

The default config file location is:

```bash
~/.local-text-search/config.toml
```

The default embedding setup is offline-first:

- Embeddings: Ollama `nomic-embed-text`
- Answers: Ollama, unless you change `providers.default_provider`
- Vectors: local Qdrant, unless you change `qdrant.location` to `remote`

Remote Qdrant example:

```toml
[qdrant]
location = "remote"
url = "http://192.168.1.8:6333/"
api_key_env = "QDRANT_API_KEY"
timeout_seconds = 10
```

If your server does not require authentication, you can omit `QDRANT_API_KEY`.

## CLI Usage

Initialize a vault:

```bash
local-text-search init ~/Documents/Notes
```

Index the active vault:

```bash
local-text-search index
```

Search with hybrid retrieval:

```bash
local-text-search search "distributed systems" --mode hybrid --top-k 8
```

Run lexical-only search:

```bash
local-text-search search "exact phrase" --mode bm25
```

Ask a question with cited sources:

```bash
local-text-search ask "What did I write about retrieval augmented generation?"
```

Start an interactive retrieval chat:

```bash
local-text-search ask
```

Or explicitly:

```bash
local-text-search ask --interactive
```

Rebuild an index from scratch:

```bash
local-text-search reindex --yes
```

Inspect the resolved config:

```bash
local-text-search config
```

## Provider Notes

- OpenAI-compatible providers use `base_url`, `api_key`, and `model`.
- Anthropic uses `ANTHROPIC_API_KEY` unless `api_key` is set in config.
- Ollama uses local HTTP endpoints and works offline once models are available.

## Example Workflow

```bash
local-text-search init ~/vaults/research --name research
local-text-search index --vault research
local-text-search search "vector databases"
local-text-search ask "What notes mention hybrid search and reranking?"
```
