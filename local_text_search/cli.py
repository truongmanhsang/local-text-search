from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from local_text_search import __version__
from local_text_search.config import (
    EXAMPLE_CONFIG,
    AppConfig,
    VaultConfig,
    ensure_app_dirs,
    get_config_path,
    load_config,
    provider_readiness,
    register_vault,
    save_config,
)
from local_text_search.indexer import IndexProgress, Indexer
from local_text_search.models import SearchMode
from local_text_search.providers.base import ProviderError, build_provider
from local_text_search.search import SearchError, SearchService

app = typer.Typer(help="Local semantic search for markdown and text note folders.")
console = Console()


def slugify_name(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "vault"


def resolve_config() -> AppConfig:
    ensure_app_dirs()
    return load_config(create_if_missing=True)


def resolve_vault(config: AppConfig, vault_name: str | None) -> VaultConfig:
    return config.require_vault(vault_name)


def render_search_results(results: list, title: str) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Source", style="magenta")
    table.add_column("Excerpt", style="white")
    for index, hit in enumerate(results, start=1):
        score = f"{hit.score:.3f}"
        table.add_row(str(index), score, hit.source_label, hit.excerpt())
    console.print(table)


def exit_with_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


def run_index_with_progress(indexer: Indexer, *, full_rebuild: bool) -> object:
    description = "Reindexing vault" if full_rebuild else "Indexing vault"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=1)

        def on_progress(event: IndexProgress) -> None:
            total = max(event.total, 1)
            if progress.tasks[task_id].total != total:
                progress.update(task_id, total=total)
            if event.path:
                label = f"{event.action}: {event.path}"
            else:
                label = event.action
            progress.update(
                task_id,
                completed=min(event.current, total),
                description=f"{description} [{label}]",
            )

        return indexer.run(full_rebuild=full_rebuild, progress_callback=on_progress)


@app.callback()
def main() -> None:
    """local-text-search CLI."""


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


@app.command()
def init(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    name: str | None = typer.Option(None, "--name", help="Optional vault name."),
) -> None:
    """Register a folder as a searchable vault."""
    config = resolve_config()
    folder = folder.expanduser().resolve()
    vault_name = name or slugify_name(folder.name)
    register_vault(config, vault_name, folder)
    save_config(config)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold green]Vault registered[/bold green]",
                    f"name: [cyan]{vault_name}[/cyan]",
                    f"path: [white]{folder}[/white]",
                    f"config: [white]{get_config_path()}[/white]",
                ]
            )
        )
    )


@app.command()
def index(vault: str | None = typer.Option(None, "--vault", help="Vault name to index.")) -> None:
    """Index new or changed files in the selected vault."""
    config = resolve_config()
    try:
        selected_vault = resolve_vault(config, vault)
    except ValueError as exc:
        exit_with_error(str(exc))
    indexer = Indexer(config=config, vault=selected_vault)
    try:
        stats = run_index_with_progress(indexer, full_rebuild=False)
    except Exception as exc:
        exit_with_error(str(exc))
    finally:
        indexer.close()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]Index complete[/bold green]",
                    f"scanned files: {stats.scanned_files}",
                    f"indexed files: {stats.indexed_files}",
                    f"reused duplicates: {stats.reused_files}",
                    f"skipped files: {stats.skipped_files}",
                    f"deleted files: {stats.deleted_files}",
                    f"chunks upserted: {stats.chunks_upserted}",
                ]
            )
        )
    )


@app.command()
def reindex(
    vault: str | None = typer.Option(None, "--vault", help="Vault name to rebuild."),
    yes: bool = typer.Option(False, "--yes", help="Confirm the destructive rebuild."),
) -> None:
    """Rebuild the selected vault from scratch."""
    if not yes:
        raise typer.BadParameter("Reindex requires --yes to confirm rebuilding the vault index.")
    config = resolve_config()
    try:
        selected_vault = resolve_vault(config, vault)
    except ValueError as exc:
        exit_with_error(str(exc))
    indexer = Indexer(config=config, vault=selected_vault)
    try:
        stats = run_index_with_progress(indexer, full_rebuild=True)
    except Exception as exc:
        exit_with_error(str(exc))
    finally:
        indexer.close()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold yellow]Full reindex complete[/bold yellow]",
                    f"scanned files: {stats.scanned_files}",
                    f"indexed files: {stats.indexed_files}",
                    f"reused duplicates: {stats.reused_files}",
                    f"deleted files: {stats.deleted_files}",
                    f"chunks upserted: {stats.chunks_upserted}",
                ]
            )
        )
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Query string."),
    vault: str | None = typer.Option(None, "--vault", help="Vault name."),
    mode: SearchMode = typer.Option(SearchMode.HYBRID, "--mode", case_sensitive=False),
    top_k: int | None = typer.Option(None, "--top-k", help="Number of results to return."),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="Enable provider reranking."),
    provider: str | None = typer.Option(None, "--provider", help="Provider used for reranking."),
) -> None:
    """Run lexical, semantic, or hybrid search."""
    config = resolve_config()
    try:
        selected_vault = resolve_vault(config, vault)
    except ValueError as exc:
        exit_with_error(str(exc))
    service = SearchService(config=config, vault=selected_vault)
    try:
        rerank_provider = build_provider(config, provider) if provider else None
        results = service.search(
            query,
            mode=mode,
            top_k=top_k,
            rerank=config.search.rerank_default_search if rerank is None else rerank,
            provider=rerank_provider,
        )
    except (SearchError, ProviderError, Exception) as exc:
        exit_with_error(str(exc))
    finally:
        service.close()
    render_search_results(results, title=f"Search Results ({mode.value})")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    vault: str | None = typer.Option(None, "--vault", help="Vault name."),
    top_k: int | None = typer.Option(None, "--top-k", help="Context chunk count."),
    provider: str | None = typer.Option(None, "--provider", help="Answer provider."),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="Enable provider reranking."),
) -> None:
    """Answer a question using retrieved vault context."""
    config = resolve_config()
    try:
        selected_vault = resolve_vault(config, vault)
    except ValueError as exc:
        exit_with_error(str(exc))
    service = SearchService(config=config, vault=selected_vault)
    try:
        result = service.ask(
            question,
            top_k=top_k,
            provider_name=provider,
            rerank=config.search.rerank_default_ask if rerank is None else rerank,
        )
    except (SearchError, ProviderError, Exception) as exc:
        exit_with_error(str(exc))
    finally:
        service.close()
    console.print(Panel(result.answer, title=f"Answer via {result.provider}:{result.model}", expand=False))
    render_search_results(result.sources, title="Sources")


@app.command()
def config(
    path: bool = typer.Option(False, "--path", help="Only print the config path."),
    example: bool = typer.Option(False, "--example", help="Print an example config."),
) -> None:
    """Inspect the resolved configuration."""
    if path:
        console.print(str(get_config_path()))
        return
    if example:
        console.print(EXAMPLE_CONFIG.rstrip())
        return
    loaded = resolve_config()
    summary = Table(title="local-text-search config")
    summary.add_column("Setting", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("config_path", str(get_config_path()))
    summary.add_row("active_vault", loaded.active_vault or "<none>")
    summary.add_row("qdrant_location", loaded.qdrant.location)
    summary.add_row("qdrant_url", loaded.qdrant.url or "<local mode>")
    summary.add_row("embedding_provider", loaded.embeddings.default_provider)
    summary.add_row("answer_provider", loaded.providers.default_provider)
    console.print(summary)

    vault_table = Table(title="Registered vaults")
    vault_table.add_column("Name", style="magenta")
    vault_table.add_column("Path", style="white")
    if loaded.vaults:
        for vault_config in loaded.vaults:
            vault_table.add_row(vault_config.name, str(vault_config.path))
    else:
        vault_table.add_row("<none>", "<none>")
    console.print(vault_table)

    readiness = Table(title="Provider readiness")
    readiness.add_column("Provider", style="green")
    readiness.add_column("Ready", style="white")
    for provider_name, ready in provider_readiness(loaded).items():
        readiness.add_row(provider_name, "yes" if ready else "no")
    console.print(readiness)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
