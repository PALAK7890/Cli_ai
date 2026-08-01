"""
Main entrypoint for the KnowledgeOS CLI.
"""

import typer
from rich.console import Console
from pathlib import Path

from knowledge.loaders.router import LoaderRouter
from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.loaders.base import LoadedDocument
from knowledge.embeddings import SentenceTransformerEmbedding

router = LoaderRouter()
chunker = RecursiveChunker()
embedder = SentenceTransformerEmbedding()

app = typer.Typer(
    name="knowledge",
    help="KnowledgeOS: A modular CLI-based Retrieval-Augmented Generation assistant.",
    add_completion=False,
)
console = Console()


@app.callback()
def callback() -> None:
    """
    KnowledgeOS - CLI RAG Assistant.
    """
    pass


@app.command("init")
def init() -> None:
    """Initialize the KnowledgeOS workspace."""

    console.print("[bold green]Initializing KnowledgeOS...[/bold green]")

    workspace = Path(".knowledge")

    directories = [
        workspace,
        workspace / "documents",
        workspace / "index",
        workspace / "cache",
        workspace / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    config_file = workspace / "config.yaml"

    if not config_file.exists():
        config_file.write_text(
            """embedding_model: all-MiniLM-L6-v2
chunk_size: 500
chunk_overlap: 100
top_k: 5
"""
        )

    console.print("[bold green]✓ Workspace initialized successfully![/bold green]")
    console.print(f"[cyan]Workspace:[/cyan] {workspace.resolve()}")

@app.command("add")
def add(path: str) -> None:
    """Add documents to the KnowledgeOS workspace."""

    source = Path(path)

    if not source.exists():
        console.print(f"[bold red]Error:[/bold red] '{path}' does not exist.")
        raise typer.Exit(code=1)

    if source.is_file():
        files = [source]
    else:
        files = [file for file in source.rglob("*") if file.is_file()]

    if not files:
        console.print("[yellow]No files found.[/yellow]")
        raise typer.Exit()

    console.print(f"[bold green]Found {len(files)} document(s):[/bold green]")

    for file in files:
        try:
            loader = router.get_loader(file)
            loaded_docs = loader.load(file)

            chunks = chunker.chunk(
                loaded_docs,
                chunk_size=500,
                chunk_overlap=100,
            )
            texts = [chunk.text for chunk in chunks]

            embeddings = embedder.encode(texts)

            total_chars = sum(len(doc.text) for doc in loaded_docs)

            console.print(f"[green]✓[/green] {file.name}")
            console.print(f"   Characters : {total_chars:,}")
            console.print(f"   Pages       : {len(loaded_docs)}")
            console.print(f"   Chunks      : {len(chunks)}")
            console.print(f"   Embeddings  : {len(embeddings)}")
            if embeddings:
                console.print(f"   Dimension   : {len(embeddings[0])}")

        except Exception as e:
            console.print(
                f"[red]✗[/red] {file.name} - {e}"
            )

if __name__ == "__main__":
    app()
