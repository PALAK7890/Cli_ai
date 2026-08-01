"""
Main entrypoint for the KnowledgeOS CLI.
"""

import typer
from rich.console import Console
from pathlib import Path
from knowledge.loaders.router import LoaderRouter
router = LoaderRouter()
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
            text = loader.load(file)

            console.print(
                f"[green]✓[/green] {file.name} "
                f"({len(text):,} characters)"
            )

        except Exception as e:
            console.print(
                f"[red]✗[/red] {file.name} - {e}"
            )

if __name__ == "__main__":
    app()
