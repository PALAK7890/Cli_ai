"""
Main entrypoint for the KnowledgeOS CLI.
"""

import typer
from rich.console import Console

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
    """Initialize KnowledgeOS workspace and default configuration."""
    console.print("[bold green]Initializing KnowledgeOS...[/bold green]")
    # Placeholder for configuration initialization
    console.print("[green]Workspace initialized successfully![/green]")


if __name__ == "__main__":
    app()
