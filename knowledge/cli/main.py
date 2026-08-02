"""
Main entrypoint for the KnowledgeOS CLI.
"""

import json
from pathlib import Path
import typer
from rich.console import Console

from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.embeddings import SentenceTransformerEmbedding
from knowledge.loaders.router import LoaderRouter
from knowledge.vectorstores.faiss_store import FAISSStore

app = typer.Typer(
    name="knowledge",
    help="KnowledgeOS: A modular CLI-based Retrieval-Augmented Generation assistant.",
    add_completion=False,
)

console = Console()
router = LoaderRouter()
chunker = RecursiveChunker()
embedder = SentenceTransformerEmbedding()


@app.callback()
def callback() -> None:
    """KnowledgeOS CLI."""
    pass


@app.command("init")
def init() -> None:
    """Initialize the KnowledgeOS workspace."""

    workspace = Path(".knowledge")

    for folder in [
        workspace,
        workspace / "documents",
        workspace / "index",
        workspace / "cache",
        workspace / "logs",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    config = workspace / "config.yaml"

    if not config.exists():
        config.write_text(
            """embedding_model: all-MiniLM-L6-v2
chunk_size: 500
chunk_overlap: 100
top_k: 5
"""
        )

    console.print("[bold green]✓ Workspace initialized successfully![/bold green]")


@app.command("add")
def add(path: str) -> None:
    """Load, chunk and index documents."""

    source = Path(path)

    if not source.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(code=1)

    files = [source] if source.is_file() else [
        f for f in source.rglob("*") if f.is_file()
    ]

    if not files:
        console.print("[yellow]No documents found.[/yellow]")
        return

    vector_store = FAISSStore()

    all_chunks = []
    all_embeddings = []

    console.print(f"[bold green]Found {len(files)} document(s)[/bold green]\n")

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

            embeddings = embedder.embed_documents(texts)

            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

            total_chars = sum(len(doc.text) for doc in loaded_docs)

            console.print(f"[green]✓[/green] {file.name}")
            console.print(f"   Characters : {total_chars:,}")
            console.print(f"   Pages      : {len(loaded_docs)}")
            console.print(f"   Chunks     : {len(chunks)}")
            console.print(f"   Embeddings : {len(embeddings)}")
            console.print()

        except Exception as e:
            console.print(f"[red]✗ {file.name}[/red] - {e}")

    if not all_embeddings:
        console.print("[red]No embeddings generated.[/red]")
        return

    vector_store.add(all_embeddings)
    vector_store.save(".knowledge/index/faiss.index")

    chunk_records = []

    for chunk in all_chunks:
        chunk_records.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
        )

    with open(".knowledge/index/chunks.json", "w") as f:
        json.dump(chunk_records, f, indent=2)

    console.print("[bold green]Index saved successfully.[/bold green]")
    console.print(f"Total Chunks     : {len(all_chunks)}")
    console.print(f"Embedding Size   : {len(all_embeddings[0])}")


@app.command("search")
def search(query: str) -> None:
    """Semantic search."""

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json", "r") as f:
        chunks = json.load(f)

    query_embedding = embedder.embed_query(query)

    scores, indices = vector_store.search(query_embedding, top_k=5)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]

        console.print(f"[green]Score:[/green] {score:.4f}")
        console.print(f"[yellow]Source:[/yellow] {chunk['source_path']}")
        console.print(chunk["text"])
        console.print("-" * 70)


if __name__ == "__main__":
    app()