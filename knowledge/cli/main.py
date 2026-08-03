"""
Main entrypoint for the KnowledgeOS CLI.
"""

import json
from pathlib import Path
import typer
from rich.console import Console
import os
import yaml

from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.embeddings import SentenceTransformerEmbedding
from knowledge.loaders.router import LoaderRouter
from knowledge.vectorstores.faiss_store import FAISSStore
from knowledge.llms.ollama_llm import OllamaLLM
from knowledge.retrievers import BM25Retriever

            

app = typer.Typer(
    name="knowledge",
    help="KnowledgeOS: A modular CLI-based Retrieval-Augmented Generation assistant.",
    add_completion=False,
)

console = Console()
router = LoaderRouter()
chunker = RecursiveChunker()
embedder = SentenceTransformerEmbedding()
llm = OllamaLLM()
bm25 = BM25Retriever()

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
            
            bm25.build(texts)

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

    document_records = []

    for file in files:
        document_records.append(
            {
                "name": file.name,
                "path": str(file.resolve()),
            }
    )

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

    index_data = {
        "documents": document_records,
        "chunks": chunk_records,
    }

    with open(".knowledge/index/chunks.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    console.print("[bold green]Index saved successfully.[/bold green]")
    console.print(f"Total Chunks     : {len(all_chunks)}")
    console.print(f"Embedding Size   : {len(all_embeddings[0])}")


@app.command("search")
def search(query: str) -> None:
    """Semantic search."""

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json", "r") as f:
        data = json.load(f)
    chunks = data["chunks"]

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

@app.command("list")
def list_documents() -> None:
    """List indexed documents."""

    index_file = Path(".knowledge/index/chunks.json")

    if not index_file.exists():
        console.print("[red]No index found.[/red]")
        raise typer.Exit()

    with open(index_file, "r") as f:
        data = json.load(f)

    documents = data["documents"]
    chunks = data["chunks"]

    console.print("\n[bold cyan]Indexed Documents[/bold cyan]\n")

    total_chunks = 0

    for i, doc in enumerate(documents, start=1):

        chunk_count = sum(
            1
            for chunk in chunks
            if Path(chunk["source_path"]).name == doc["name"]
        )

        total_chunks += chunk_count

        console.print(f"{i}. {doc['name']}")
        console.print(f"   Path   : {doc['path']}")
        console.print(f"   Chunks : {chunk_count}")
        console.print()

    console.print(f"Total Documents : {len(documents)}")
    console.print(f"Total Chunks    : {total_chunks}")

@app.command("stats")
def stats() -> None:
    """Show KnowledgeOS statistics."""

    workspace = Path(".knowledge")
    config_path = workspace / "config.yaml"
    index_path = workspace / "index" / "faiss.index"
    chunks_path = workspace / "index" / "chunks.json"

    if not chunks_path.exists():
        console.print("[red]No index found.[/red]")
        raise typer.Exit()

    with open(chunks_path, "r") as f:
        data = json.load(f)

    documents = data["documents"]
    chunks = data["chunks"]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    embedding_dim = "Unknown"

    if chunks:
        vector_store = FAISSStore()
        vector_store.load(str(index_path))

        if vector_store.index is not None:
            embedding_dim = vector_store.index.d

    index_size = (
        f"{os.path.getsize(index_path)/1024:.2f} KB"
        if index_path.exists()
        else "0 KB"
    )

    chunk_size = (
        f"{os.path.getsize(chunks_path)/1024:.2f} KB"
        if chunks_path.exists()
        else "0 KB"
    )

    console.print("\n[bold cyan]KnowledgeOS Statistics[/bold cyan]\n")

    console.print(f"Workspace        : {workspace.resolve()}")
    console.print(f"Documents        : {len(documents)}")
    console.print(f"Chunks           : {len(chunks)}")
    console.print(f"Embedding Model  : {config['embedding_model']}")
    console.print(f"Embedding Size   : {embedding_dim}")
    console.print("Vector Store     : FAISS")
    console.print(f"Index Size       : {index_size}")
    console.print(f"Chunk File Size  : {chunk_size}")

@app.command("ask")
def ask(question: str) -> None:
    """Ask a question about indexed documents."""

    import json

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json") as f:
        data = json.load(f)

    chunks = data["chunks"]

    query_embedding = embedder.embed_query(question)

    scores, indices = vector_store.search(query_embedding, top_k=5)

    context = []

    for idx in indices[0]:
        if idx == -1:
            continue

        context.append(chunks[idx]["text"])

    answer = llm.generate(
        question,
        "\n\n".join(context),
    )

    console.print("\n[bold green]Answer[/bold green]\n")
    console.print(answer)

    console.print("\n[bold cyan]Sources[/bold cyan]")

    seen = set()

    for idx in indices[0]:
        if idx == -1:
            continue

        source = chunks[idx]["source_path"]

        if source not in seen:
            seen.add(source)
            console.print(f"- {source}")
if __name__ == "__main__":
    app()