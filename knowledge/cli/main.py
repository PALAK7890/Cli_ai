"""
Main entrypoint for the KnowledgeOS CLI.
"""

import json
from pathlib import Path
import typer
from rich.console import Console
import os
import yaml
import time

from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.embeddings import SentenceTransformerEmbedding
from knowledge.loaders.router import LoaderRouter
from knowledge.vectorstores.faiss_store import FAISSStore
from knowledge.llms.ollama_llm import OllamaLLM
from knowledge.retrievers import BM25Retriever, HybridRetriever
from knowledge.rerankers import CrossEncoderReranker
from knowledge.query import QueryExpander

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
reranker = CrossEncoderReranker()
expander = QueryExpander()

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
    """Hybrid search with CrossEncoder reranking."""

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json", "r") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [chunk["text"] for chunk in chunks]

    bm25 = BM25Retriever()
    bm25.build(texts)

    expanded_query = expander.expand(query)

    console.print(
        f"[cyan]Expanded Query:[/cyan] {expanded_query}"
    )
#fiass
    query_embedding = embedder.embed_query(query)

    scores, indices = vector_store.search(
        query_embedding,
        top_k=20,
    )

    faiss_results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        faiss_results.append((idx, float(score)))
    # BM25
    bm25_results = bm25.search(
        expanded_query,
        top_k=20,
    )

    # hybrid retrieval

    hybrid = HybridRetriever()

    results = hybrid.fuse(
        faiss_results,
        bm25_results,
    )


    # ce reranking

    candidate_chunks = []

    for idx, hybrid_score in results:

        chunk = chunks[idx].copy()

        chunk["hybrid_score"] = hybrid_score

        candidate_chunks.append(chunk)

    ranked = reranker.rerank(
        query=query,
        chunks=candidate_chunks,
        top_k=5,
    )

    # display
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    for i, chunk in enumerate(ranked, start=1):

        console.print(f"[bold]{i}.[/bold] {Path(chunk['source_path']).name}")

        console.print(
            f"[green]Cross Score :[/green] "
            f"{chunk['rerank_score']:.4f}"
        )

        console.print(
            f"[blue]Hybrid Score:[/blue] "
            f"{chunk['hybrid_score']:.4f}"
        )

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
def ask(question: str) ->None:
    """Answer questions using Hybrid Retrieval + CrossEncoder reranking."""

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [chunk["text"] for chunk in chunks]

    bm25 = BM25Retriever()
    bm25.build(texts)
    expanded_question = expander.expand(question)

    console.print(
        f"[cyan]Expanded Query:[/cyan] {expanded_question}"
    )

    query_embedding = embedder.embed_query(question)

    scores, indices = vector_store.search(
        query_embedding,
        top_k=20,
    )

    faiss_results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        faiss_results.append((idx, float(score)))


    bm25_results = bm25.search(
        expanded_question,
        top_k=20,
    )

#hybrid searching
    hybrid = HybridRetriever()

    results = hybrid.fuse(
        faiss_results,
        bm25_results,
    )

#cross encoder
    candidate_chunks = []

    for idx, hybrid_score in results:

        chunk = chunks[idx].copy()

        chunk["hybrid_score"] = hybrid_score

        candidate_chunks.append(chunk)

    ranked = reranker.rerank(
        question,
        candidate_chunks,
        top_k=5,
    )

#building context
    context = "\n\n".join(
        chunk["text"]
        for chunk in ranked
    )

    answer = llm.generate(
        question,
        context,
    )

    console.print("\n[bold green]Answer[/bold green]\n")

    console.print(answer)

    console.print("\n[bold cyan]Retrieved Sources[/bold cyan]\n")

    for chunk in ranked:

        console.print(
            f"• {Path(chunk['source_path']).name}"
        )

        console.print(
            f"  Cross Score : {chunk['rerank_score']:.4f}"
        )

        console.print(
            f"  Hybrid Score: {chunk['hybrid_score']:.4f}\n"
        )
@app.command("benchmark")
def benchmark(question: str) -> None:
    """Benchmark the complete KnowledgeOS retrieval pipeline."""

    total_start = time.perf_counter()

    vector_store = FAISSStore()
    vector_store.load(".knowledge/index/faiss.index")

    with open(".knowledge/index/chunks.json") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["text"] for c in chunks]

    bm25 = BM25Retriever()
    bm25.build(texts)

    hybrid = HybridRetriever()

#query expansion
    expanded_query = expander.expand(question)

#embeeding
    start = time.perf_counter()

    query_embedding = embedder.embed_query(expanded_query)

    embedding_time = (time.perf_counter() - start) * 1000

#faiss
    start = time.perf_counter()

    scores, indices = vector_store.search(
        query_embedding,
        top_k=20,
    )

    faiss_time = (time.perf_counter() - start) * 1000

    faiss_results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        faiss_results.append((idx, float(score)))


    start = time.perf_counter()

    bm25_results = bm25.search(
        expanded_query,
        top_k=20,
    )

    bm25_time = (time.perf_counter() - start) * 1000

#hybrid fusion
    start = time.perf_counter()

    fused = hybrid.fuse(
        faiss_results,
        bm25_results,
    )

    hybrid_time = (time.perf_counter() - start) * 1000

    candidate_chunks = []

    for idx, score in fused:

        chunk = chunks[idx].copy()
        chunk["hybrid_score"] = score

        candidate_chunks.append(chunk)

#cross encodeoing
    start = time.perf_counter()

    ranked = reranker.rerank(
        question,
        candidate_chunks,
        top_k=5,
    )

    rerank_time = (time.perf_counter() - start) * 1000

#llm
    context = "\n\n".join(
        c["text"]
        for c in ranked
    )

    start = time.perf_counter()

    answer = llm.generate(
        question,
        context,
    )

    llm_time = (time.perf_counter() - start) * 1000

    total_time = (time.perf_counter() - total_start) * 1000
    # report
    console.print("\n[bold cyan]KnowledgeOS Benchmark[/bold cyan]\n")

    console.print(f"Query              : {question}\n")

    console.print(f"Embedding          : {embedding_time:.2f} ms")
    console.print(f"FAISS Search       : {faiss_time:.2f} ms")
    console.print(f"BM25 Search        : {bm25_time:.2f} ms")
    console.print(f"Hybrid Fusion      : {hybrid_time:.2f} ms")
    console.print(f"Cross Encoder      : {rerank_time:.2f} ms")
    console.print(f"LLM Generation     : {llm_time:.2f} ms")

    console.print("-" * 45)

    console.print(f"Total Pipeline     : {total_time:.2f} ms\n")

    console.print(f"Retrieved Chunks   : {len(ranked)}")

    if ranked:
        console.print(f"Embedding Size     : {len(query_embedding)}")

    console.print("\n[bold green]Generated Answer[/bold green]\n")
    console.print(answer)

if __name__ == "__main__":
    app()