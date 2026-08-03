# RAGForge

A modular Retrieval-Augmented Generation (RAG) framework that indexes local documents, performs hybrid retrieval using dense and sparse search, reranks retrieved passages with a Cross-Encoder, and answers questions using a local LLM through Ollama.

RAGForge is designed as a modular research-oriented codebase where each stage of the retrieval pipeline is independently replaceable and benchmarkable.

---

## Features

- Multi-format document ingestion
  - PDF
  - TXT
  - Markdown
  - HTML
  - DOCX

- Recursive document chunking with configurable overlap

- Dense semantic retrieval
  - SentenceTransformers
  - FAISS vector indexing

- Sparse lexical retrieval
  - BM25 ranking

- Hybrid retrieval
  - Reciprocal Rank Fusion (RRF)

- Query expansion

- Cross-Encoder reranking

- Local LLM inference using Ollama

- End-to-end benchmarking

- Modular architecture for experimentation

---

## Architecture

```
                 Documents
                     │
                     ▼
            Document Loaders
                     │
                     ▼
          Recursive Chunking
                     │
                     ▼
      SentenceTransformer Embeddings
                     │
                     ▼
                FAISS Index
                     │
                     │
User Query ──────────┘
     │
     ▼
Query Expansion
     │
     ▼
Dense Retrieval (FAISS)
     │
Sparse Retrieval (BM25)
     │
     ▼
Hybrid Fusion (RRF)
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Context Construction
     │
     ▼
Ollama LLM
     │
     ▼
Answer
```

---

## Repository Structure

```
knowledge/
│
├── chunkers/
├── cli/
├── confidence/
├── config/
├── embeddings/
├── evaluation/
├── llms/
├── loaders/
├── query/
├── rerankers/
├── retrievers/
├── vectorstores/
└── tests/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/KnowledgeOS.git

cd KnowledgeOS
```

Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -e .
```

Install Ollama

```bash
brew install ollama
```

Pull a model

```bash
ollama pull llama3.2
```

Start Ollama

```bash
ollama serve
```

---

## CLI Usage

Initialize workspace

```bash
knowledge init
```

Index documents

```bash
knowledge add docs/
```

Search documents

```bash
knowledge search "hybrid retrieval"
```

Ask questions

```bash
knowledge ask "What is KnowledgeOS?"
```

List indexed documents

```bash
knowledge list
```

Show workspace statistics

```bash
knowledge stats
```

Benchmark the pipeline

```bash
knowledge benchmark "What is KnowledgeOS?"
```

---

## Retrieval Pipeline

KnowledgeOS follows a multi-stage retrieval pipeline.

1. Query Expansion
2. Dense Retrieval using FAISS
3. Sparse Retrieval using BM25
4. Reciprocal Rank Fusion
5. Cross-Encoder Reranking
6. Local LLM Generation

This design improves retrieval quality while maintaining low retrieval latency and modularity.

---

## Benchmark Example

```
KnowledgeOS Benchmark

Embedding          : 541 ms
FAISS Search       : 60 ms
BM25 Search        : 1 ms
Hybrid Fusion      : 0.06 ms
Cross Encoder      : 491 ms
LLM Generation     : 8756 ms

Total Pipeline     : 9855 ms
```

Pipeline latency is primarily dominated by local LLM inference. Retrieval stages remain lightweight and independently measurable.

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| CLI | Typer |
| Dense Retrieval | SentenceTransformers |
| Vector Database | FAISS |
| Sparse Retrieval | BM25 |
| Reranking | CrossEncoder |
| LLM | Ollama |
| Configuration | YAML |
| Console | Rich |

---

## Future Work

- Streaming generation
- Multi-vector retrieval
- Metadata filtering
- Incremental indexing
- Retrieval evaluation datasets
- Citation-aware answer generation

---

## License

MIT License