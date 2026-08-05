"""
Evaluation pipeline for KnowledgeOS retrieval.
"""

import time
from pathlib import Path

from knowledge.embeddings.sentence_transformer import SentenceTransformerEmbedding
from knowledge.retrievers.bm25 import BM25Retriever
from knowledge.retrievers.hybrid import HybridRetriever
from knowledge.rerankers.cross_encoder import CrossEncoderReranker
from knowledge.vectorstores.faiss_store import FAISSStore

from knowledge.evaluation.dataset import EvaluationDataset
from knowledge.evaluation.metrics import (
    accuracy_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


class RetrievalEvaluator:

    def __init__(self):

        self.embedder = SentenceTransformerEmbedding()

        self.vector_store = FAISSStore()
        self.vector_store.load(".knowledge/index/faiss.index")

        self.reranker = CrossEncoderReranker()

    def evaluate(
        self,
        dataset_path: str,
    ):

        dataset = EvaluationDataset(dataset_path).load()

        with open(".knowledge/index/chunks.json", "r", encoding="utf-8") as f:
            import json

            data = json.load(f)

        chunks = data["chunks"]

        texts = [chunk["text"] for chunk in chunks]

        bm25 = BM25Retriever()
        bm25.build(texts)

        hybrid = HybridRetriever()

        accuracy1 = []
        accuracy3 = []
        precision3 = []
        recall3 = []
        rr_scores = []

        latencies = []

        results = []

        for sample in dataset:

            question = sample["question"]
            expected = sample["expected_document"]

            start = time.perf_counter()

            embedding = self.embedder.embed_query(question)

            scores, indices = self.vector_store.search(
                embedding,
                top_k=20,
            )

            faiss_results = []

            for score, idx in zip(scores[0], indices[0]):

                if idx == -1:
                    continue

                faiss_results.append((idx, float(score)))

            bm25_results = bm25.search(
                question,
                top_k=20,
            )

            fused = hybrid.fuse(
                faiss_results,
                bm25_results,
            )

            candidate_chunks = []

            for idx, score in fused:

                candidate_chunks.append(
                    {
                        "idx": idx,
                        "text": chunks[idx]["text"],
                        "score": score,
                    }
                )

            reranked = self.reranker.rerank(
                question,
                candidate_chunks,
                top_k=5,
            )

            retrieved = []

            for item in reranked:

                idx = item["idx"]
                retrieved.append(idx)

            relevant = []

            for i, chunk in enumerate(chunks):

                name = Path(chunk["source_path"]).name

                if name == expected:
                    relevant.append(i)

            accuracy1.append(
                accuracy_at_k(retrieved, relevant, 1)
            )

            accuracy3.append(
                accuracy_at_k(retrieved, relevant, 3)
            )

            precision3.append(
                precision_at_k(retrieved, relevant, 3)
            )

            recall3.append(
                recall_at_k(retrieved, relevant, 3)
            )

            rr_scores.append(
                reciprocal_rank(retrieved, relevant)
            )

            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            results.append(
                {
                    "question": question,
                    "expected": expected,
                    "retrieved": [
                        Path(chunks[i]["source_path"]).name
                        for i in retrieved
                    ],
                    "latency_ms": latency,
                }
            )

        return {
            "queries": len(dataset),
            "accuracy@1": sum(accuracy1) / len(accuracy1),
            "accuracy@3": sum(accuracy3) / len(accuracy3),
            "precision@3": sum(precision3) / len(precision3),
            "recall@3": sum(recall3) / len(recall3),
            "mrr": mean_reciprocal_rank(rr_scores),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "results": results,
        }