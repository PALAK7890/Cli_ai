"""
Hybrid FAISS + BM25 retrieval using Reciprocal Rank Fusion.
"""

from collections import defaultdict


class HybridRetriever:

    def fuse(
        self,
        faiss_results,
        bm25_results,
        k: int = 60,
    ):
        """
        Reciprocal Rank Fusion.
        """

        scores = defaultdict(float)

        for rank, (idx, _) in enumerate(faiss_results):
            scores[idx] += 1 / (k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += 1 / (k + rank + 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked