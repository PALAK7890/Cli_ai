from collections import defaultdict
from typing import Dict, List, Tuple


class HybridRetriever:

    def fuse(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion.
        """

        scores: Dict[int, float] = defaultdict(float)

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