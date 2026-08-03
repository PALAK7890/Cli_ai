"""
Evaluation metrics for retrieval systems.
"""

from typing import List


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Recall@K.
    """
    retrieved = retrieved[:k]

    hits = len(set(retrieved) & set(relevant))

    if len(relevant) == 0:
        return 0.0

    return hits / len(relevant)


def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Precision@K.
    """

    retrieved = retrieved[:k]

    if len(retrieved) == 0:
        return 0.0

    hits = len(set(retrieved) & set(relevant))

    return hits / len(retrieved)


def mrr(retrieved: List[int], relevant: List[int]) -> float:
    """
    Mean Reciprocal Rank.
    """

    for rank, idx in enumerate(retrieved):

        if idx in relevant:
            return 1 / (rank + 1)

    return 0.0