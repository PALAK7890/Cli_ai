"""
Evaluation metrics for retrieval systems.
"""

from typing import List


def accuracy_at_k(
    retrieved: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Returns 1 if any relevant document appears in the top-k results,
    otherwise 0.
    """

    retrieved = retrieved[:k]

    return float(
        any(idx in relevant for idx in retrieved)
    )


def precision_at_k(
    retrieved: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Precision@K
    """

    retrieved = retrieved[:k]

    if not retrieved:
        return 0.0

    hits = len(set(retrieved) & set(relevant))

    return hits / len(retrieved)


def recall_at_k(
    retrieved: List[int],
    relevant: List[int],
    k: int,
) -> float:
    """
    Recall@K
    """

    if not relevant:
        return 0.0

    retrieved = retrieved[:k]

    hits = len(set(retrieved) & set(relevant))

    return hits / len(relevant)


def reciprocal_rank(
    retrieved: List[int],
    relevant: List[int],
) -> float:
    """
    Reciprocal Rank for a single query.
    """

    for rank, idx in enumerate(retrieved, start=1):

        if idx in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    scores: List[float],
) -> float:
    """
    Mean Reciprocal Rank over multiple queries.
    """

    if not scores:
        return 0.0

    return sum(scores) / len(scores)