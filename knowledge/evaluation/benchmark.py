"""
Benchmark retrieval quality.
"""

import time

from knowledge.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
)


class Benchmark:

    def evaluate(
        self,
        query: str,
        retrieved: list[int],
        relevant: list[int],
    ):

        start = time.perf_counter()

        metrics = {
            "Recall@5": recall_at_k(
                retrieved,
                relevant,
                5,
            ),
            "Precision@5": precision_at_k(
                retrieved,
                relevant,
                5,
            ),
            "MRR": mrr(
                retrieved,
                relevant,
            ),
        }

        metrics["Latency"] = (
            time.perf_counter() - start
        ) * 1000

        return metrics