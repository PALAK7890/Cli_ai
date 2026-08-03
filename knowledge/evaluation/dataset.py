"""
Evaluation dataset.
"""

from dataclasses import dataclass


@dataclass
class EvaluationExample:

    query: str

    relevant_chunks: list[int]