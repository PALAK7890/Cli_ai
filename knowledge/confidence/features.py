"""
Confidence feature extraction.
"""

from dataclasses import dataclass


@dataclass
class ConfidenceFeatures:
    semantic_score: float
    keyword_score: float
    retrieval_rank: int
    agreement: int
    context_length: int
    query_length: int


class FeatureExtractor:
    """Extract retrieval features."""

    def extract(
        self,
        semantic_score: float,
        keyword_score: float,
        retrieval_rank: int,
        agreement: int,
        context: str,
        query: str,
    ) -> ConfidenceFeatures:

        return ConfidenceFeatures(
            semantic_score=float(semantic_score),
            keyword_score=float(keyword_score),
            retrieval_rank=int(retrieval_rank),
            agreement=int(agreement),
            context_length=len(context),
            query_length=len(query.split()),
        )