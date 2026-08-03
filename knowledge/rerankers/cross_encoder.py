"""
Cross Encoder reranker.
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a CrossEncoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = []

        for chunk, score in zip(chunks, scores):
            chunk = chunk.copy()
            chunk["rerank_score"] = float(score)
            ranked.append(chunk)

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]