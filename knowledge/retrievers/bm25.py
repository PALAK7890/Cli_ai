"""
BM25 keyword retriever.
"""

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Simple BM25 retriever."""

    def __init__(self) -> None:
        self.documents = []
        self.tokenized = []
        self.bm25 = None

    def build(self, texts: list[str]) -> None:
        """Build the BM25 index."""

        self.documents = texts
        self.tokenized = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 5):
        """Return ranked document indices."""

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query.lower().split())

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]