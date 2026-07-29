"""
Base interface for embedding models.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a search query.
        """
        raise NotImplementedError