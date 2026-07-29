"""
Text chunking module.
"""

from knowledge.chunkers.base import BaseChunker, DocumentChunk
from knowledge.chunkers.fixed_size import FixedSizeChunker

__all__ = ["BaseChunker", "DocumentChunk", "FixedSizeChunker"]
