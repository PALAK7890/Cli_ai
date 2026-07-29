"""
Fixed-size text chunker implementation.
"""

from typing import List
from knowledge.loaders.base import LoadedDocument
from knowledge.chunkers.base import BaseChunker, DocumentChunk


class FixedSizeChunker(BaseChunker):
    """Splits documents into fixed-size character sequences with overlap."""

    def chunk(self, docs: List[LoadedDocument], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """
        Chunks documents by slicing them at fixed character indices.
        
        Args:
            docs: List of loaded documents to chunk.
            chunk_size: The character size of each chunk.
            chunk_overlap: The character overlap between consecutive chunks.
            
        Returns:
            A list of DocumentChunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        chunks: List[DocumentChunk] = []

        for doc in docs:
            text = doc.text.strip()
            if not text:
                continue

            text_len = len(text)
            stride = chunk_size - chunk_overlap
            chunk_index = 0
            start = 0

            # If the text is shorter than chunk_size, output it directly
            if text_len <= chunk_size:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.document_id}_{chunk_index}",
                        text=text,
                        document_id=doc.document_id,
                        source_path=doc.source_path,
                        page_number=doc.page_number,
                        chunk_index=chunk_index,
                        metadata=doc.metadata.copy()
                    )
                )
                continue

            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.document_id}_{chunk_index}",
                        text=chunk_text,
                        document_id=doc.document_id,
                        source_path=doc.source_path,
                        page_number=doc.page_number,
                        chunk_index=chunk_index,
                        metadata=doc.metadata.copy()
                    )
                )
                chunk_index += 1

                if end == text_len:
                    break

                start += stride

        return chunks
