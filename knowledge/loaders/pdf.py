"""
Loader for PDF documents.
"""

from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from knowledge.loaders.base import BaseLoader, LoadedDocument


class PdfLoader(BaseLoader):
    """Parses PDF documents using PyMuPDF."""

    def load(self, path: Path) -> List[LoadedDocument]:
        """
        Loads a PDF document page by page.
        
        Args:
            path: Path to the PDF document.
            
        Returns:
            A list of LoadedDocument instances, one per page.
        """
        documents: List[LoadedDocument] = []
        
        # Open PDF file using fitz (PyMuPDF)
        with fitz.open(str(path)) as doc:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text().strip()
                
                # We save each page as a separate LoadedDocument to preserve page numbers
                documents.append(
                    LoadedDocument(
                        text=text,
                        source_path=str(path.resolve()),
                        page_number=page_idx + 1,
                        metadata={"total_pages": len(doc)}
                    )
                )
                
        return documents
