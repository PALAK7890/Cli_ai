"""
Loader for PDF documents.
"""

from pathlib import Path
from typing import List
from knowledge.loaders.base import BaseLoader, LoadedDocument


class PdfLoader(BaseLoader):
    """Parses PDF documents using PyMuPDF."""

    def load(self, path: Path) -> List[LoadedDocument]:
        raise NotImplementedError("PdfLoader is not yet implemented")
