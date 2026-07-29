"""
Loader for DOCX documents.
"""

from pathlib import Path
from typing import List
from knowledge.loaders.base import BaseLoader, LoadedDocument


class DocxLoader(BaseLoader):
    """Parses Word document (.docx) files using python-docx."""

    def load(self, path: Path) -> List[LoadedDocument]:
        raise NotImplementedError("DocxLoader is not yet implemented")
