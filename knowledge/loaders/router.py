"""
Document loader router. Matches file extensions to appropriate loaders.
"""

from pathlib import Path
from typing import Dict, Type
from knowledge.loaders.base import BaseLoader
from knowledge.loaders.txt import TxtLoader
from knowledge.loaders.pdf import PdfLoader
from knowledge.loaders.docx import DocxLoader
from knowledge.loaders.html_loader import HtmlLoader


class LoaderRouter:
    """Routes files to the appropriate document loader based on extension."""

    def __init__(self) -> None:
        self.loaders: Dict[str, Type[BaseLoader]] = {
            ".txt": TxtLoader,
            ".md": TxtLoader,
            ".pdf": PdfLoader,
            ".docx": DocxLoader,
            ".html": HtmlLoader,
            ".htm": HtmlLoader,
        }

    def get_loader(self, path: Path) -> BaseLoader:
        """
        Retrieves the appropriate loader for the given file path.
        
        Args:
            path: Path to the document.
            
        Returns:
            An instantiated loader matching the file's extension.
            
        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = path.suffix.lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file format: {ext} for file {path.name}")
        return self.loaders[ext]()
