"""
Loader for plain text and Markdown files.
"""

from pathlib import Path
from typing import List
from knowledge.loaders.base import BaseLoader, LoadedDocument


class TxtLoader(BaseLoader):
    """Parses plain text (.txt) and Markdown (.md) documents."""

    def load(self, path: Path) -> List[LoadedDocument]:
        """
        Loads a plain text or Markdown file.
        
        Args:
            path: Path to the text or markdown file.
            
        Returns:
            A list containing a single LoadedDocument.
        """
        # Read the file as UTF-8, handling decoding issues gracefully if they arise later.
        text = path.read_text(encoding="utf-8").strip()
        
        return [
            LoadedDocument(
                text=text,
                source_path=str(path.resolve()),
                page_number=None,
                metadata={}
            )
        ]
