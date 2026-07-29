"""
Loader for plain text and Markdown files.
"""

from pathlib import Path
from typing import List
from knowledge.loaders.base import BaseLoader, LoadedDocument


class TxtLoader(BaseLoader):
    """Parses plain text (.txt) and Markdown (.md) documents."""

    def load(self, path: Path) -> List[LoadedDocument]:
        raise NotImplementedError("TxtLoader is not yet implemented")
