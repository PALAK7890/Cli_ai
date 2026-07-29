"""
Loader for HTML documents.
"""

from pathlib import Path
from typing import List
from knowledge.loaders.base import BaseLoader, LoadedDocument


class HtmlLoader(BaseLoader):
    """Parses HTML documents using BeautifulSoup4."""

    def load(self, path: Path) -> List[LoadedDocument]:
        raise NotImplementedError("HtmlLoader is not yet implemented")
