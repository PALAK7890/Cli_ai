"""
Document loaders module.
"""

from knowledge.loaders.base import BaseLoader, LoadedDocument
from knowledge.loaders.router import LoaderRouter, load_file_safely

__all__ = ["BaseLoader", "LoadedDocument", "LoaderRouter", "load_file_safely"]
