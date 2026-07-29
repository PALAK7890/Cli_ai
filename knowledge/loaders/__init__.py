"""
Document loaders module.
"""

from knowledge.loaders.base import BaseLoader, LoadedDocument
from knowledge.loaders.router import LoaderRouter

__all__ = ["BaseLoader", "LoadedDocument", "LoaderRouter"]
