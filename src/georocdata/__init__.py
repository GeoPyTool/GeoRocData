"""GEOROC database downloader and SQLite converter."""

__version__ = "0.1.0"
__author__ = "GeoPyTool Team"

from .downloader import GEOROCDownloader
from .processor import GEOROCProcessor
from .compilations import GEOROC_COMPILATIONS, LEGACY_CATEGORIES

__all__ = [
    "GEOROCDownloader",
    "GEOROCProcessor",
    "GEOROC_COMPILATIONS",
    "LEGACY_CATEGORIES",
]