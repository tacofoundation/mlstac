"""
Utility Functions

This module provides utility functions used throughout the mlstac package.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

# Define the valid scheme types for typing
SchemeType = Literal[
    "http", "https", "ftp", "s3", "gs", "local", "snippet", "pt2_list"
]

def get_scheme(source: str | Path | list) -> SchemeType:
    """
    Determine the protocol scheme for a given source.

    This function analyzes a URL, file path, or list of paths and determines
    how it should be accessed. It supports HTTP(S), FTP, cloud storage (S3, GS),
    local files, special "snippet" references, and direct .pt2 model lists.

    Args:
        source: A URL, file path, or list of .pt2 model paths

    Returns:
        The identified scheme: 'http', 'https', 'ftp', 's3', 'gs', 'local',
        'snippet', or 'pt2_list'

    Examples:
        >>> get_scheme("https://example.com/model.safetensor")
        'https'
        >>> get_scheme("/path/to/local/model")
        'local'
        >>> get_scheme("resnet50")  # A snippet reference
        'snippet'
        >>> get_scheme([Path("model1.pt2"), Path("model2.pt2")])
        'pt2_list'
    """
    # Check if it's a list (direct .pt2 paths)
    if isinstance(source, list):
        return "pt2_list"

    # Accept Path too: normalize to str so urlparse and Path both work.
    source = str(source)

    parsed = urlparse(source)

    # Check for URL schemes
    if parsed.scheme in {"http", "https", "ftp", "s3", "gs"}:
        return parsed.scheme

    # Check if it's a local path that exists
    if Path(source).exists():
        return "local"

    # If not a URL or local path, assume it's a model snippet reference
    return "snippet"
