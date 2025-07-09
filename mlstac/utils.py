"""
Utility Functions

This module provides utility functions used throughout the mlstac package.
"""
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

# Define the valid scheme types for typing
SchemeType = Literal["http", "https", "ftp", "s3", "gs", "local", "snippet"]


def get_scheme(source: str) -> SchemeType:
    """
    Determine the protocol scheme for a given source.

    This function analyzes a URL or file path and determines how it should be accessed.
    It supports HTTP(S), FTP, cloud storage (S3, GS), local files, and special "snippet"
    references.

    Args:
        source: A URL or file path

    Returns:
        The identified scheme: 'http', 'https', 'ftp', 's3', 'gs', 'local', or 'snippet'

    Examples:
        >>> get_scheme("https://example.com/model.safetensor")
        'https'
        >>> get_scheme("/path/to/local/model")
        'local'
        >>> get_scheme("resnet50")  # A snippet reference
        'snippet'
    """
    parsed = urlparse(source)

    # Check for URL schemes
    if parsed.scheme in {"http", "https", "ftp", "s3", "gs"}:
        return parsed.scheme

    # Check if it's a local path that exists
    if Path(source).exists():
        return "local"

    # If not a URL or local path, assume it's a model snippet reference
    return "snippet"
