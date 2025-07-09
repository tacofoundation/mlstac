"""
Model Fetching Utilities

This module provides utilities for retrieving models and their metadata
from various storage backends including HTTP, local filesystem, AWS S3,
and Google Cloud Storage.

It handles downloading model files and loading Python modules dynamically.
"""
import json
import sys
import types
from pathlib import Path
from shutil import copyfile
from typing import Callable
from urllib.parse import urlparse

import pystac.item
import requests

from mlstac.utils import get_scheme


def _fetch_http(source: str) -> str:
    """
    Fetch content from an HTTP/HTTPS/FTP endpoint.

    Args:
        source: URL to fetch content from

    Returns:
        Text content from the URL

    Raises:
        requests.HTTPError: If the request fails
    """
    response = requests.get(source, timeout=30)
    response.raise_for_status()
    return response.text

def _fetch_local(source: str) -> str:
    """
    Read content from the local filesystem.

    Args:
        source: Path to local file

    Returns:
        Text content of the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read due to permissions
    """
    with open(source, encoding="utf-8") as f:
        return f.read()

def _fetch_s3(source: str) -> str:
    """
    Fetch content from an AWS S3 bucket.

    Args:
        source: S3 URL in the format s3://bucket-name/path/to/file

    Returns:
        Text content from the S3 object

    Raises:
        ImportError: If boto3 is not installed
        boto3.exceptions.Boto3Error: If S3 access fails
    """
    try:
        import boto3
    except ImportError as e:
        raise ImportError("S3 access requires boto3: pip install boto3") from e

    parsed = urlparse(source)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    return obj["Body"].read().decode("utf-8")

def _fetch_gs(source: str) -> str:
    """
    Fetch content from Google Cloud Storage.

    Args:
        source: GCS URL in the format gs://bucket-name/path/to/file

    Returns:
        Text content from the GCS object

    Raises:
        ImportError: If google-cloud-storage is not installed
        google.cloud.exceptions.GoogleCloudError: If GCS access fails
    """
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError("GCS access requires google-cloud-storage: pip install google-cloud-storage") from e

    parsed = urlparse(source)
    client = storage.Client()
    blob = client.bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
    return blob.download_as_text()

# Map of schemes to their handler functions
_SCHEME_HANDLERS: dict[str, Callable[[str], str]] = {
    "http": _fetch_http,
    "https": _fetch_http,
    "ftp": _fetch_http,
    "local": _fetch_local,
    "s3": _fetch_s3,
    "gs": _fetch_gs,
}

def fetch_source(source: str, snippet_suffix: str = "") -> str:
    """
    Retrieve textual content from a given source.

    Handles various protocols and appends the snippet suffix if needed.

    Args:
        source: A URL, file path, or model identifier
        snippet_suffix: Suffix to append if the source is a snippet

    Returns:
        The fetched content as a string

    Raises:
        ValueError: If the scheme is not supported
        RuntimeError: If fetching fails
    """
    scheme = get_scheme(source)
    full_source = f"{source}{snippet_suffix}"

    handler = _SCHEME_HANDLERS.get(scheme)
    if not handler:
        raise ValueError(f"Unsupported scheme: {scheme}. Supported schemes: {', '.join(_SCHEME_HANDLERS.keys())}")

    try:
        return handler(full_source)
    except Exception as e:
        raise RuntimeError(f"Failed to load from {full_source}: {e!s}") from e

# --- Public Text-Based Loaders ---

def load_python_module(
    source: str,
    module_name: str = "mlstac_model_loader"
) -> types.ModuleType:
    """
    Dynamically load a Python module from the given source.

    This function retrieves Python code and compiles it into a module
    that can be imported and used within the current Python environment.

    Args:
        source: A URI or local file path to the Python code
        module_name: The name to assign to the imported module

    Returns:
        The loaded module object

    Raises:
        RuntimeError: If loading or compilation fails
        SyntaxError: If the loaded code has syntax errors
    """
    try:
        code = fetch_source(source, snippet_suffix="/load.py")

        # Create a new module
        module = types.ModuleType(module_name)
        module.__file__ = f"{source}/load.py"

        # Execute the code in the module's context
        exec(code, module.__dict__)

        # Register in sys.modules
        sys.modules[module_name] = module
        return module
    except Exception as e:
        raise RuntimeError(f"Failed to load Python module from {source}: {e!s}") from e


def load_stac_item(source: str) -> pystac.Item:
    """
    Load a STAC JSON item from the given source.

    This function expects the JSON to follow the STAC item specification.
    A snippet-based source will have '/mlm.json' appended to its URL.

    Args:
        source: A URI or local file path to the JSON file

    Returns:
        A STAC Item object representing the model metadata

    Raises:
        RuntimeError: If loading or parsing fails
        ValueError: If the JSON does not conform to STAC specification
    """
    try:
        content = fetch_source(source, snippet_suffix="")
        json_dict = json.loads(content)
        return pystac.item.Item.from_dict(json_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in STAC item from {source}: {e!s}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load STAC item from {source}: {e!s}") from e

def download_file(
    source: str,
    snippet_suffix: str = "",
    outpath: str | Path = "."
) -> Path:
    """
    Download a file from the given source and save it in the specified output folder.

    This function handles various protocols (HTTP, local, S3, GS) and uses
    streaming for remote downloads to efficiently write the file to disk.

    Args:
        source: A URL, local file path, or model identifier
        snippet_suffix: Suffix to append if the source is a snippet
        outpath: Directory where the file will be saved

    Returns:
        A Path object representing the saved file

    Raises:
        ValueError: If the scheme is not supported
        RuntimeError: If downloading fails
    """
    scheme = get_scheme(source)

    # Determine the output filename based on the source path
    to_download = f"{source}{snippet_suffix}"
    out_file = Path(outpath) / Path(snippet_suffix).name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if scheme in {"http", "https", "ftp"}:
            # Streaming download to file
            with requests.get(to_download, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(out_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        elif scheme == "local":
            # For local sources, simply copy the file
            copyfile(to_download, out_file)
        elif scheme == "s3":
            try:
                import boto3
            except ImportError as e:
                raise ImportError("S3 access requires boto3: pip install boto3") from e

            parsed = urlparse(to_download)
            s3 = boto3.client("s3")
            s3.download_file(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip("/"),
                Filename=str(out_file)
            )
        elif scheme == "gs":
            try:
                from google.cloud import storage
            except ImportError as e:
                raise ImportError("GCS access requires google-cloud-storage: pip install google-cloud-storage") from e

            parsed = urlparse(to_download)
            client = storage.Client()
            bucket = client.bucket(parsed.netloc)
            blob = bucket.blob(parsed.path.lstrip("/"))
            blob.download_to_filename(str(out_file))
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        return out_file
    except Exception as e:
        # Clean up partially downloaded file if it exists
        if out_file.exists():
            out_file.unlink()
        raise RuntimeError(f"Failed to download {to_download} to {out_file}: {e!s}") from e
