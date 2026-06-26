"""MLSTAC - Machine Learning STAC metadata for model distribution."""

import importlib as _importlib

_importlib.import_module("mlstac._quiet")

from mlstac.main import ModelLoader, download, load

__all__ = ["ModelLoader", "download", "load"]

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("mlstac")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"
