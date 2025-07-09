"""
MLSTAC: A machine learning model-sharing specification based on STAC MLM and Safetensors.

MLSTAC is a Python package designed for seamless machine learning model sharing and loading.
It builds on the STAC MLM specification for standardized model metadata and utilizes Safetensors
for secure and efficient model storage.

With MLSTAC, you can discover, download, and load ML models with a single line of code from various
data repositories, making model deployment and integration easier than ever.

Example:
    >>> import mlstac
    >>>
    >>> # Load a model by ID
    >>> model = mlstac.load("UNetMobV2_V1")
    >>>
    >>> # Download the model locally
    >>> model.download("./models/UNetMobV2_V1")
    >>>
    >>> # Load for inference
    >>> inference_model = model.load_compiled_model()
    >>>
    >>> # Load for training
    >>> training_model = model.load_trainable_model()
"""

from mlstac.main import load, download

__all__ = ["load", "download"]


# Dynamic version import
import importlib.metadata

__version__ = importlib.metadata.version("mlstac")


