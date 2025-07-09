"""
MLSTAC main module.

This module provides a clean interface for working with MLSTAC models through STAC metadata,
supporting multiple storage backends.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fsspec
import pystac
from tqdm import tqdm

from mlstac.fetch import download_file, load_python_module, load_stac_item
from mlstac.utils import get_scheme


def download(file: Path | str, output_dir: Path | str) -> Path:
    """
    Download model files of a specified model from a remote source.
    Args:
        output_path: Target directory for downloaded files
    Returns:
        Path object pointing to the output directory
    Raises:
        RuntimeError: If any of the specified files fail to download
    """

    # Convert strings to Path objects
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    item = load_stac_item(file)

    # First save the mlm.json file in the output directory
    mlm_file = output_dir / Path(file).name
    with open(mlm_file, "w") as f:
        f.write(json.dumps(item.to_dict(), indent=4))

    # Define model artifacts to download with their suffixes
    to_download =  {}
    for key, value in item.assets.items():
        to_download[key] = value.href

    # Download selected files
    downloaded_files = []
    for name, source in tqdm(to_download.items(), desc="Downloading files"):
        try:
            file_path = download_file(
                source=source,
                outpath=output_dir / Path(source).name
            )
            downloaded_files.append(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {name}: {e!s}") from e
    # Reload model metadata after download
    return ModelLoader((output_dir / "mlm.json").absolute().as_posix())


class ModelLoader:
    """
    Manages machine learning models with capabilities for fetching, downloading, and loading.

    This class provides a unified interface for working with ML models regardless of
    their storage location (local, remote) or format. It uses STAC metadata to describe
    model properties and capabilities.

    Attributes:
        source (str): Location of the model (URL or local path)
        scheme (str): Access scheme ('snippet', 'local', 'http', etc.)
        item (pystac.Item): STAC metadata for the model
        module (types.ModuleType, optional): Python module containing model loading functions

    Examples:
        >>> # Load a model from a snippet reference
        >>> model = ModelLoader("resnet50")
        >>>
        >>> # Download the model locally
        >>> model.download("./models")
        >>>
        >>> # Load the model for inference
        >>> inference_model = model.load_compiled_model()
    """

    def __init__(self, file: str):
        """
        Initialize the model manager.

        Args:
            file: The JSON file that contains the model metadata

        Raises:
            ValueError: If the source cannot be resolved or the model cannot be loaded.
        """        
        self.file = file
        self.scheme = get_scheme(file)
        
        if self.scheme == "local":
            if Path(file).is_dir():
                self.file = Path(file) / "mlm.json"
            self.item = self._load()
            self.source = Path(self.file).parent.as_posix()
        else:
            self.item = load_stac_item(file)
            self.source = None 

        self.module = None           


    def print_schema(self) -> None:
        """
        Prints a visually appealing schema of the model.

        Automatically detects if running in a Jupyter/Colab notebook or terminal
        and formats the output accordingly.
        """
        # Check if running in notebook environment
        in_notebook = 'ipykernel' in sys.modules

        # Gather model details
        model_id = self.item.id
        framework = self.item.properties.get("mlm:framework", "Not specified")
        architecture = self.item.properties.get("mlm:architecture", "Not specified")
        tasks = self.item.properties.get("mlm:tasks", [])
        dependencies = self.item.properties.get("dependencies", "Not specified")

        # Convert file size from bytes to MB if available
        file_size = self.item.properties.get("file:size", 0)
        file_size_mb = f"{file_size / (1024 * 1024):.2f} MB" if file_size else "Unknown"

        if in_notebook:
            from IPython.display import HTML, display
            # Rich display for notebooks
            html_content = f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #007bff;">
                <h3 style="color: #007bff; margin-top: 0;">📜 MLSTAC Model Schema</h3>
                <hr style="border-top: 1px solid #e9ecef;">
                <p style="color: #2E2F2F;"><b>🆔 Model ID:</b> {model_id}</p>
                <p style="color: #2E2F2F;"><b>🌍 Source:</b> {self.file}</p>
                <p style="color: #2E2F2F;"><b>📡 Scheme:</b> {self.scheme}</p>
                <p style="color: #2E2F2F;"><b>🛠️ Framework:</b> {framework}</p>
                <p style="color: #2E2F2F;"><b>👁️‍🗨️ Dependencies:</b> {dependencies}</p>
                <p style="color: #2E2F2F;"><b>🏗️ Architecture:</b> {architecture}</p>
                <p style="color: #2E2F2F;"><b>📊 Tasks:</b> {', '.join(tasks) if tasks else 'None specified'}</p>
                <p style="color: #2E2F2F;"><b>📦 Size:</b> {file_size_mb}</p>
            </div>
            """
            display(HTML(html_content))
        else:
            # Terminal-friendly output with borders and spacing
            border = "-" * 50
            print(f"{border}")
            print("📜 MLSTAC Model Schema")
            print(f"{border}")
            print(f"🆔 Model ID:      {model_id}")
            print(f"🌍 Source:        {self.file}")
            print(f"📡 Scheme:        {self.scheme}")
            print(f"🛠️ Framework:     {framework}")
            print(f"👁️‍🗨️ Dependencies:  {dependencies}")
            print(f"🏗️ Architecture:  {architecture}")
            print(f"📊 Tasks:         {', '.join(tasks) if tasks else 'None specified'}")
            print(f"📦 Size:          {file_size_mb}")
            print(f"{border}")

    def _load(self) -> pystac.Item:
        """
        Load and update model metadata from local storage.

        Returns:
            Updated STAC item with local file references

        Raises:
            FileNotFoundError: If the metadata file doesn't exist
            ValueError: If the metadata file is invalid or corrupted
        """        

        try:
            with fsspec.open(self.file, "r") as f:
                mlm_data = json.load(f)
                
                # Update asset paths to absolute local paths
                asset_files = {}
                for key, value in mlm_data["assets"].items():
                    asset_files[key] = (Path(self.file).parent / Path(value["href"]).name).absolute().as_posix()

            return pystac.item.Item.from_dict(mlm_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model metadata file not found at {self.file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid model metadata format: {e!s}") from e

    def example_data(self, *args, **kwargs) -> Any:
        """
        Load example data for model testing.

        Returns:
            Processed example data in the format expected by the model

        Raises:
            FileNotFoundError: If example data file doesn't exist
            ValueError: If model hasn't been downloaded locally
        """
        
        # Ensure model is downloaded locally
        self._verify_local_access()
        
        try:
            # Ensure module is loaded
            if self.module is None:
                self.module = load_python_module(self.source)
            return self.module.example_data(Path(self.source), *args, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Example data file not found at {self.source}/example_data.safetensor")
        except AttributeError:
            raise AttributeError("Model loader module doesn't implement 'example_data' function")

    def trainable_model(self, *args, **kwargs) -> Any:
        """
        Load the trainable version of the model for fine-tuning.

        Returns:
            Trainable model instance

        Raises:
            ValueError: If model hasn't been downloaded locally
            FileNotFoundError: If trainable model file doesn't exist
            AttributeError: If model loader doesn't implement required functions
        """
        self._verify_local_access()
        self.item = self._load()
        

        # Load the Python module containing model loading functions
        if self.module is None:
            self.module = load_python_module(self.source)

        try:
            return self.module.trainable_model(Path(self.source), *args, **kwargs)
        except KeyError:
            raise KeyError("Trainable model asset not found in metadata")
        except AttributeError:
            raise AttributeError("Model loader module doesn't implement 'trainable_model' function")

    def compiled_model(self, *args, **kwargs) -> Any:
        """
        Load the compiled (optimized) version of the model for inference.

        Returns:
            Compiled model instance for inference

        Raises:
            ValueError: If model hasn't been downloaded locally
            FileNotFoundError: If compiled model file doesn't exist
            AttributeError: If model loader doesn't implement required functions
        """
        self._verify_local_access()
        self.item = self._load()


        # Load the Python module containing model loading functions
        if self.module is None:
            self.module = load_python_module(self.source)

        try:            
            return self.module.compiled_model(Path(self.source), *args, **kwargs)
        except KeyError:
            raise KeyError("Compiled model asset not found in metadata")
        except AttributeError:
            raise AttributeError("Model loader module doesn't implement 'compiled_model' function")


    def display_results(self, *args, **kwargs) -> Any:
        """
        Load the function to display the results of the model.

        Returns:
            Compiled model instance for inference

        Raises:
            ValueError: If model hasn't been downloaded locally
            FileNotFoundError: If compiled model file doesn't exist
            AttributeError: If model loader doesn't implement required functions
        """
        self._verify_local_access()
        self.item = self._load()


        # Load the Python module containing model loading functions
        if self.module is None:
            self.module = load_python_module(self.source)

        try:            
            return self.module.display_results(Path(self.source), *args, **kwargs)
        except KeyError:
            raise KeyError("Compiled model asset not found in metadata")
        except AttributeError:
            raise AttributeError("Model loader module doesn't implement 'compiled_model' function")

    def get_model_summary(self) -> dict[str, Any]:
        """
        Returns a dictionary with key information about the model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "id": self.item.id,
            "source": self.file,
            "scheme": self.scheme,
            "framework": self.item.properties.get("mlm:framework"),
            "architecture": self.item.properties.get("mlm:architecture"),
            "tasks": self.item.properties.get("mlm:tasks", []),
            "dependencies": self.item.properties.get("dependencies"),
            "size_bytes": self.item.properties.get("file:size", 0)
        }

    def _verify_local_access(self) -> None:
        """
        Verify model is available locally before attempting to load.

        Raises:
            ValueError: If model hasn't been downloaded locally
        """
        if self.scheme != "local":
            raise ValueError(
                "The model must be downloaded locally first. "
                "Run .download(path) to download the model files."
            )

    def __repr__(self) -> str:
        """Return string representation of the ModelLoader instance."""
        self.print_schema()
        return f"ModelLoader(file='{self.file}', scheme='{self.scheme}')"

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        self.print_schema()
        return f"ModelLoader for '{self.item.id}'"


# For backward compatibility
load = ModelLoader
