"""
MLSTAC main module.

This module provides a clean interface for working with MLSTAC models through STAC metadata,
supporting multiple storage backends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec
import pystac
from tqdm import tqdm

from mlstac.fetch import download_file, load_python_module, load_stac_item
from mlstac.utils import get_scheme

if TYPE_CHECKING:
    import numpy as np
    import torch


def download(file: Path | str, output_dir: Path | str) -> ModelLoader:
    """
    Download model files of a specified model from a remote source.

    Args:
        file: URL or path to the MLM-STAC JSON metadata file
        output_dir: Target directory for downloaded files

    Returns:
        ModelLoader instance pointing to the downloaded model

    Raises:
        RuntimeError: If any of the specified files fail to download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item = load_stac_item(file)

    json_filename = Path(file).name if Path(file).suffix == ".json" else "mlm.json"
    mlm_file = output_dir / json_filename
    with open(mlm_file, "w", encoding="utf-8") as f:
        json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)

    to_download = {}
    for key, value in item.assets.items():
        to_download[key] = value.href

    downloaded_files = []
    for name, source in tqdm(to_download.items(), desc="Downloading files"):
        try:
            file_path = download_file(source=source, outpath=output_dir)
            downloaded_files.append(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {name}: {e!s}") from e

    return ModelLoader(mlm_file.absolute().as_posix())


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

    def __init__(self, file: str | list):
        """
        Initialize the model manager.

        Args:
            file: The JSON file that contains the model metadata, a directory path,
                or a list of .pt2 model files for ad-hoc ensemble creation

        Raises:
            ValueError: If the source cannot be resolved or the model cannot be loaded.
        """
        self.file = file
        self.scheme = get_scheme(file)

        if self.scheme == "pt2_list":
            self.item = self._create_minimal_stac_from_pt2s(file)
            self.source = str(Path(file[0]).parent)

        elif self.scheme == "local":
            if Path(file).is_dir():
                self.file = Path(file) / "mlm.json"
            self.item = self._load()
            self.source = Path(self.file).parent.as_posix()
        else:
            self.item = load_stac_item(file)
            self.source = None

        self.module = None
        self.status = "downloaded"
        self.device = None

    def download(self, output_dir: Path | str) -> ModelLoader:
        """
        Download this model's files into a local directory.

        Thin wrapper around the module-level download(): it resolves the
        assets from this loader's metadata source and returns a new loader
        pointing at the local copy.

        Args:
            output_dir: Target directory for the downloaded files

        Returns:
            A ModelLoader for the downloaded model
        """
        # `download` here is the module-level function, not this method.
        return download(self.file, output_dir)

    @property
    def thr(self) -> float | None:
        """
        Get the recommended threshold value for the model output.

        Returns:
            Recommended threshold value, or None if not available
        """
        mlm_output = self.item.properties.get("mlm:output", [{}])
        if mlm_output and mlm_output[0]:
            return mlm_output[0].get("recommended_threshold")
        return None

    @property
    def is_ensemble(self) -> bool:
        """
        Check if this is an ensemble model requiring runtime aggregation.

        An ensemble model is one that requires loading multiple .pt2 files
        and aggregating them at runtime (mean/max/min).

        A pre-fused ensemble (single .pt2 with embedded aggregation) is NOT
        considered an ensemble for runtime purposes.
        """
        if self.item.properties.get("custom:ensemble_fused", False):
            return False

        pt2_count = sum(
            1 for asset in self.item.assets.values() if asset.href.endswith(".pt2")
        )

        return pt2_count > 1

    @property
    def model(self):
        """
        Convenience property to get the compiled model (single models only).
        For ensembles, use compiled_model(mode=...) instead.

        Example:
            >>> # Single model - quick access
            >>> model = loader.model
        """
        if self.is_ensemble:
            raise AttributeError(
                "Cannot use .model property for ensemble models. "
                "Use .compiled_model(mode='mean'|'max'|'min') instead."
            )
        return self.compiled_model()

    def _verify_local_access(self) -> None:
        """
        Verify model is available locally before attempting to load.

        Raises:
            ValueError: If model hasn't been downloaded locally
        """
        if self.scheme not in {"local", "pt2_list"}:
            raise ValueError(
                "The model must be downloaded locally first. "
                "Run .download(path) to download the model files."
            )

    def _create_minimal_stac_from_pt2s(self, model_paths: list) -> pystac.Item:
        """
        Create minimal STAC metadata from a list of .pt2 model files.
        """
        from datetime import datetime, timezone

        model_paths = [Path(p) for p in model_paths]

        for p in model_paths:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            if p.suffix != ".pt2":
                raise ValueError(f"Expected .pt2 file, got: {p}")

        assets = {}
        for i, model_path in enumerate(sorted(model_paths), start=1):
            asset_key = f"model_{i}_{model_path.stem}"
            assets[asset_key] = {
                "href": str(model_path.absolute()),
                "type": "application/octet-stream",
                "title": f"Model {i}: {model_path.name}",
                "roles": ["mlm:model", "mlm:weights"],
            }

        stac_dict = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": f"ENSEMBLE_ADHOC_{len(model_paths)}_MODELS",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[-180, -90], [-180, 90], [180, 90], [180, -90], [-180, -90]]
                ],
            },
            "bbox": [-180, -90, 180, 90],
            "properties": {
                "datetime": datetime.now(timezone.utc).isoformat(),
                "title": f"Ad-hoc Ensemble ({len(model_paths)} models)",
                "description": f"Ensemble from {len(model_paths)} .pt2 files - no metadata available",
                "mlm:name": "adhoc_ensemble",
                "mlm:architecture": f"Ensemble of {len(model_paths)} models",
                "mlm:framework": "pytorch",
            },
            "links": [],
            "assets": assets,
        }

        return pystac.Item.from_dict(stac_dict)

    def print_schema(self) -> None:
        """
        Prints a visually appealing schema of the model.

        Automatically detects if running in a Jupyter/Colab notebook or terminal
        and formats the output accordingly.
        """
        in_notebook = "ipykernel" in sys.modules

        model_id = self.item.id
        title = self.item.properties.get("title", "Untitled Model")
        description = self.item.properties.get(
            "description", "No description available"
        )

        framework = self.item.properties.get("mlm:framework", "Not specified")
        framework_version = self.item.properties.get("mlm:framework_version", "")
        architecture = self.item.properties.get("mlm:architecture", "Not specified")
        tasks = self.item.properties.get("mlm:tasks", [])

        total_params = self.item.properties.get("mlm:total_parameters", 0)
        params_m = f"{total_params / 1_000_000:.2f}M" if total_params else "Unknown"

        file_size = self.item.properties.get("file:size", 0)
        file_size_mb = f"{file_size / (1024 * 1024):.2f} MB" if file_size else "Unknown"

        sensors = self.item.properties.get("custom:sensors", [])
        spatial_res = self.item.properties.get("custom:spatial_resolution", "Unknown")
        project = self.item.properties.get("custom:project", "")
        project_url = self.item.properties.get("custom:project_url", "")

        hyperparams = self.item.properties.get("mlm:hyperparameters", {})

        if hyperparams is None:
            hyperparams = {}

        learning_rate = hyperparams.get("learning_rate", "N/A")
        batch_size = hyperparams.get("batch_size", "N/A")
        epochs = hyperparams.get("training_epochs", "N/A")
        val_loss = hyperparams.get("final_val_loss", "N/A")

        mlm_input = self.item.properties.get("mlm:input", [{}])
        if mlm_input is None:
            mlm_input = [{}]

        input_shape = mlm_input[0].get("input", {}).get("shape", [])
        input_bands = mlm_input[0].get("bands", [])

        mlm_output = self.item.properties.get("mlm:output", [{}])

        if mlm_output is None:
            mlm_output = [{}]

        output_shape = mlm_output[0].get("result", {}).get("shape", [])
        standard_threshold = mlm_output[0].get("standard_threshold", "N/A")
        recommended_threshold = mlm_output[0].get("recommended_threshold", "N/A")

        links = {link.rel: link.href for link in self.item.links}

        dependencies = self.item.properties.get("dependencies", [])
        deps_str = (
            ", ".join([d.split(">=")[0] for d in dependencies[:3]])
            if dependencies
            else "None"
        )

        if in_notebook:
            from IPython.display import HTML, display

            html_content = f"""
            <style>
                .mlstac-container {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                    padding: 15px;
                    border-radius: 10px;
                    color: white;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                    margin: 15px 0;
                }}
                .mlstac-header {{
                    text-align: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid rgba(255,255,255,0.3);
                }}
                .mlstac-header h2 {{
                    margin: 0 0 5px 0;
                    font-size: 24px;
                    font-weight: 600;
                }}
                .mlstac-header p {{
                    margin: 0;
                    font-size: 13px;
                    opacity: 0.9;
                }}
                .mlstac-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 10px;
                    margin-bottom: 15px;
                }}
                .mlstac-card {{
                    background: rgba(255, 255, 255, 0.15);
                    backdrop-filter: blur(10px);
                    border-radius: 8px;
                    padding: 12px;
                    border: 1px solid rgba(255,255,255,0.2);
                    transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
                }}
                .mlstac-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                }}
                .mlstac-card h3 {{
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    font-weight: 600;
                    opacity: 0.95;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }}
                .mlstac-card-content {{
                    font-size: 13px;
                    line-height: 1.5;
                }}
                .mlstac-card-content p {{
                    margin: 4px 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .mlstac-card-content .label {{
                    opacity: 0.8;
                    font-weight: 500;
                }}
                .mlstac-card-content .value {{
                    background: rgba(255,255,255,0.2);
                    padding: 2px 6px;
                    border-radius: 5px;
                    font-weight: 600;
                    text-align: right;
                    font-size: 13px;
                }}
                .mlstac-badge {{
                    display: inline-block;
                    background: rgba(255,255,255,0.25);
                    padding: 3px 8px;
                    border-radius: 15px;
                    font-size: 11px;
                    margin: 2px;
                    font-weight: 500;
                }}
                .mlstac-description {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    font-size: 12px;
                    line-height: 1.5;
                    border-left: 3px solid rgba(255,255,255,0.4);
                }}
                .mlstac-footer {{
                    text-align: center;
                    margin-top: 15px;
                    padding-top: 10px;
                    border-top: 1px solid rgba(255,255,255,0.3);
                    font-size: 12px;
                    opacity: 0.9;
                }}
                .mlstac-footer a {{
                    color: white;
                    text-decoration: none;
                    font-weight: 600;
                    border-bottom: 1px solid rgba(255,255,255,0.5);
                    transition: border-color 0.2s;
                }}
                .mlstac-footer a:hover {{
                    border-bottom-color: white;
                }}
                .icon {{
                    font-size: 16px;
                }}
            </style>
            <div class="mlstac-container">
                <div class="mlstac-header">
                    <h2>🚀 {title}</h2>
                    <p>Model ID: <strong>{model_id}</strong></p>
                </div>
                <div class="mlstac-description">
                    {description}
                </div>
                <div class="mlstac-grid">
                    <div class="mlstac-card">
                        <h3><span class="icon">🛠️</span> Framework & arch.</h3>
                        <div class="mlstac-card-content">
                            <p><span class="label">Framework:</span> <span class="value">{framework} {framework_version[:6]}</span></p>
                            <p><span class="label">Architecture:</span> <span class="value">{architecture}</span></p>
                            <p><span class="label">Parameters:</span> <span class="value">{params_m}</span></p>
                            <p><span class="label">Model Size:</span> <span class="value">{file_size_mb}</span></p>
                        </div>
                    </div>
                    <div class="mlstac-card">
                        <h3><span class="icon">🛰️</span> Data specs</h3>
                        <div class="mlstac-card-content">
                            <p><span class="label">Spatial Res:</span> <span class="value">{spatial_res}</span></p>
                            <p><span class="label">Input Shape:</span> <span class="value">{input_shape}</span></p>
                            <p><span class="label">Bands:</span> <span class="value">{len(input_bands)}</span></p>
                            <p><span class="label">Sensors:</span></p>
                            <div style="margin-left: 5px; text-align: right;">
                                {''.join([f'<span class="mlstac-badge">{s}</span>' for s in sensors])}
                            </div>
                        </div>
                    </div>
                    <div class="mlstac-card">
                        <h3><span class="icon">📊</span> Training metrics</h3>
                        <div class="mlstac-card-content">
                            <p><span class="label">Learning Rate:</span> <span class="value">{learning_rate}</span></p>
                            <p><span class="label">Batch Size:</span> <span class="value">{batch_size}</span></p>
                            <p><span class="label">Epochs:</span> <span class="value">{epochs}</span></p>
                            <p><span class="label">Val Loss:</span> <span class="value">{val_loss}</span></p>
                        </div>
                    </div>
                    <div class="mlstac-card">
                        <h3><span class="icon">🎯</span> Tasks & output</h3>
                        <div class="mlstac-card-content">
                            <p><span class="label">Output Shape:</span> <span class="value">{output_shape}</span></p>
                            <p><span class="label">Std Threshold:</span> <span class="value">{standard_threshold}</span></p>
                            <p><span class="label">Rec. Threshold:</span> <span class="value">{recommended_threshold}</span></p>
                            <p><span class="label">Dependencies:</span> <span class="value">{deps_str}</span></p>
                            <p><span class="label">Tasks:</span></p>
                            <div style="margin-left: 5px; text-align: right;">
                                {''.join([f'<span class="mlstac-badge">{t}</span>' for t in tasks])}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mlstac-footer">
                    {'<strong>' + project + '</strong> | ' if project else ''}<a href="{project_url}" target="_blank">Project Info</a>{' | <a href="' + links.get('license', '#') + '" target="_blank">License</a>' if 'license' in links else ''} | Source: <strong>{self.scheme.upper()}</strong> | Status: <strong>{self.status.capitalize()}</strong>
                </div>
            </div>
            """
            display(HTML(html_content))

        else:
            CYAN = "\033[96m"
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            BOLD = "\033[1m"
            RESET = "\033[0m"
            DIM = "\033[2m"

            print(f"\n{CYAN}{BOLD}🚀 {title}{RESET}")
            print(f"{DIM}   ID: {model_id}{RESET}")

            print(f"{BLUE}   {description}{RESET}\n")

            print(f"{GREEN}{BOLD}🛠️  Framework & architecture{RESET}")
            print(
                f"   Framework:    {YELLOW}{framework} {framework_version[:10]}{RESET}"
            )
            print(f"   Architecture: {YELLOW}{architecture}{RESET}")
            print(f"   Parameters:   {YELLOW}{params_m}{RESET}")
            print(f"   Model Size:   {YELLOW}{file_size_mb}{RESET}")

            print(f"{GREEN}{BOLD}🛰️  Data specifications{RESET}")
            print(f"   Sensors:      {YELLOW}{', '.join(sensors)}{RESET}")
            print(f"   Spatial Res:  {YELLOW}{spatial_res}{RESET}")
            print(f"   Input Shape:  {YELLOW}{input_shape}{RESET}")
            print(f"   Bands:        {YELLOW}{len(input_bands)} bands{RESET}")

            print(f"{GREEN}{BOLD}📊 Training metrics{RESET}")
            print(f"   Learning Rate: {YELLOW}{learning_rate}{RESET}")
            print(f"   Batch Size:    {YELLOW}{batch_size}{RESET}")
            print(f"   Epochs:        {YELLOW}{epochs}{RESET}")
            print(f"   Val Loss:     {YELLOW}{val_loss}{RESET}")

            print(f"{GREEN}{BOLD}🎯 Tasks & output{RESET}")
            print(f"   Tasks:            {YELLOW}{', '.join(tasks)}{RESET}")
            print(f"   Output Shape:     {YELLOW}{output_shape}{RESET}")
            print(f"   Std Threshold:    {YELLOW}{standard_threshold}{RESET}")
            print(f"   Rec. Threshold:   {YELLOW}{recommended_threshold}{RESET}")
            print(f"   Dependencies:     {YELLOW}{deps_str}{RESET}")

            status_str = f"| Status: {self.status.capitalize()} "
            print(
                f"\n{DIM}   {project} | Source: {self.scheme.upper()} {status_str}{RESET}"
            )
            if project_url:
                print(f"{DIM}   🔗 {project_url}{RESET}\n")

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
            with fsspec.open(self.file, "r", encoding="utf-8") as f:
                mlm_data = json.load(f)

                for key, value in mlm_data["assets"].items():
                    filename = Path(value["href"]).name
                    mlm_data["assets"][key]["href"] = (
                        (Path(self.file).parent / filename).absolute().as_posix()
                    )

            return pystac.item.Item.from_dict(mlm_data)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model metadata file not found at {self.file}"
            ) from e
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
        self._verify_local_access()

        try:
            if self.module is None:
                self.module = load_python_module(self.source)
            return self.module.example_data(Path(self.source), *args, **kwargs)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Example data file not found at {self.source}/example_data.safetensor"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                "Model loader module doesn't implement 'example_data' function"
            ) from e

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

        if self.module is None:
            self.module = load_python_module(self.source)

        try:
            return self.module.trainable_model(Path(self.source), *args, **kwargs)
        except KeyError as e:
            raise KeyError("Trainable model asset not found in metadata") from e
        except AttributeError as e:
            raise AttributeError(
                "Model loader module doesn't implement 'trainable_model' function"
            ) from e

    def compiled_model(self, **kwargs):
        """
        Load the compiled model for inference.

        For single models: No parameters needed
        For ensembles: Accepts mode='mean'|'median'|'max'|'min' (default: 'max')

        Returns:
            Compiled model instance ready for inference

        Example:
            >>> # Single model
            >>> model = loader.compiled_model()
            >>>
            >>> # Ensemble model
            >>> model = loader.compiled_model(mode="mean")
            >>> model = loader.compiled_model(mode="median")  # More robust to outliers
            >>> model = loader.compiled_model(mode="max")     # Conservative (more clouds)
        """
        self._verify_local_access()

        if self.scheme == "local":
            self.item = self._load()

        if self.module is None:
            self.module = load_python_module(self.source)

        if self.is_ensemble and "mode" not in kwargs:
            kwargs["mode"] = "max"

        return self.module.compiled_model(
            Path(self.source), stac_item=self.item, **kwargs
        )

    def predict_large(
        self,
        image: np.ndarray,
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        """
        Predict on large arrays using overlapping tiles.

        Args:
            image: Input array with shape (C, H, W)
            model: Pre-loaded model (optional, will load if not provided)
            chunk_size: Size of inference tiles (default: 512)
            overlap: Overlap between tiles (default: 64)
            device: 'cpu' or 'cuda' (default: 'cpu')
            nodata: No-data value (default: 0.0)

        Returns:
            - For ensembles: Tuple of (probabilities, uncertainty), both (1, H, W)
            - For single models: probabilities array (1, H, W)

        Example:
            >>> model = loader.compiled_model()
            >>> result = loader.predict_large(image, model=model, device="cuda")
        """
        self._verify_local_access()

        if self.module is None:
            self.module = load_python_module(self.source)

        if model is None:
            model = self.compiled_model()

        return self.module.predict_large(image=image, model=model, **kwargs)

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

        if self.scheme == "local":
            self.item = self._load()

        if self.module is None:
            self.module = load_python_module(self.source)

        try:
            return self.module.display_results(
                Path(self.source), *args, stac_item=self.item, **kwargs
            )
        except KeyError as e:
            raise KeyError("Compiled model asset not found in metadata") from e
        except AttributeError as e:
            raise AttributeError(
                "Model loader module doesn't implement 'compiled_model' function"
            ) from e

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
            "size_bytes": self.item.properties.get("file:size", 0),
        }

    def __repr__(self) -> str:
        """Return string representation of the ModelLoader instance."""
        self.print_schema()
        return ""

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        self.print_schema()
        return ""


load = ModelLoader
