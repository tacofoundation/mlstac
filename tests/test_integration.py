"""
End-to-end local integration test.

Builds a toy mlm.json with a local asset, loads it with ModelLoader and
verifies the summary and the download via the instance method. No network.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from mlstac.main import ModelLoader


def _build_model_dir(base: Path) -> Path:
    """Create a minimal model directory with mlm.json and one asset."""
    model_dir = base / "model"
    model_dir.mkdir()

    asset = model_dir / "weights.safetensor"
    asset.write_bytes(b"fake-weights")

    mlm = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": "demo_model",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": datetime.now(timezone.utc).isoformat(),
            # accented title to exercise the UTF-8 read path
            "title": "Modelo de demostración",
            "mlm:framework": "pytorch",
        },
        "links": [],
        "assets": {
            "weights": {
                # absolute href so download() can resolve it on any machine
                "href": str(asset),
                "type": "application/octet-stream",
                "roles": ["mlm:model"],
            }
        },
    }
    (model_dir / "mlm.json").write_text(
        json.dumps(mlm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model_dir


def test_load_local_directory(tmp_path):
    model_dir = _build_model_dir(tmp_path)

    loader = ModelLoader(str(model_dir))

    assert loader.scheme == "local"
    assert loader.item.id == "demo_model"
    # accents survive the UTF-8 read
    assert loader.item.properties["title"] == "Modelo de demostración"


def test_get_model_summary(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    summary = ModelLoader(str(model_dir)).get_model_summary()

    assert summary["id"] == "demo_model"


def test_instance_download_copies_assets(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    loader = ModelLoader(str(model_dir))

    out_dir = tmp_path / "downloaded"
    new_loader = loader.download(str(out_dir))

    # mlm.json and the asset both landed in the output dir
    assert (out_dir / "mlm.json").exists()
    assert (out_dir / "weights.safetensor").read_bytes() == b"fake-weights"
    # download() returns a usable loader for the local copy
    assert isinstance(new_loader, ModelLoader)
    assert new_loader.item.id == "demo_model"