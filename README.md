<p align="center">
  <img src="https://raw.githubusercontent.com/tacofoundation/mlstac/main/docs/assets/logo.png" alt="MLSTAC logo" width="220"/>
</p>

<h1 align="center">MLSTAC</h1>

<p align="center">
  <em>Machine learning model sharing, built on the STAC MLM extension and Safetensors.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/mlstac/"><img src="https://img.shields.io/pypi/v/mlstac.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/mlstac/"><img src="https://img.shields.io/pypi/pyversions/mlstac.svg" alt="Python versions"></a>
  <a href="https://github.com/tacofoundation/mlstac/actions/workflows/tests.yml"><img src="https://github.com/tacofoundation/mlstac/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/tacofoundation/mlstac/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/mlstac.svg" alt="License"></a>
  <a href="https://tacofoundation.github.io/mlstac/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
</p>

---

## Overview

MLSTAC gives you a single, consistent way to publish and consume machine learning
models. Each model is described by a STAC Item using the
[MLM extension](https://github.com/stac-extensions/mlm), and its weights travel as
[Safetensors](https://github.com/huggingface/safetensors). From that metadata you can
load a model from many backends (HTTP, local disk, S3, Google Cloud Storage), download
it, inspect it, and turn it into a usable PyTorch model.

> **Status:** experimental. The API may change between minor versions.

## Installation

```bash
pip install mlstac
```

## Quickstart

```python
import mlstac

# Load only the metadata first (no weights are downloaded yet)
model = mlstac.load("https://example.com/my-model/mlm.json")

# Inspect what you have
print(model.get_model_summary())
model.print_schema()

# Download every file into a local folder; you get back a loader
# pointing at the local copy
local = model.download("./my-model")

# Build a usable model from the local files
net = local.compiled_model()
```

### Loading from different sources

`mlstac.load` is the main entry point. It accepts a URL, a local path, or a list of
`.pt2` files for an ad-hoc ensemble.

```python
# Remote metadata
mlstac.load("https://example.com/model/mlm.json")

# A local directory that contains an mlm.json
mlstac.load("./my-model")

# An ensemble built directly from .pt2 files
ensemble = mlstac.load(["model_a.pt2", "model_b.pt2"])
ensemble.is_ensemble  # True
```

### Downloading

You can download from the loader instance, or call the module-level function directly.

```python
# From an existing loader (reuses the source it was created from)
local = model.download("./my-model")

# Or as a one-shot call
local = mlstac.download("https://example.com/model/mlm.json", "./my-model")
```

## Supported backends

| Scheme | Example |
| --- | --- |
| `http` / `https` / `ftp` | `https://example.com/model/mlm.json` |
| `local` | `./my-model` or `/abs/path/mlm.json` |
| `s3` | `s3://bucket/key/mlm.json` (needs `boto3`) |
| `gs` | `gs://bucket/key/mlm.json` (needs `google-cloud-storage`) |

## Documentation

Full docs live at **[tacofoundation.github.io/mlstac](https://tacofoundation.github.io/mlstac/)**.

## Development

```bash
# install with dev dependencies
poetry install --with dev

# run the test suite
poetry run pytest tests/ -v
```

## License

See [LICENSE](LICENSE).