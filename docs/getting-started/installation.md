# Installation

MLSTAC is published on PyPI.

```bash
pip install mlstac
```

## Optional backends

Cloud sources need an extra package:

```bash
# Amazon S3
pip install boto3

# Google Cloud Storage
pip install google-cloud-storage
```

## From source

```bash
git clone https://github.com/tacofoundation/mlstac
cd mlstac
poetry install --with dev
```
