# Loading models

`mlstac.load` is the main entry point. It is an alias for `ModelLoader`, so both of
these are equivalent:

```python
import mlstac

model = mlstac.load("./my-model")
model = mlstac.ModelLoader("./my-model")
```

## Accepted sources

`load` figures out how to reach the model from the value you pass in.

| You pass | Detected as |
| --- | --- |
| `https://...`, `http://...`, `ftp://...` | remote HTTP(S)/FTP |
| `s3://bucket/key` | Amazon S3 (needs `boto3`) |
| `gs://bucket/key` | Google Cloud Storage (needs `google-cloud-storage`) |
| an existing local path | local files |
| a list of `.pt2` files | an ad-hoc ensemble |

```python
# Remote metadata
mlstac.load("https://example.com/model/mlm.json")

# A local directory containing an mlm.json
mlstac.load("./my-model")

# An ensemble from .pt2 files
ensemble = mlstac.load(["model_a.pt2", "model_b.pt2"])
ensemble.is_ensemble  # True
```

!!! note
    A bare name like `"resnet50"` is **not** a valid source on its own. Pass a full
    URL or a local path. If you pass an unresolvable name you get a clear error
    explaining what happened.

## Inspecting a model

```python
model = mlstac.load("./my-model")

model.get_model_summary()   # dict with id and key metadata
model.print_schema()        # human-readable schema
model.is_ensemble           # True for multi-model bundles
```
