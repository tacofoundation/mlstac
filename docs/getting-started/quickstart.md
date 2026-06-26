# Quickstart

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

That is the whole loop: **load metadata, inspect, download, build**. The next pages
break down each step.
