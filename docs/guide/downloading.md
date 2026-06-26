# Downloading

Loading the metadata does **not** download the weights. When you are ready to pull
the files, use `download`.

## From a loader instance

The instance method reuses the source the loader was created from, so you do not
repeat the URL.

```python
model = mlstac.load("https://example.com/model/mlm.json")
local = model.download("./my-model")
```

`download` returns a **new** loader that points at the local copy, so you can chain
straight into building the model:

```python
net = model.download("./my-model").compiled_model()
```

## As a one-shot call

If you do not already have a loader, the module-level function does both steps at once.

```python
local = mlstac.download("https://example.com/model/mlm.json", "./my-model")
```

Both forms write the model metadata as UTF-8, so titles and descriptions with accents
or other non-ASCII characters round-trip correctly on every platform.
