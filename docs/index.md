# MLSTAC

<p style="font-size: 1.15rem; opacity: 0.85; margin-top: -0.4rem;">
A single, consistent way to publish and consume machine learning models, built on the
<a href="https://github.com/stac-extensions/mlm">STAC MLM extension</a> and
<a href="https://github.com/huggingface/safetensors">Safetensors</a>.
</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Quickstart](getting-started/quickstart.md){ .md-button }

!!! note "Experimental"
    The API may still change between minor versions.

---

## Why MLSTAC

<div class="grid cards" markdown>

-   :material-layers-triple:{ .lg .middle } **Models as metadata**

    ---

    Every model is a STAC Item described with the MLM extension. The weights travel
    as Safetensors, so the description and the data stay in sync.

-   :material-cloud-download:{ .lg .middle } **Many backends, one call**

    ---

    Load from HTTP(S), local disk, Amazon S3 or Google Cloud Storage. `mlstac.load`
    figures out how to reach the model for you.

-   :material-flash:{ .lg .middle } **Metadata first, weights later**

    ---

    Inspect a model before downloading a single byte. Pull the files only when you
    are ready, straight from the loader.

-   :material-puzzle:{ .lg .middle } **Ensembles out of the box**

    ---

    Point at a list of `.pt2` files and MLSTAC builds an ad-hoc ensemble with a
    minimal STAC description on the fly.

</div>

---

## Install

```bash
pip install mlstac
```

## In 30 seconds

```python
import mlstac

# 1. Load only the metadata (no weights yet)
model = mlstac.load("https://example.com/my-model/mlm.json")

# 2. Look before you leap
print(model.get_model_summary())
model.print_schema()

# 3. Download, then build a usable model
net = model.download("./my-model").compiled_model()
```

---

## Explore the docs

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install MLSTAC and run your first load.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Loading from any source, and downloading the right way.

    [:octicons-arrow-right-24: Loading models](guide/loading-models.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Every class and function, generated from the source.

    [:octicons-arrow-right-24: ModelLoader](api/modelloader.md)

</div>
