# MLSTAC: Machine Learning with STAC

[![PyPI version](https://img.shields.io/pypi/v/mlstac.svg)](https://pypi.org/project/mlstac/)
[![Python Versions](https://img.shields.io/pypi/pyversions/mlstac.svg)](https://pypi.org/project/mlstac/)
[![License](https://img.shields.io/pypi/l/mlstac.svg)](https://github.com/csaybar/isp-models/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://csaybar.github.io/isp-models/)


We take advantage of the new mlm STAC extension to provide a unified interface for working with machine learning models.
**Experimental**


## Installation

```bash
pip install mlstac
```

## Quick Start

```python
import mlstac
import matplotlib.pyplot as plt


# Download model
file="https://huggingface.co/tacofoundation/supers2/resolve/main/CNN_Light_SR/mlm.json"
output_dir="models2/CNN_Light_SR"
mlstac.download(file, output_dir)

# Create a mlstac object
mlstac_object = mlstac.load(output_dir)
device = "cpu" # "cpu"

# Load model
#srmodel = mlstac_object.trainable_model() # for fine-tuning
srmodel = mlstac_object.compiled_model(device=device) # for benchmarking

# Load Demo Data
lr, hr = mlstac_object.example_data()

# Inference
sr = srmodel(lr.to(device))


# Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(lr[0, 0:3].permute(1, 2, 0)*3)
ax[0].set_title("Low Resolution")
ax[1].imshow(hr[0, 0:3].permute(1, 2, 0)*3)
ax[1].set_title("High Resolution")
ax[2].imshow(sr[0, 0:3].permute(1, 2, 0)*3)
ax[2].set_title("Super Resolution")
plt.show()


# Fast plot
fig = mlstac_object.display_results()
plt.show()
```

