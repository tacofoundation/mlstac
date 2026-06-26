"""
_quiet.py - Silence HuggingFace and other warnings on import.

Silences:
  • HuggingFace HF_TOKEN warnings in Colab
  • tqdm nested progress bar warnings
  • fsspec/requests warnings
"""

from __future__ import annotations

import os
import warnings

# =============================================================================
# 1. ENVIRONMENT VARIABLES (must be set BEFORE importing huggingface_hub)
# =============================================================================

# Disable HuggingFace progress bars (optional, cleaner output)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Disable tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =============================================================================
# 2. WARNING FILTERS
# =============================================================================

# --- HuggingFace Hub ---
# This is the main one causing the issue in Colab
warnings.filterwarnings(
    "ignore",
    message=r".*HF_TOKEN.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Hugging Face Hub.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*huggingface.*token.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module=r"huggingface_hub.*",
    category=UserWarning,
)

# --- tqdm ---
try:
    from tqdm import TqdmWarning

    warnings.filterwarnings("ignore", category=TqdmWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module=r"tqdm(\.|$)")

# --- fsspec ---
warnings.filterwarnings("ignore", category=UserWarning, module=r"fsspec(\.|$)")

# --- requests/urllib3 ---
warnings.filterwarnings("ignore", category=UserWarning, module=r"requests(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"urllib3(\.|$)")

# --- PyTorch (in case models use it) ---
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")
