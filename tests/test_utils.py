"""
Tests for mlstac.utils and the pt2_list path of ModelLoader.

These do not touch the network or download anything, so they run fast and
serve as a base to build the rest on. Run with: pytest tests/test_utils.py -v
"""

from pathlib import Path

import pytest

from mlstac.utils import get_scheme


# --- get_scheme: scheme detection ---

@pytest.mark.parametrize(
    "source, expected",
    [
        ("https://example.com/model.safetensor", "https"),
        ("http://example.com/x", "http"),
        ("ftp://example.com/x", "ftp"),
        ("s3://bucket/key", "s3"),
        ("gs://bucket/key", "gs"),
        ("resnet50", "snippet"),  # bare name, does not exist locally
    ],
)
def test_get_scheme_remote_and_snippet(source, expected):
    assert get_scheme(source) == expected


def test_get_scheme_list_is_pt2_list():
    assert get_scheme(["m1.pt2", "m2.pt2"]) == "pt2_list"


def test_get_scheme_existing_local_path(tmp_path):
    file = tmp_path / "mlm.json"
    file.write_text("{}", encoding="utf-8")
    assert get_scheme(str(file)) == "local"


def test_get_scheme_accepts_path_object(tmp_path):
    # get_scheme must accept Path, not only str
    file = tmp_path / "mlm.json"
    file.write_text("{}", encoding="utf-8")
    assert get_scheme(file) == "local"


# --- ModelLoader: pt2_list path (ad-hoc ensemble) ---

def test_pt2_list_builds_minimal_stac(tmp_path):
    from mlstac.main import ModelLoader

    pt2_a = tmp_path / "a.pt2"
    pt2_b = tmp_path / "b.pt2"
    pt2_a.write_bytes(b"fake")
    pt2_b.write_bytes(b"fake")

    loader = ModelLoader([str(pt2_a), str(pt2_b)])

    assert loader.scheme == "pt2_list"
    assert loader.is_ensemble is True
    assert len(loader.item.assets) == 2


def test_pt2_list_missing_file_raises(tmp_path):
    from mlstac.main import ModelLoader

    with pytest.raises(FileNotFoundError):
        ModelLoader([str(tmp_path / "does_not_exist.pt2")])


def test_pt2_list_wrong_extension_raises(tmp_path):
    from mlstac.main import ModelLoader

    bad = tmp_path / "model.bin"
    bad.write_bytes(b"fake")
    with pytest.raises(ValueError):
        ModelLoader([str(bad)])


# --- Local access guard ---

def test_compiled_model_remote_requires_download():
    from mlstac.main import ModelLoader

    # a remote item cannot be compiled without downloading first
    loader = ModelLoader.__new__(ModelLoader)
    loader.scheme = "https"
    loader.module = None
    with pytest.raises(ValueError, match="downloaded locally"):
        loader._verify_local_access()