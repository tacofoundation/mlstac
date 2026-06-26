"""
Tests for mlstac.fetch.

HTTP downloads are simulated with monkeypatch over requests, so they never
touch the network and run cleanly in CI.
"""

from pathlib import Path

import pytest

from mlstac import fetch


# --- fetch_source: text retrieval ---

def test_fetch_source_local_reads_file(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("hello world", encoding="utf-8")
    assert fetch.fetch_source(str(file)) == "hello world"


def test_fetch_source_http_uses_requests(monkeypatch):
    class FakeResponse:
        text = "remote content"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        assert url == "https://example.com/file.json"
        return FakeResponse()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    assert fetch.fetch_source("https://example.com/file.json") == "remote content"


def test_fetch_source_snippet_clear_message():
    # a bare name is neither a URL nor an existing local path
    with pytest.raises(ValueError, match="Could not resolve"):
        fetch.fetch_source("resnet50")


# --- download_file: writing to disk ---

def test_download_file_local_copies(tmp_path):
    src = tmp_path / "src" / "model.safetensor"
    src.parent.mkdir()
    src.write_bytes(b"weights")

    dest_dir = tmp_path / "out"
    out = fetch.download_file(str(src), outpath=dest_dir)

    assert out == dest_dir / "model.safetensor"
    assert out.read_bytes() == b"weights"


def test_download_file_http_streaming(monkeypatch, tmp_path):
    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"chunk1"
            yield b"chunk2"

    def fake_get(url, stream, timeout):
        return FakeStream()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    out = fetch.download_file("https://example.com/w.bin", outpath=tmp_path)

    assert out == tmp_path / "w.bin"
    assert out.read_bytes() == b"chunk1chunk2"


def test_download_file_snippet_clear_message(tmp_path):
    # download_file wraps its errors in RuntimeError, but the clear
    # "Could not resolve" message is preserved inside it.
    with pytest.raises(RuntimeError, match="Could not resolve"):
        fetch.download_file("resnet50", outpath=tmp_path)