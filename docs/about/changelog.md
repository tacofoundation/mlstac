# Changelog

## Unreleased

- Write and read model metadata as UTF-8 so non-ASCII content round-trips on Windows.
- Add `ModelLoader.download()` as an instance method.
- `get_scheme` now accepts `Path` objects, not only strings.
- Corrected type hints (`download` returns a `ModelLoader`; `get_scheme` reports
  `pt2_list`).
- Clearer error message when a source cannot be resolved.
- Added a test suite and continuous integration.

## 0.4.5

- Last release recovered from PyPI.
