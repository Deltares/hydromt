from pathlib import Path
from typing import Optional

import pytest

from hydromt.data_catalog.predefined_catalog import (
    PredefinedCatalog,
    _valid_key,
    create_registry_file,
)


def _make_catalog(tmp_path: Path, specs: dict[str, Optional[str]]) -> PredefinedCatalog:
    root = tmp_path / "src"
    for version, spec in specs.items():
        folder = root / version
        folder.mkdir(parents=True)
        meta = f"meta:\n  hydromt_version: '{spec}'\n" if spec else "meta: {}\n"
        (folder / "data_catalog.yml").write_text(meta + "src:\n  uri: x\n")
    create_registry_file(root)

    class _Catalog(PredefinedCatalog):
        base_url = root.as_posix()
        name = "test_catalog"

    return _Catalog(format_version="v1", cache_dir=tmp_path / "cache")


_SPECS: dict[str, Optional[str]] = {
    "v1.0.0": ">1.0a,<2",
    "v2.0.0": ">=1.5.0,<2",
}


def test_latest_picks_newest_compatible(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hydromt.__version__", "1.5.0")
    catalog = _make_catalog(tmp_path, _SPECS)
    assert catalog._latest_compatible_version() == "v2.0.0"


def test_latest_skips_incompatible_version(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hydromt.__version__", "1.4.0")
    catalog = _make_catalog(tmp_path, _SPECS)
    assert catalog._latest_compatible_version() == "v1.0.0"
    assert catalog.get_catalog_file().parent.name == "v1.0.0"


def test_latest_without_spec_is_compatible(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hydromt.__version__", "1.4.0")
    catalog = _make_catalog(tmp_path, {"v1.0.0": None, "v2.0.0": None})
    assert catalog._latest_compatible_version() == "v2.0.0"


def test_latest_raises_when_none_compatible(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hydromt.__version__", "1.4.0")
    catalog = _make_catalog(tmp_path, {"v1.1.0": ">=1.5.0,<2"})
    with pytest.raises(RuntimeError, match="compatible"):
        catalog._latest_compatible_version()


def test_explicit_version_skips_compat_check(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hydromt.__version__", "1.4.0")
    catalog = _make_catalog(tmp_path, _SPECS)
    assert catalog.get_catalog_file("v2.0.0").parent.name == "v2.0.0"


def test_valid_key_v1_format_accepts_v1_and_v2():
    assert _valid_key("v1.0.0/data_catalog.yml", "v1")
    assert _valid_key("v2.0.0/data_catalog.yml", "v1")


def test_valid_key_v0_format_only_accepts_v0():
    assert _valid_key("v0.1.0/data_catalog.yml", "v0")
    assert not _valid_key("v1.0.0/data_catalog.yml", "v0")
    assert not _valid_key("v2.0.0/data_catalog.yml", "v0")


def test_valid_key_v1_format_rejects_v0():
    assert not _valid_key("v0.1.0/data_catalog.yml", "v1")


def test_valid_key_v1_format_does_not_prefix_match_v10():
    # major 10 is >= 1 by design; the old startswith("v1") matched it by accident
    assert _valid_key("v10.0.0/data_catalog.yml", "v1")


def test_valid_key_rejects_invalid():
    assert not _valid_key("not_a_version/data_catalog.yml", "v1")
    assert not _valid_key("registry.txt", "v1")
