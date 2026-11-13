from pathlib import Path

import pytest

from hydromt._utils.catalog_format import (
    LATEST_CATALOG_FORMAT,
    catalog_format,
    check_catalog_format,
)
from hydromt.data_catalog.predefined_catalog import (
    PredefinedCatalog,
    _valid_key,
    create_registry_file,
)


def _make_catalog(tmp_path: Path, formats: dict[str, int | None]) -> PredefinedCatalog:
    root = tmp_path / "src"
    for version, fmt in formats.items():
        folder = root / version
        folder.mkdir(parents=True)
        meta = f"meta:\n  format_version: {fmt}\n" if fmt else "meta: {}\n"
        (folder / "data_catalog.yml").write_text(meta + "src:\n  uri: x\n")
    create_registry_file(root)

    class _Catalog(PredefinedCatalog):
        base_url = root.as_posix()
        name = "test_catalog"

    return _Catalog(format_version="v1", cache_dir=tmp_path / "cache")


def test_latest_picks_newest_version(tmp_path: Path):
    catalog = _make_catalog(tmp_path, {"v1.0.0": None, "v1.2.0": 1, "v2.0.0": 2})
    assert catalog.get_catalog_file().parent.name == "v2.0.0"


def test_latest_skips_unsupported_newer_format(tmp_path: Path):
    catalog = _make_catalog(
        tmp_path, {"v1.0.0": None, "v2.0.0": LATEST_CATALOG_FORMAT + 1}
    )
    assert catalog.get_catalog_file().parent.name == "v1.0.0"


def test_explicit_version_with_unsupported_format_is_not_skipped(tmp_path: Path):
    catalog = _make_catalog(tmp_path, {"v3.0.0": LATEST_CATALOG_FORMAT + 1})
    assert catalog.get_catalog_file("v3.0.0").parent.name == "v3.0.0"


def test_latest_raises_when_no_supported_version(tmp_path: Path):
    catalog = _make_catalog(tmp_path, {"v3.0.0": LATEST_CATALOG_FORMAT + 1})
    with pytest.raises(RuntimeError, match="can be read by this hydromt version"):
        catalog.get_catalog_file()


def test_valid_key_v1_format_accepts_v1_and_v2():
    assert _valid_key("v1.0.0/data_catalog.yml", "v1")
    assert _valid_key("v2.0.0/data_catalog.yml", "v1")


def test_valid_key_v0_format_only_accepts_v0():
    assert _valid_key("v0.1.0/data_catalog.yml", "v0")
    assert not _valid_key("v1.0.0/data_catalog.yml", "v0")
    assert not _valid_key("v2.0.0/data_catalog.yml", "v0")


def test_valid_key_v1_format_rejects_v0():
    assert not _valid_key("v0.1.0/data_catalog.yml", "v1")


def test_valid_key_rejects_invalid():
    assert not _valid_key("not_a_version/data_catalog.yml", "v1")
    assert not _valid_key("registry.txt", "v1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2, 2), ("2", 2), ("v1", 1)],
)
def test_catalog_format(value, expected):
    assert catalog_format(value) == expected


def test_catalog_format_rejects_non_integer():
    with pytest.raises(ValueError, match="Invalid catalog format_version"):
        catalog_format("2024-draft")


def test_check_catalog_format_accepts_supported():
    assert check_catalog_format(1) == 1
    assert check_catalog_format(LATEST_CATALOG_FORMAT) == LATEST_CATALOG_FORMAT
    assert check_catalog_format(None) == 1  # unset: assume v1


def test_check_catalog_format_rejects_newer():
    with pytest.raises(RuntimeError, match="not supported by hydromt"):
        check_catalog_format(LATEST_CATALOG_FORMAT + 1, "my catalog")


def test_check_catalog_format_rejects_legacy_with_upgrade_help():
    with pytest.raises(RuntimeError, match="hydromt check .* --upgrade"):
        check_catalog_format(0)
