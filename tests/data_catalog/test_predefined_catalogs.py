from pathlib import Path

import pytest

from hydromt.data_catalog.predefined_catalog import (
    ArtifactDataCatalog,
    PredefinedCatalog,
    create_registry_file,
)


@pytest.fixture()
def cat_root() -> Path:
    return Path(__file__).parents[2] / "data/catalogs"


@pytest.fixture()
def tmp_catalog_files(tmpdir):
    _base_url = Path(tmpdir) / "test_catalog"
    for version in ["v0.1.0", "v0.2.0", "v1.0.0"]:
        catalog_path = _base_url / version / "data_catalog.yml"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text("meta:\n  version: " + version)
    return _base_url


@pytest.fixture()
def tmp_catalog_class(tmp_catalog_files) -> type[PredefinedCatalog]:
    _base_url = tmp_catalog_files

    # write registry file
    create_registry_file(_base_url)

    class TestCatalog(PredefinedCatalog):
        name = "test_catalog"
        base_url = str(_base_url)

    return TestCatalog


def test_predefined_catalog(tmp_catalog_class, tmpdir):
    catalog = tmp_catalog_class(format_version="v0", cache_dir=Path(tmpdir) / "cache")
    assert catalog.name == "test_catalog"
    assert catalog._format_version == "v0"
    assert catalog._pooch is None
    assert catalog._versions is None
    assert (
        catalog.get_catalog_file("v0.1.0")
        == catalog._cache_dir / "test_catalog" / "v0.1.0" / "data_catalog.yml"
    )
    assert (
        catalog.get_catalog_file()
        == catalog._cache_dir / "test_catalog" / "v0.2.0" / "data_catalog.yml"
    )
    assert catalog.versions == ["v0.1.0", "v0.2.0"]

    # remove source and cached registry file and check if one is created based on
    # the cached catalog files
    registry_path = catalog._cache_dir / "test_catalog" / "registry.txt"
    registry_path.unlink()
    Path(tmp_catalog_class.base_url, "registry.txt").unlink()
    catalog._load_registry_file(overwrite=True)
    assert registry_path.exists()

    with pytest.raises(ValueError, match="Version v0.0.0 not found"):
        catalog.get_catalog_file("v0.0.0")


def test_create_registry_file(tmpdir, tmp_catalog_files):
    # test create registry file
    root = tmp_catalog_files
    create_registry_file(root)
    registry_path = root / "registry.txt"
    assert registry_path.exists()

    # no catalog files
    with pytest.raises(FileNotFoundError):
        create_registry_file(Path(tmpdir, "not_existing"))

    # create a dummy catalog file with version folder
    cat_path = root / "data_catalog.yml"
    cat_path.write_text("meta:\n  version: v0.1.0")
    with pytest.raises(ValueError, match="No valid version found"):
        create_registry_file(root)


def test_get_versions_artifacts():
    versions = ArtifactDataCatalog().versions
    assert len(versions) > 0
    assert "v0.0.8" in versions


def test_catalog_versions(cat_root: Path, tmpdir):
    # assert all subdirs are catalogs and have a versions.yml file
    catalogs = filter(
        lambda dir: "__pycache__" not in str(dir),
        filter(lambda dir: dir.is_dir(), cat_root.iterdir()),
    )
    for cat_dir in catalogs:
        registry_file = cat_dir / "registry.txt"
        assert registry_file.exists()
        tmp_registry_file = Path(tmpdir) / f"{cat_dir.name}_registry.txt"
        create_registry_file(cat_dir, tmp_registry_file)
        # check if both registry files (incl hashes) are the same
        with open(registry_file, "r") as f:
            registry = sorted(f.readlines())
        with open(tmp_registry_file, "r") as f:
            tmp_registry = sorted(f.readlines())
        assert registry == tmp_registry
